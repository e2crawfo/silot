import tensorflow as tf
import numpy as np
from functools import partial
import itertools
import shutil
import os

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from matplotlib.colors import to_rgb

from dps import cfg
from dps.utils import Param, Config, AttrDict
from dps.utils.tf import build_scheduled_value, tf_shape, RenderHook
from dps.tf.updater import DataManager

from auto_yolo.models.core import AP, Updater as _Updater, Evaluator

from silot.core import VideoNetwork, MOTMetrics

from sqair.sqair_modules import FastPropagate, Propagate, FastDiscover, Discover
from sqair.core import DiscoveryCore, PropagationCore
from sqair.modules import (
    Encoder, StochasticTransformParam, FixedStepsPredictor, StepsPredictor,
    Decoder, AIRDecoder, AIREncoder
)
from sqair.seq import SequentialAIR
from sqair.propagate import make_prior
from sqair import index
from sqair import targets
from sqair import ops
from sqair.neural import MLP


class SQAIRUpdater(_Updater):
    VI_TARGETS = 'iwae reinforce'.split()
    TARGETS = VI_TARGETS

    lr_schedule = Param()
    l2_schedule = Param()
    output_std = Param()
    k_particles = Param()
    debug = Param()

    def __init__(self, env, scope=None, **kwargs):
        self.lr_schedule = build_scheduled_value(self.lr_schedule, "lr")
        self.l2_schedule = build_scheduled_value(self.l2_schedule, "l2_weight")

        super().__init__(env, scope=scope, **kwargs)

    def resample(self, *args, axis=-1):
        """ Just resample, but potentially applied to several args. """
        res = list(args)

        if self.k_particles > 1:
            for i, arg in enumerate(res):
                res[i] = self._resample(arg, axis)

        if len(res) == 1:
            res = res[0]

        return res

    def _resample(self, arg, axis=-1):
        iw_sample_idx = self.iw_resampling_idx + tf.range(self.batch_size) * self.k_particles

        resampled = index.gather_axis(arg, iw_sample_idx, axis)

        shape = arg.shape.as_list()
        shape[axis] = self.batch_size

        resampled.set_shape(shape)

        return resampled

    def _log_resampled(self, name):
        tensor = self.tensors[name + "_per_sample"]
        self.tensors['resampled_' + name] = self._resample(tensor)
        self.recorded_tensors[name] = self._imp_weighted_mean(tensor)
        self.tensors[name] = self.recorded_tensors[name]

    def _imp_weighted_mean(self, tensor):
        tensor = tf.reshape(tensor, (-1, self.batch_size, self.k_particles))
        tensor = tf.reduce_mean(tensor, 0)
        return tf.reduce_mean(self.importance_weights * tensor * self.k_particles)

    def compute_validation_pixelwise_mean(self, data):
        sess = tf.get_default_session()

        mean = None
        n_points = 0
        feed_dict = self.data_manager.do_val()

        while True:
            try:
                inp = sess.run(data["image"], feed_dict=feed_dict)
            except tf.errors.OutOfRangeError:
                break

            n_new = inp.shape[0] * inp.shape[1]
            if mean is None:
                mean = np.mean(inp, axis=(0, 1))
            else:
                mean = mean * (n_points / (n_points + n_new)) + np.sum(inp, axis=(0, 1)) / (n_points + n_new)
            n_points += n_new
        return mean

    def _build_graph(self):
        self.data_manager = DataManager(datasets=self.env.datasets)
        self.data_manager.build_graph()

        if self.k_particles <= 1:
            raise Exception("`k_particles` must be > 1.")

        data = self.data_manager.iterator.get_next()
        data['mean_img'] = self.compute_validation_pixelwise_mean(data)
        self.batch_size = cfg.batch_size

        self.tensors = AttrDict()

        network_outputs = self.network(data, self.data_manager.is_training)

        network_tensors = network_outputs["tensors"]
        network_recorded_tensors = network_outputs["recorded_tensors"]
        network_losses = network_outputs["losses"]
        assert not network_losses

        self.tensors.update(network_tensors)

        self.recorded_tensors = recorded_tensors = dict(global_step=tf.train.get_or_create_global_step())
        self.recorded_tensors.update(network_recorded_tensors)

        # --- values for training ---

        log_weights = tf.reduce_sum(self.tensors.log_weights_per_timestep, 0)
        self.log_weights = tf.reshape(log_weights, (self.batch_size, self.k_particles))

        self.elbo_vae = tf.reduce_mean(self.log_weights)
        self.elbo_iwae_per_example = targets.iwae(self.log_weights)
        self.elbo_iwae = tf.reduce_mean(self.elbo_iwae_per_example)

        self.normalised_elbo_vae = self.elbo_vae / tf.to_float(self.network.dynamic_n_frames)
        self.normalised_elbo_iwae = self.elbo_iwae / tf.to_float(self.network.dynamic_n_frames)

        self.importance_weights = tf.stop_gradient(tf.nn.softmax(self.log_weights, -1))
        self.ess = ops.ess(self.importance_weights, average=True)
        self.iw_distrib = tf.distributions.Categorical(probs=self.importance_weights)
        self.iw_resampling_idx = self.iw_distrib.sample()

        # --- count accuracy ---

        if "annotations" in data:
            gt_num_steps = tf.transpose(self.tensors.n_valid_annotations, (1, 0))[:, :, None]
            num_steps_per_sample = tf.reshape(
                self.tensors.num_steps_per_sample, (-1, self.batch_size, self.k_particles))
            count_1norm = tf.abs(num_steps_per_sample - tf.cast(gt_num_steps, tf.float32))
            count_error = tf.cast(count_1norm > 0.5, tf.float32)

            self.recorded_tensors.update(
                count_1norm=self._imp_weighted_mean(count_1norm),
                count_error=self._imp_weighted_mean(count_error),
            )

        # --- losses ---

        log_probs = tf.reduce_sum(self.tensors.discrete_log_prob, 0)
        target = targets.vimco(self.log_weights, log_probs, self.elbo_iwae_per_example)

        target /= tf.to_float(self.network.dynamic_n_frames)
        loss_l2 = targets.l2_reg(self.l2_schedule)
        target += loss_l2

        # --- train op ---

        tvars = tf.trainable_variables()
        pure_gradients = tf.gradients(target, tvars)

        clipped_gradients = pure_gradients
        if self.max_grad_norm is not None and self.max_grad_norm > 0.0:
            clipped_gradients, _ = tf.clip_by_global_norm(pure_gradients, self.max_grad_norm)

        grads_and_vars = list(zip(clipped_gradients, tvars))

        lr = self.lr_schedule
        valid_lr = tf.Assert(
            tf.logical_and(tf.less(lr, 1.0), tf.less(0.0, lr)),
            [lr], name="valid_learning_rate")

        opt = tf.train.RMSPropOptimizer(self.lr_schedule, momentum=.9)

        with tf.control_dependencies([valid_lr]):
            self.train_op = opt.apply_gradients(grads_and_vars, global_step=None)

        recorded_tensors.update(
            grad_norm_pure=tf.global_norm(pure_gradients),
            grad_norm_processed=tf.global_norm(clipped_gradients),
            grad_lr_norm=lr * tf.global_norm(clipped_gradients),
        )

        # gvs = opt.compute_gradients(target)
        # assert len(gvs) == len(tf.trainable_variables())
        # for g, v in gvs:
        #     assert g is not None, 'Gradient for variable {} is None'.format(v)

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     self.train_op = opt.apply_gradients(gvs)

        # --- record ---

        self.tensors['mse_per_sample'] = tf.reduce_mean(
            (self.tensors['processed_image'] - self.tensors['canvas']) ** 2, (0, 2, 3, 4))
        self.raw_mse = tf.reduce_mean(self.tensors['mse_per_sample'])

        self._log_resampled('mse')
        self._log_resampled('data_ll')
        self._log_resampled('log_p_z')
        self._log_resampled('log_q_z_given_x')
        self._log_resampled('kl')

        try:
            self._log_resampled('num_steps')
            self._log_resampled('num_disc_steps')
            self._log_resampled('num_prop_steps')
        except AttributeError:
            pass

        recorded_tensors.update(
            raw_mse=self.raw_mse,
            elbo_vae=self.elbo_vae,
            elbo_iwae=self.elbo_iwae,
            normalised_elbo_vae=self.normalised_elbo_vae,
            normalised_elbo_iwae=self.normalised_elbo_iwae,
            ess=self.ess,
            loss=target,
            target=target,
            loss_l2=loss_l2,
        )
        self.train_records = {}

        # --- recorded values ---

        intersection = recorded_tensors.keys() & self.network.eval_funcs.keys()
        assert not intersection, "Key sets have non-zero intersection: {}".format(intersection)

        # --- for rendering and eval ---
        resampled_names = (
            'obj_id canvas glimpse presence_prob presence presence_logit '
            'where_coords num_steps_per_sample'.split())

        for name in resampled_names:
            try:
                resampled_tensor = self.resample(self.tensors[name], axis=1)
                permutation = [1, 0] + list(range(2, len(resampled_tensor.shape)))
                self.tensors['resampled_' + name] = tf.transpose(resampled_tensor, permutation)
            except AttributeError:
                pass

        # For running functions, during evaluation, that are not implemented in tensorflow
        self.evaluator = Evaluator(self.network.eval_funcs, self.tensors, self)


class SQAIR_AP(AP):
    keys_accessed = (
        ["resampled_" + name for name in "where_coords presence_prob num_steps_per_sample".split()]
        + "annotations n_annotations".split()
    )

    def _process_data(self, tensors, updater):
        obj = tensors["resampled_presence_prob"]
        predicted_n_digits = tensors['resampled_num_steps_per_sample']
        w, h, x, y = np.split(tensors['resampled_where_coords'], 4, axis=3)

        image_shape = (updater.image_height, updater.image_width)
        top, left, height, width = coords_to_pixel_space(y, x, h, w, image_shape, top_left=True)

        annotations = tensors["annotations"]
        n_annotations = tensors["n_annotations"]

        return obj, predicted_n_digits, top, left, height, width, annotations, n_annotations


class SQAIR_MOTMetrics(MOTMetrics):
    keys_accessed = (
        ["resampled_" + name for name in "obj_id where_coords num_steps_per_sample".split()]
        + "annotations n_annotations".split()
    )

    def _process_data(self, tensors, updater):
        pred_n_objects = tensors['resampled_num_steps_per_sample']
        obj_id = tensors['resampled_obj_id']

        shape = obj_id.shape
        obj = (obj_id != -1).astype('i')

        w, h, x, y = np.split(tensors['resampled_where_coords'], 4, axis=3)
        w = w.reshape(shape)
        h = h.reshape(shape)
        x = x.reshape(shape)
        y = y.reshape(shape)

        image_shape = (updater.image_height, updater.image_width)
        top, left, height, width = coords_to_pixel_space(y, x, h, w, image_shape, top_left=True)

        return obj, pred_n_objects, obj_id, top, left, height, width


class SQAIR_Prior_AP(SQAIR_AP):
    def get_feed_dict(self, updater):
        feed_dict = super().get_feed_dict(updater)
        feed_dict[updater.network._prior_start_step] = self.start_frame
        return feed_dict


class SQAIR_Prior_MOTMetrics(SQAIR_MOTMetrics):
    def get_feed_dict(self, updater):
        feed_dict = super().get_feed_dict(updater)
        feed_dict[updater.network._prior_start_step] = self.start_frame
        return feed_dict


class SQAIR(VideoNetwork):
    disc_prior_type = Param()
    prop_prior_type = Param()
    disc_step_bias = Param()
    prop_step_bias = Param()
    prop_prior_step_bias = Param()
    step_success_prob = Param()
    output_scale = Param()
    output_std = Param()
    object_shape = Param()
    fixed_presence = Param()

    rnn_class = Param()
    time_rnn_class = Param()
    prior_rnn_class = Param()
    n_what = Param()
    debug = Param()
    masked_glimpse = Param()

    noisy = Param()
    eval_noisy = Param()

    fast_discovery = Param()
    fast_propagation = Param()

    n_hidden = Param()
    n_layers = Param()

    transform_var_bias = Param()
    n_objects = Param()
    scale_prior = Param()
    rec_where_prior = Param()

    k_particles = Param()
    scale_bounds = Param()

    prior_start_step = Param()
    eval_prior_start_step = Param()

    # Don't think we need these for this network
    attr_prior_mean = None
    attr_prior_std = None

    needs_background = False

    def std_nonlinearity(self, std_logit):
        # return tf.exp(std)
        return (
            self._noisy * 2 * tf.nn.sigmoid(tf.clip_by_value(std_logit, -10, 10))
            + (1 - self._noisy) * tf.zeros_like(std_logit)
        )

    def z_nonlinearity(self, z_logit):
        return tf.nn.sigmoid(tf.clip_by_value(z_logit, -10, 10))

    @property
    def _noisy(self):
        return (
            self.float_is_training * tf.to_float(self.noisy)
            + (1 - self.float_is_training) * tf.to_float(self.eval_noisy)
        )

    def __init__(self, env, updater, scope=None, **kwargs):
        self._prior_start_step = tf.constant(self.prior_start_step, tf.int32)

        ap_iou_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.eval_funcs = {"AP_at_point_{}".format(int(10 * v)): SQAIR_AP(v) for v in ap_iou_values}
        self.eval_funcs["AP"] = SQAIR_AP(ap_iou_values)
        self.eval_funcs["AP_train"] = SQAIR_AP(ap_iou_values, is_training=True)
        self.eval_funcs["MOT"] = SQAIR_MOTMetrics()
        self.eval_funcs["MOT_train"] = SQAIR_MOTMetrics(is_training=True)

        self.eval_funcs["prior_AP"] = SQAIR_Prior_AP(ap_iou_values, start_frame=self.eval_prior_start_step)
        self.eval_funcs["prior_MOT"] = SQAIR_Prior_MOTMetrics(start_frame=self.eval_prior_start_step)

        super().__init__(env, updater, scope=scope, **kwargs)

    def build_representation(self):
        processed_image = index.tile_input_for_iwae(
            tf.transpose(self.inp, (1, 0, 2, 3, 4)), self.k_particles, with_time=True)
        shape = list(tf_shape(processed_image))
        shape[1] = cfg.batch_size * self.k_particles
        processed_image = tf.reshape(processed_image, shape)

        self._tensors.update(
            processed_image=processed_image,
            mean_img=self.data['mean_img'],
        )

        _, _, *img_size = processed_image.shape.as_list()

        layers = [self.n_hidden] * self.n_layers

        def glimpse_encoder():
            return AIREncoder(img_size, self.object_shape, self.n_what, Encoder(layers),
                              masked_glimpse=self.masked_glimpse, debug=self.debug)

        steps_pred_hidden = self.n_hidden / 2

        transform_estimator = partial(StochasticTransformParam, layers, self.transform_var_bias)

        if self.fixed_presence:
            disc_steps_predictor = partial(FixedStepsPredictor, discovery=True)
        else:
            disc_steps_predictor = partial(StepsPredictor, steps_pred_hidden, self.disc_step_bias)

        if cfg.build_input_encoder is None:
            input_encoder = partial(Encoder, layers)
        else:
            input_encoder = cfg.build_input_encoder

        _input_encoder = input_encoder()

        T, B, *rest = tf_shape(processed_image)
        images = tf.reshape(processed_image, (T*B, *rest))
        encoded_input = _input_encoder(images)
        encoded_input = tf.reshape(encoded_input, (T, B, *tf_shape(encoded_input)[1:]))

        with tf.variable_scope('discovery'):

            discover_cell = DiscoveryCore(
                processed_image, encoded_input, self.object_shape, self.n_what, self.rnn_class(self.n_hidden),
                glimpse_encoder, transform_estimator, disc_steps_predictor, debug=self.debug)

            if self.fast_discovery:
                object_state_predictor = MLP([256, 256, self.n_hidden])
                discover = FastDiscover(
                    object_state_predictor, self.n_objects, discover_cell,
                    step_success_prob=self.step_success_prob, where_mean=[*self.scale_prior, 0, 0],
                    disc_prior_type=self.disc_prior_type, rec_where_prior=self.rec_where_prior)

            else:
                discover = Discover(
                    self.n_objects, discover_cell, step_success_prob=self.step_success_prob,
                    where_mean=[*self.scale_prior, 0, 0], disc_prior_type=self.disc_prior_type,
                    rec_where_prior=self.rec_where_prior)

        with tf.variable_scope('propagation'):
            # Prop cell should have a different rnn cell but should share all other estimators
            glimpse_encoder = lambda: discover_cell._glimpse_encoder
            transform_estimator = partial(StochasticTransformParam, layers, self.transform_var_bias)

            if self.fixed_presence:
                prop_steps_predictor = partial(FixedStepsPredictor, discovery=False)
            else:
                prop_steps_predictor = partial(StepsPredictor, steps_pred_hidden, self.prop_step_bias)

            prior_rnn = self.prior_rnn_class(self.n_hidden)
            propagation_prior = make_prior(self.prop_prior_type, self.n_what, prior_rnn, self.prop_prior_step_bias)

            propagate_rnn_cell = self.rnn_class(self.n_hidden)
            temporal_rnn_cell = self.time_rnn_class(self.n_hidden)

            propagation_cell = PropagationCore(processed_image, encoded_input, self.object_shape, self.n_what,
                                               propagate_rnn_cell, glimpse_encoder, transform_estimator,
                                               prop_steps_predictor, temporal_rnn_cell, debug=self.debug)

            if self.fast_propagation:
                propagate = FastPropagate(propagation_cell, propagation_prior)
            else:
                propagate = Propagate(propagation_cell, propagation_prior)

        with tf.variable_scope('decoder'):
            glimpse_decoder = partial(Decoder, layers, output_scale=self.output_scale)
            decoder = AIRDecoder(img_size, self.object_shape, glimpse_decoder,
                                 batch_dims=2,
                                 mean_img=self._tensors.mean_img,
                                 output_std=self.output_std,
                                 scale_bounds=self.scale_bounds)

        with tf.variable_scope('sequence'):
            time_cell = self.time_rnn_class(self.n_hidden)

            sequence_apdr = SequentialAIR(
                self.n_objects, self.object_shape, discover, propagate,
                time_cell, decoder, prior_start_step=self._prior_start_step)

        outputs = sequence_apdr(processed_image)
        outputs['where_coords'] = decoder._transformer.to_coords(outputs['where'])

        self._tensors.update(outputs)


def coords_to_pixel_space(y, x, h, w, image_shape, top_left):
    h = h * (image_shape[0] + 1)
    w = w * (image_shape[1] + 1)
    y = 0.5 * image_shape[0] * (y + 1.) - 0.5
    x = 0.5 * image_shape[1] * (x + 1.) - 0.5

    if top_left:
        y = y - h / 2
        x = x - w / 2

    return y, x, h, w


class SQAIR_RenderHook(RenderHook):
    N = 16
    linewidth = 0.5
    on_color = np.array(to_rgb("xkcd:azure"))
    off_color = np.array(to_rgb("xkcd:red"))
    gt_color = "xkcd:yellow"
    _BBOX_COLORS = 'rgbymcw'
    fig_scale = 1.5

    def build_fetches(self, updater):
        resampled_names = (
            'obj_id canvas glimpse presence_prob presence presence_logit '
            'where_coords num_steps_per_sample'.split())
        fetches = (
            ["resampled_" + name for name in resampled_names]
        )

        if "n_annotations" in updater.tensors:
            fetches.extend(" annotations n_annotations".split())

        fetches.append("inp")

        return fetches

    def __call__(self, updater):
        fetched = self._fetch(updater)
        fetched = Config(fetched)
        self._prepare_fetched(updater, fetched)
        o = AttrDict(**fetched)

        N, T, image_height, image_width, _ = o.inp.shape

        # --- static ---

        fig_width = 2 * N
        fig_height = T
        figsize = self.fig_scale * np.asarray((fig_width, fig_height))
        fig, axes = plt.subplots(fig_height, fig_width, figsize=figsize)
        fig.suptitle("n_updates={}".format(updater.n_updates), fontsize=20, fontweight='bold')
        axes = axes.reshape((fig_height, fig_width))

        unique_ids = [int(i) for i in np.unique(o.obj_id)]
        if unique_ids[0] < 0:
            unique_ids = unique_ids[1:]

        color_by_id = {i: c for i, c in zip(unique_ids, itertools.cycle(self._BBOX_COLORS))}
        color_by_id[-1] = 'k'

        cmap = self._cmap(o.inp)
        for t, ax in enumerate(axes):
            for n in range(N):
                pres_time = o.presence[n, t, :]
                obj_id_time = o.obj_id[n, t, :]
                self.imshow(ax[2 * n], o.inp[n, t], cmap=cmap)

                n_obj = str(int(np.round(pres_time.sum())))
                id_string = ('{}{}'.format(color_by_id[int(i)], i) for i in o.obj_id[n, t] if i > -1)
                id_string = ', '.join(id_string)
                title = '{}: {}'.format(n_obj, id_string)

                ax[2 * n + 1].set_title(title, fontsize=6 * self.fig_scale)
                self.imshow(ax[2 * n + 1], o.canvas[n, t], cmap=cmap)
                for i, (p, o_id) in enumerate(zip(pres_time, obj_id_time)):
                    c = color_by_id[int(o_id)]
                    if p > .5:
                        r = patches.Rectangle(
                            (o.left[n, t, i], o.top[n, t, i]), o.width[n, t, i], o.height[n, t, i],
                            linewidth=self.linewidth, edgecolor=c, facecolor='none')
                        ax[2 * n + 1].add_patch(r)

        for n in range(N):
            axes[0, 2 * n].set_ylabel('gt #{:d}'.format(n))
            axes[0, 2 * n + 1].set_ylabel('rec #{:d}'.format(n))

        for ax in axes.flatten():
            # ax.grid(False)
            # ax.set_xticks([])
            # ax.set_yticks([])
            ax.set_axis_off()

        self.savefig('static', fig, updater)

        # --- moving ---

        fig_width = 2 * N
        n_objects = o.obj_id.shape[2]
        fig_height = n_objects + 2
        figsize = self.fig_scale * np.asarray((fig_width, fig_height))
        fig, axes = plt.subplots(fig_height, fig_width, figsize=figsize)
        title_text = fig.suptitle('', fontsize=10)
        axes = axes.reshape((fig_height, fig_width))

        def func(t):
            title_text.set_text("t={}, n_updates={}".format(t, updater.n_updates))

            for i in range(N):
                self.imshow(axes[0, 2*i], o.inp[i, t], cmap=cmap, vmin=0, vmax=1)
                self.imshow(axes[1, 2*i], o.canvas[i, t], cmap=cmap, vmin=0, vmax=1)

                for j in range(n_objects):
                    if o.presence[i, t, j] > .5:
                        c = color_by_id[int(o.obj_id[i, t, j])]
                        r = patches.Rectangle(
                            (o.left[i, t, j], o.top[i, t, j]), o.width[i, t, j], o.height[i, t, j],
                            linewidth=self.linewidth, edgecolor=c, facecolor='none')
                        axes[1, 2*i].add_patch(r)

                    ax = axes[2+j, 2*i]

                    self.imshow(ax, o.presence[i, t, j] * o.glimpse[i, t, j], cmap=cmap)
                    title = '{:d} with p({:d}) = {:.02f}, id = {}'.format(
                        int(o.presence[i, t, j]), i + 1, o.presence_prob[i, t, j], o.obj_id[i, t, j])
                    ax.set_title(title, fontsize=4 * self.fig_scale)

                    if o.presence[i, t, j] > .5:
                        c = color_by_id[int(o.obj_id[i, t, j])]
                        for spine in 'bottom top left right'.split():
                            ax.spines[spine].set_color(c)
                            ax.spines[spine].set_linewidth(2.)

            for ax in axes.flatten():
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])

            axes[0, 0].set_ylabel('ground-truth')
            axes[1, 0].set_ylabel('reconstruction')

            for j in range(n_objects):
                axes[j+2, 0].set_ylabel('glimpse #{}'.format(j + 1))

        plt.subplots_adjust(left=0.02, right=.98, top=.95, bottom=0.02, wspace=0.1, hspace=0.15)

        anim = animation.FuncAnimation(fig, func, frames=T, interval=500)

        path = self.path_for('moving', updater, ext="mp4")
        anim.save(path, writer='ffmpeg', codec='hevc', extra_args=['-preset', 'ultrafast'])

        plt.close(fig)

        shutil.copyfile(
            path,
            os.path.join(
                os.path.dirname(path),
                'latest_stage{:0>4}.mp4'.format(updater.stage_idx)))

    def _cmap(self, obs, with_time=True):
        ndims = len(obs.shape)
        cmap = None
        if ndims == (3 + with_time) or (ndims == (4 + with_time) and obs.shape[-1] == 1):
            cmap = 'gray'
        return cmap

    @staticmethod
    def normalize_images(images):
        mx = images.reshape(*images.shape[:-3], -1).max(axis=-1)
        return images / mx[..., None, None, None]

    def _prepare_fetched(self, updater, fetched):
        inp = fetched['inp']
        output = fetched['resampled_canvas']
        prediction = fetched.get("prediction", None)
        targets = fetched.get("targets", None)

        N, T, image_height, image_width, _ = inp.shape

        w, h, x, y = np.split(fetched['resampled_where_coords'], 4, axis=3)

        top, left, height, width = coords_to_pixel_space(y, x, h, w, (image_height, image_width), top_left=True)

        n_annotations = fetched.get("n_annotations", np.zeros(N, dtype='i'))
        annotations = fetched.get("annotations", None)

        diff = self.normalize_images(np.abs(inp - output).mean(axis=-1, keepdims=True))
        squared_diff = self.normalize_images(((inp - output)**2).mean(axis=-1, keepdims=True))

        fetched.update(
            prediction=prediction,
            targets=targets,
            top=top,
            left=left,
            height=height,
            width=width,
            n_annotations=n_annotations,
            annotations=annotations,
            diff=diff,
            squared_diff=squared_diff,
        )

        new = {}
        for k, v in fetched.items():
            if k.startswith('resampled_'):
                n_chars = len('resampled_')
                new[k[n_chars:]] = v
        fetched.update(new)
