import tensorflow as tf
import numpy as np
from functools import partial
from orderedattrdict import AttrDict

from dps import cfg
from dps.utils import Param
from dps.utils.tf import (
    build_gradient_train_op, ScopedFunction, build_scheduled_value, FIXED_COLLECTION, tf_shape)
from dps.updater import DataManager

from auto_yolo.models.core import mAP, Updater as _Updater

from spair_video.core import VideoNetwork

from sqair.sqair_modules import Propagate, Discover
from sqair.core import DiscoveryCore, PropagationCore
from sqair.modules import Encoder, StochasticTransformParam, StepsPredictor, Decoder, AIRDecoder, AIREncoder
from sqair.seq import SequentialAIR
from sqair.propagate import make_prior, SequentialSSM
from sqair import index
from sqair import targets
from sqair import ops


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

    def img_summaries(self):
        recs = tf.cast(tf.round(tf.clip_by_value(self.resampled_canvas, 0., 1.) * 255), tf.uint8)
        rec = tf.summary.image('reconstructions', recs[0])
        inpt = tf.summary.image('inputs', self.obs[0])

        return tf.summary.merge([rec, inpt])

    def compute_validation_pixelwise_mean(self):
        sess = tf.get_default_session()

        mean = None
        n_points = 0
        feed_dict = self.data_manager.do_val()

        while True:
            try:
                inp = sess.run(self.data["image"], feed_dict=feed_dict)
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
        self.data_manager = DataManager(self.env.datasets['train'],
                                        self.env.datasets['val'],
                                        self.env.datasets['test'],
                                        cfg.batch_size)
        self.data_manager.build_graph()

        data = self.data_manager.iterator.get_next()

        obs = data["image"]
        self.batch_size = cfg.batch_size

        shape = list(tf_shape(obs))
        shape[0] = cfg.batch_size
        obs.set_shape(shape)

        obs = tf.transpose(obs, (1, 0, 2, 3, 4))

        self.tiled_obs = index.tile_input_for_iwae(obs, self.k_particles, with_time=True)
        data["tiled_obs"] = self.tiled_obs

        self.data = data

        network_outputs = self.network(data, self.data_manager.is_training)

        network_tensors = network_outputs["tensors"]
        network_recorded_tensors = network_outputs["recorded_tensors"]
        network_losses = network_outputs["losses"]
        assert not network_losses

        self.tensors = network_outputs

        self.recorded_tensors = recorded_tensors = dict(global_step=tf.train.get_or_create_global_step())
        self.recorded_tensors.update(network_recorded_tensors)

        # --- loss ---

        log_weights = tf.reduce_sum(self.outputs.log_weights_per_timestep, 0)
        self.log_weights = tf.reshape(log_weights, (self.batch_size, self.k_particles))

        self.elbo_vae = tf.reduce_mean(self.log_weights)
        self.elbo_iwae_per_example = targets.iwae(self.log_weights)
        self.elbo_iwae = tf.reduce_mean(self.elbo_iwae_per_example)

        self.normalised_elbo_vae = self.elbo_vae / tf.to_float(self.n_timesteps)
        self.normalised_elbo_iwae = self.elbo_iwae / tf.to_float(self.n_timesteps)

        self.importance_weights = tf.stop_gradient(tf.nn.softmax(self.log_weights, -1))
        self.ess = ops.ess(self.importance_weights, average=True)
        self.iw_distrib = tf.distributions.Categorical(probs=self.importance_weights)
        self.iw_resampling_idx = self.iw_distrib.sample()

        # --- losses ---

        if hasattr(self, 'discrete_log_prob'):
            log_probs = tf.reduce_sum(self.discrete_log_prob, 0)
            target = targets.vimco(self.log_weights, log_probs, self.elbo_iwae_per_example)
        else:
            target = -self.elbo_iwae

        loss_reconstruction = target / tf.to_float(self.n_timesteps)
        loss_l2 = targets.l2_reg(self.l2_schedule)
        self.loss = loss_reconstruction + loss_l2

        # --- train op ---

        opt = tf.train.RMSPropOptimizer(self.lr_schedule, momentum=.9)

        gvs = opt.compute_gradients(target)

        assert len(gvs) == len(tf.trainable_variables())
        for g, v in gvs:
            assert g is not None, 'Gradient for variable {} is None'.format(v)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = opt.apply_gradients(gvs)

        # --- record ---

        self.tensors['mse_per_sample'] = tf.reduce_mean(
            (self.tiled_obs - self.tensors['canvas']) ** 2, (0, 2, 3, 4))
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
            log_weights=self.log_weights,
            elbo_vae=self.elbo_vae,
            elbo_iwae_per_example=self.elbo_iwae_per_example,
            elbo_iwae=self.elbo_iwae,
            normalised_elbo_vae=self.normalised_elbo_vae,
            normalised_elbo_iwae=self.normalised_elbo_iwae,
            importance_weights=self.importance_weights,
            ess=self.ess,
            iw_distrib=self.iw_distrib,
            iw_resampling_idx=self.iw_resampling_idx,
            loss=self.loss,
            loss_reconstruction=loss_reconstruction,
            loss_l2=loss_l2,
        )

        # --- recorded values ---

        intersection = recorded_tensors.keys() & network_recorded_tensors.keys()
        assert not intersection, "Key sets have non-zero intersection: {}".format(intersection)
        recorded_tensors.update(network_recorded_tensors)

        intersection = recorded_tensors.keys() & self.network.eval_funcs.keys()
        assert not intersection, "Key sets have non-zero intersection: {}".format(intersection)

        # For running functions, during evaluation, that are not implemented in tensorflow
        self.evaluator = Evaluator(self.network.eval_funcs, network_tensors, self)

        # TODO vvv count accuracy
        # if self.gt_presence is not None:
        #     self.gt_num_steps = tf.reduce_sum(self.gt_presence, -1)

        #     num_steps_per_sample = tf.reshape(self.num_steps_per_sample, (-1, self.batch_size, self.k_particles))
        #     gt_num_steps = tf.expand_dims(self.gt_num_steps, -1)

        #     self.num_step_accuracy_per_example = tf.to_float(tf.equal(gt_num_steps, num_steps_per_sample))
        #     self.raw_num_step_accuracy = tf.reduce_mean(self.num_step_accuracy_per_example)
        #     self.num_step_accuracy = self._imp_weighted_mean(self.num_step_accuracy_per_example)
        #     tf.summary.scalar('num_step_acc', self.num_step_accuracy)



        # # For rendering
        # resampled_names = 'obj_id canvas glimpse presence_prob presence presence_logit where'.split()
        # for name in resampled_names:
        #     try:
        #         setattr(self, 'resampled_' + name, self.resample(getattr(self, name), axis=1))
        #     except AttributeError:
        #         pass


class SQAIR_AP(object):
    """ TODO: this whole class """
    keys_accessed = "scale shift predicted_n_digits annotations n_annotations"

    def __init__(self, iou_threshold=None):
        if iou_threshold is not None:
            try:
                iou_threshold = list(iou_threshold)
            except (TypeError, ValueError):
                iou_threshold = [float(iou_threshold)]
        self.iou_threshold = iou_threshold

    def __call__(self, _tensors, updater):
        network = updater.network
        w, h = np.split(_tensors['scale'], 2, axis=2)
        x, y = np.split(_tensors['shift'], 2, axis=2)
        predicted_n_digits = _tensors['predicted_n_digits']
        annotations = _tensors["annotations"]
        n_annotations = _tensors["n_annotations"]

        batch_size = w.shape[0]

        transformed_x = 0.5 * (x + 1.)
        transformed_y = 0.5 * (y + 1.)

        height = h * network.image_height
        width = w * network.image_width

        top = network.image_height * transformed_y - height / 2
        left = network.image_width * transformed_x - width / 2

        bottom = top + height
        right = left + width

        ground_truth_boxes = []
        predicted_boxes = []

        for idx in range(batch_size):
            _a = [
                [0, *rest]
                for (valid, cls, *rest), _
                in zip(annotations[idx], range(n_annotations[idx]))
                if valid]
            ground_truth_boxes.append(_a)

            _predicted_boxes = []

            for t in range(predicted_n_digits[idx]):
                _predicted_boxes.append(
                    [0, 1,
                     top[idx, t, 0],
                     bottom[idx, t, 0],
                     left[idx, t, 0],
                     right[idx, t, 0]])

            predicted_boxes.append(_predicted_boxes)

        return mAP(
            predicted_boxes, ground_truth_boxes, n_classes=1,
            iou_threshold=self.iou_threshold)


class SQAIR(VideoNetwork):
    disc_prior_type = Param()
    prop_prior_type = Param()
    disc_step_bias = Param()
    prop_step_bias = Param()
    prop_prior_step_bias = Param()
    step_success_prob = Param()
    sample_from_prior = Param()
    output_scale = Param()
    output_std = Param()
    glimpse_size = Param()

    rnn_class = Param()
    time_rnn_class = Param()
    prior_rnn_class = Param()
    n_what = Param()
    debug = Param()
    masked_glimpse = Param()

    n_hidden = Param()
    n_layers = Param()

    transform_var_bias = Param()
    n_steps_per_image = Param()
    scale_prior = Param()
    rec_where_prior = Param()

    sample_from_prior = Param()

    # Don't think we need these for this network
    attr_prior_mean = None
    attr_prior_std = None
    noisy = None

    def __init__(self, env, updater, scope=None, **kwargs):
        # ap_iou_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # self.eval_funcs = {"AP_at_point_{}".format(int(10 * v)): SQAIR_AP(v) for v in ap_iou_values}
        # self.eval_funcs["AP"] = SQAIR_AP(ap_iou_values)
        super().__init__(env, updater, scope=scope, **kwargs)

    def _call(self, data, is_training):
        self.data = data
        inp = data["tiled_obs"]

        self._tensors = AttrDict(
            inp=inp,
            is_training=is_training,
            float_is_training=tf.to_float(is_training),
            batch_size=tf.shape(inp)[1],
        )

        # if "annotations" in data:
        #     self._tensors.update(
        #         annotations=data["annotations"]["data"],
        #         n_annotations=data["annotations"]["shapes"][:, 1],
        #         n_valid_annotations=tf.to_int32(
        #             tf.reduce_sum(
        #                 data["annotations"]["data"][:, :, :, 0]
        #                 * tf.to_float(data["annotations"]["mask"][:, :, :, 0]),
        #                 axis=2
        #             )
        #         )
        #     )

        # if "label" in data:
        #     self._tensors.update(
        #         targets=data["label"],
        #     )

        # if "background" in data:
        #     self._tensors.update(
        #         background=data["background"],
        #     )

        self.record_tensors(
            batch_size=tf.to_float(self.batch_size),
            float_is_training=self.float_is_training
        )

        self.losses = dict()

        with tf.variable_scope("representation", reuse=self.initialized):
            if self.needs_background:
                self.build_background()
            self.build_representation()

        return dict(
            tensors=self._tensors,
            recorded_tensors=self.recorded_tensors,
            losses=self.losses,
        )

    def build_representation(self):
        _, _, *img_size = self.data['tiled_obs'].shape.as_list()

        layers = [self.n_hidden] * self.n_layers

        def glimpse_encoder():
            return AIREncoder(img_size, self.glimpse_size, self.n_what, Encoder(layers),
                              masked_glimpse=self.masked_glimpse, debug=self.debug)

        steps_pred_hidden = self.n_hidden / 2

        transform_estimator = partial(StochasticTransformParam, layers, self.transform_var_bias)
        steps_predictor = partial(StepsPredictor, steps_pred_hidden, self.disc_step_bias)

        input_encoder = partial(Encoder, layers)

        with tf.variable_scope('discovery'):
            discover_cell = DiscoveryCore(img_size, self.glimpse_size, self.n_what, self.rnn_class(self.n_hidden),
                                          input_encoder, glimpse_encoder, transform_estimator, steps_predictor,
                                          debug=self.debug)

            discover = Discover(self.n_steps_per_image, discover_cell,
                                step_success_prob=self.step_success_prob,
                                where_mean=[*self.scale_prior, 0, 0],
                                disc_prior_type=self.disc_prior_type,
                                rec_where_prior=self.rec_where_prior)

        with tf.variable_scope('propagation'):
            # Prop cell should have a different rnn cell but should share all other estimators
            input_encoder = lambda: discover_cell._input_encoder
            glimpse_encoder = lambda: discover_cell._glimpse_encoder
            transform_estimator = partial(StochasticTransformParam, layers, self.transform_var_bias)
            steps_predictor = partial(StepsPredictor, steps_pred_hidden, self.prop_step_bias)

            # Prop cell should have a different rnn cell but should share all other estimators
            propagate_rnn_cell = self.rnn_class(self.n_hidden)
            temporal_rnn_cell = self.time_rnn_class(self.n_hidden)
            propagation_cell = PropagationCore(img_size, self.glimpse_size, self.n_what, propagate_rnn_cell,
                                               input_encoder, glimpse_encoder, transform_estimator,
                                               steps_predictor, temporal_rnn_cell,
                                               debug=self.debug)
            ssm = SequentialSSM(propagation_cell)

            prior_rnn = self.prior_rnn_class(self.n_hidden)
            propagation_prior = make_prior(self.prop_prior_type, self.n_what, prior_rnn, self.prop_prior_step_bias)

            propagate = Propagate(ssm, propagation_prior)

        with tf.variable_scope('decoder'):
            mean_img = self.updater.compute_validation_pixelwise_mean()

            glimpse_decoder = partial(Decoder, layers, output_scale=self.output_scale)
            decoder = AIRDecoder(img_size, self.glimpse_size, glimpse_decoder,
                                 batch_dims=2,
                                 mean_img=mean_img,
                                 output_std=self.output_std,
                                 )

        with tf.variable_scope('sequence'):
            time_cell = self.time_rnn_class(self.n_hidden)

            sequence_apdr = SequentialAIR(
                self.n_steps_per_image, self.glimpse_size, discover, propagate,
                time_cell, decoder, sample_from_prior=self.sample_from_prior)

        outputs = sequence_apdr(self.data['tiled_obs'])

        self._tensors.update(outputs)
