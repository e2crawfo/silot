import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from collections import defaultdict
import sonnet as snt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb
import pprint

from dps import cfg
from dps.utils.tf import (
    build_scheduled_value, GridConvNet, ConvNet, RenderHook, tf_cosine_similarity,
    apply_object_wise, tf_shape
)
from dps.utils import Param, map_structure
from dps.utils.tensor_arrays import apply_keys, append_to_tensor_arrays, make_tensor_arrays

from auto_yolo.models.core import xent_loss, AP

from silot.core import VideoNetwork, MOTMetrics


class TBA_Backbone_old(ConvNet):
    def __init__(self, check_output_shape=False, **kwargs):
        # Not sure how i decided on the pooling here. They use adaptive max pooling in order to get a desired output size.
        # we can look at the desired output size for each layer, stored in model_config.json, and set the pooling based on
        # that. They want 8 x 8 for a 128 x 128 image. So that's 16 x 16 pixels per cell. The first 4 layers reduce the
        # size by 2. So lets pool for those.
        layers = [
            dict(kernel_size=5, strides=1, filters=32, pool=True),
            dict(kernel_size=3, strides=1, filters=64, pool=True),
            dict(kernel_size=1, strides=1, filters=128, pool=True),
            dict(kernel_size=3, strides=1, filters=256, pool=False),
            dict(kernel_size=1, strides=1, filters=20, pool=False),
        ]
        super().__init__(layers, check_output_shape=False)


class TBA_Backbone(GridConvNet):
    def __init__(self, check_output_shape=False, **kwargs):
        # TensorFlow's lack of an adaptive pooling operator is annoying.
        # So let's use GridConvNet, and strides instead of pooling.
        layers = [
            dict(kernel_size=5, strides=2, filters=32, pool=False),
            dict(kernel_size=3, strides=2, filters=64, pool=False),
            dict(kernel_size=1, strides=2, filters=128, pool=False),
            dict(kernel_size=3, strides=2, filters=256, pool=False),
            dict(kernel_size=1, strides=1, filters=20, pool=False),
        ]
        super().__init__(layers, check_output_shape=False)


class TBA_AP(AP):
    keys_accessed = "normalized_box conf annotations n_annotations".split()

    def _process_data(self, fetched, updater):
        conf = fetched['conf']

        nb = np.split(fetched['normalized_box'], 4, axis=-1)
        top, left, height, width = tba_coords_to_pixel_space(
            *nb, (updater.image_height, updater.image_width),
            updater.network.anchor_box, top_left=True)

        batch_size, n_frames, n_objects, *_ = conf.shape
        shape = (batch_size, n_frames, n_objects)

        predicted_n_digits = n_objects * np.ones((batch_size, n_frames), dtype=np.int32)

        obj = conf.reshape(*shape)
        top = top.reshape(*shape)
        left = left.reshape(*shape)
        height = height.reshape(*shape)
        width = width.reshape(*shape)

        annotations = fetched["annotations"]
        n_annotations = fetched["n_annotations"]

        return obj, predicted_n_digits, top, left, height, width, annotations, n_annotations


class TBA_MOTMetrics(MOTMetrics):
    keys_accessed = "normalized_box conf annotations n_annotations".split()

    def _process_data(self, fetched, updater):
        """
        Use a confidence threshold of 0.5. Note that SILOT and SQAIR both do this as well.
        For MOT there is no precision/recall balance, so you only want to include objects
        you are sure of (in the case of AP, the confidence ordering takes care of it).
        """
        conf = fetched['conf']

        nb = np.split(fetched['normalized_box'], 4, axis=-1)
        top, left, height, width = tba_coords_to_pixel_space(
            *nb, (updater.image_height, updater.image_width),
            updater.network.anchor_box, top_left=True)

        batch_size, n_frames, n_objects, *_ = conf.shape
        shape = (batch_size, n_frames, n_objects)

        obj = conf.reshape(*shape)
        top = top.reshape(*shape)
        left = left.reshape(*shape)
        height = height.reshape(*shape)
        width = width.reshape(*shape)

        B, F, n_objects = shape
        pred_ids = np.zeros((B, F), dtype=np.object)

        for b in range(B):
            next_id = 0
            ids = [-1] * n_objects

            for f in range(F):
                _pred_ids = []
                for i in range(n_objects):
                    if obj[b, f, i] > 0.5:
                        is_new = f == 0 or not (obj[b, f-1, i] > 0.5)
                        if is_new:
                            ids[i] = next_id
                            next_id += 1
                        _pred_ids.append(ids[i])
                pred_ids[b, f] = _pred_ids

        pred_n_digits = n_objects * np.ones((batch_size, n_frames), dtype=np.int32)
        return obj, pred_n_digits, pred_ids, top, left, height, width


def tba_coords_to_pixel_space(y, x, h, w, image_shape, anchor_box, top_left):
    h = h * anchor_box[0]
    w = w * anchor_box[1]

    y = y * image_shape[0] - 0.5
    x = x * image_shape[1] - 0.5

    if top_left:
        y -= h / 2
        x -= w / 2

    return y, x, h, w


def tba_coords_to_image_space(y, x, h, w, image_shape, anchor_box, top_left):
    h = h * anchor_box[0] / image_shape[0]
    w = w * anchor_box[1] / image_shape[1]

    y = 2 * y - 1
    x = 2 * x - 1

    if top_left:
        y -= h / 2
        x -= w / 2

    return y, x, h, w


@tf.custom_gradient
def limit_grad_norm(x, max_norm):

    def grad(dy):
        axes = list(range(1, len(dy.shape)))
        _dy = tf.clip_by_norm(dy, max_norm, axes=axes)
        return _dy, tf.zeros_like(max_norm)

    return tf.identity(x), grad


class TrackingByAnimation(VideoNetwork):
    build_backbone = Param()
    build_cell = Param()
    build_key_network = Param()
    build_beta_network = Param()
    build_write_network = Param()
    build_erase_network = Param()
    build_output_network = Param()

    lmbda = Param()
    n_trackers = Param()
    n_layers = Param()
    n_hidden = Param()
    S = Param(help="Channel dimension of the encoded frames and memory.")
    eta = Param(help="Scaling co-efficients (y, x).")
    object_shape = Param()
    prioritize = Param()
    anchor_box = Param(help="Box with respect to which objects are scaled (y, x).")
    discrete_eval = Param()
    learn_initial_state = Param()
    fixed_mask = Param()
    clamp_appearance = Param()

    backbone = None
    cell = None
    key_network = None
    beta_network = None
    write_network = None
    erase_network = None
    output_network = None
    background_encoder = None
    background_decoder = None

    attr_prior_mean = None
    attr_prior_std = None
    noisy = None

    needs_background = False

    def __init__(self, env, updater, scope=None, **kwargs):
        super().__init__(env, updater, scope=scope)

        self.lmbda = build_scheduled_value(self.lmbda, "lmbda")

        self.output_size_per_object = np.prod(self.object_shape) * (self.image_depth + 1) + 1 + 4 + self.n_layers

    @property
    def eval_funcs(self):
        if getattr(self, '_eval_funcs', None) is None:
            if "annotations" in self._tensors:
                ap_iou_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                eval_funcs = {"AP_at_point_{}".format(int(10 * v)): TBA_AP(v) for v in ap_iou_values}
                eval_funcs["AP"] = TBA_AP(ap_iou_values)
                eval_funcs["AP_train"] = TBA_AP(ap_iou_values, is_training=True)
                eval_funcs["MOT"] = TBA_MOTMetrics()
                eval_funcs["MOT_train"] = TBA_MOTMetrics(is_training=True)

                self._eval_funcs = eval_funcs
            else:
                self._eval_funcs = {}

        return self._eval_funcs

    def _loop_cond(self, f, *_):
        return f < self.dynamic_n_frames

    def _loop_body(self, f, conf, hidden_states, *tensor_arrays):
        batch_size = self.batch_size

        memory = self.encoded_frames[:, f]  # (batch_size, H*W, S)

        delta = 0.0001 * tf.range(self.n_trackers, dtype=tf.float32)[:, None, None]
        sort_criteria = tf.round(conf) - delta

        sorted_order = tf.contrib.framework.argsort(sort_criteria, axis=0, direction='DESCENDING')
        sorted_order = tf.reshape(sorted_order, (self.n_trackers, batch_size, 1))

        order = tf.cond(
            tf.logical_or(tf.equal(f, 0), not self.prioritize),
            lambda: tf.tile(tf.range(self.n_trackers)[:, None, None], (1, batch_size, 1)),
            lambda: sorted_order,
        )
        order = tf.reshape(order, (self.n_trackers, batch_size, 1))

        inverse_order = tf.contrib.framework.argsort(order, axis=0, direction='ASCENDING')

        tensors = defaultdict(list)

        for i in range(self.n_trackers):
            tensors["memory_activation"].append(tf.reduce_mean(tf.abs(memory), axis=2))

            # --- apply ordering if applicable ---

            indices = order[i]  # (batch_size, 1)
            indexor = tf.concat([indices, tf.range(batch_size)[:, None]], axis=1)  # (batch_size, 2)
            _hidden_states = tf.gather_nd(hidden_states, indexor)  # (batch_size, n_hidden)

            # --- access the memory using spatial attention ---

            keys = self.key_network(_hidden_states, self.S, self.is_training)  # (batch_size, self.S)
            beta_logit = self.beta_network(_hidden_states, 1, self.is_training)  # (batch_size, 1)

            # beta = 1 + tf.math.softplus(beta_logit)

            beta_pos = tf.maximum(0.0, beta_logit)
            beta_neg = tf.minimum(0.0, beta_logit)
            beta = tf.log1p(tf.exp(beta_neg)) + beta_pos + tf.log1p(tf.exp(-beta_pos)) + (1 - np.log(2))

            _memory = tf.identity(memory)
            _memory = limit_grad_norm(_memory, 1.)

            key_activation = beta * tf_cosine_similarity(_memory, keys[:, None, :])  # (batch_size, H*W)
            attention_weights = tf.nn.softmax(key_activation, axis=1)[:, :, None]  # (batch_size, H*W, 1)

            _attention_weights = tf.identity(attention_weights)
            _attention_weights = limit_grad_norm(_attention_weights, 1.)

            attention_result = tf.reduce_sum(_attention_weights * memory, axis=1)  # (batch_size, S)

            # --- update tracker hidden state and output ---

            tracker_output, new_hidden = self.cell(attention_result, _hidden_states)

            # --- update the memory for the next trackers ---

            write = self.write_network(tracker_output, self.S, self.is_training)
            erase = self.erase_network(tracker_output, self.S, self.is_training)
            erase = tf.nn.sigmoid(erase)

            memory = (
                (1 - attention_weights * erase[:, None, :]) * memory
                + attention_weights * write[:, None, :]
            )

            tensors["hidden_states"].append(new_hidden)
            tensors["tracker_output"].append(tracker_output)
            tensors["attention_result"].append(attention_result)
            tensors["attention_weights"].append(attention_weights[..., 0])

        tensors = {k: tf.stack(v, axis=0) for k, v in tensors.items()}

        # --- invert the ordering ---

        batch_indices = tf.tile(tf.range(batch_size)[None, :, None], (self.n_trackers, 1, 1))
        inverse_indexor = tf.concat([inverse_order, batch_indices], axis=2)  # (n_trackers, batch_size, 2)
        tensors = {k: tf.gather_nd(v, inverse_indexor) for k, v in tensors.items()}

        # --- compute the output values ---

        output = apply_object_wise(
            self.output_network, tensors["tracker_output"],
            output_size=self.output_size_per_object, is_training=self.is_training)

        conf, layer, pose, mask, appearance = tf.split(
            output,
            [1, self.n_layers, 4, np.prod(self.object_shape), self.image_depth * np.prod(self.object_shape)],
            axis=-1)

        conf = tf.abs(tf.nn.tanh(conf))

        conf = (
            self.float_is_training * conf
            + (1 - self.float_is_training) * tf.round(conf)
        )

        layer = tf.nn.softmax(layer, axis=-1)
        layer = tf.transpose(layer, (1, 0, 2))
        layer = limit_grad_norm(layer, 10.)
        layer = tf.transpose(layer, (1, 0, 2))
        layer = tfp.distributions.RelaxedOneHotCategorical(self.layer_temperature, probs=layer).sample()

        pose = tf.nn.tanh(pose)

        mask = tfp.distributions.RelaxedBernoulli(self.mask_temperature, logits=mask).sample()

        if self.fixed_mask:
            mask = tf.ones_like(mask)

        appearance = tf.nn.sigmoid(appearance)

        output = dict(
            conf=conf, layer=layer, pose=pose, mask=mask, appearance=appearance, order=order, **tensors
        )

        tensor_arrays = append_to_tensor_arrays(f, output, tensor_arrays)

        f += 1

        return [f, conf, tensors["hidden_states"], *tensor_arrays]

    def build_representation(self):
        # --- init modules ---

        if self.backbone is None:
            self.backbone = self.build_backbone(scope="backbone")

        if self.cell is None:
            self.cell = cfg.build_cell(self.n_hidden, name="cell")

            # self.cell must be a Sonnet RNNCore

            if self.learn_initial_state:
                self.initial_hidden_state = snt.trainable_initial_state(
                    1, self.cell.state_size, tf.float32, name="initial_hidden_state")

        if self.key_network is None:
            self.key_network = cfg.build_key_network(scope="key_network")
        if self.beta_network is None:
            self.beta_network = cfg.build_beta_network(scope="beta_network")
        if self.write_network is None:
            self.write_network = cfg.build_write_network(scope="write_network")
        if self.erase_network is None:
            self.erase_network = cfg.build_erase_network(scope="erase_network")

        if self.output_network is None:
            self.output_network = cfg.build_output_network(scope="output_network")

        d_n_frames, n_trackers, batch_size = self.dynamic_n_frames, self.n_trackers, self.batch_size

        # --- encode ---

        video = tf.reshape(self.inp, (batch_size * d_n_frames, *self.obs_shape[1:]))

        zero_to_one_Y = tf.linspace(0., 1., self.image_height)
        zero_to_one_X = tf.linspace(0., 1., self.image_width)
        X, Y = tf.meshgrid(zero_to_one_X, zero_to_one_Y)
        X = tf.tile(X[None, :, :, None], (batch_size * d_n_frames, 1, 1, 1))
        Y = tf.tile(Y[None, :, :, None], (batch_size * d_n_frames, 1, 1, 1))
        video = tf.concat([video, Y, X], axis=-1)

        encoded_frames, _, _ = self.backbone(video, self.S, self.is_training)

        _, H, W, _ = tf_shape(encoded_frames)
        self.H = H
        self.W = W
        encoded_frames = tf.reshape(encoded_frames, (batch_size, d_n_frames, H*W, self.S))
        self.encoded_frames = encoded_frames

        cts = tf.minimum(1., self.float_is_training + (1 - float(self.discrete_eval)))
        self.mask_temperature = cts * 1.0 + (1 - cts) * 1e-5
        self.layer_temperature = cts * 1.0 + (1 - cts) * 1e-5

        f = tf.constant(0, dtype=tf.int32)

        if self.learn_initial_state:
            hidden_state = self.initial_hidden_state[None, ...]
        else:
            hidden_state = self.cell.zero_state(1, tf.float32)[None, ...]

        hidden_states = tf.tile(
            hidden_state, (self.n_trackers, self.batch_size,) + (1,) * (len(hidden_state.shape)-2))

        conf = tf.zeros((n_trackers, batch_size, 1))

        structure = dict(
            hidden_states=hidden_states,
            tracker_output=tf.zeros((self.n_trackers, self.batch_size, self.n_hidden)),
            attention_result=tf.zeros((self.n_trackers, self.batch_size, self.S)),
            attention_weights=tf.zeros((self.n_trackers, self.batch_size, H*W)),
            memory_activation=tf.zeros((self.n_trackers, self.batch_size, H*W)),
            conf=conf,
            layer=tf.zeros((self.n_trackers, self.batch_size, self.n_layers)),
            pose=tf.zeros((self.n_trackers, self.batch_size, 4)),
            mask=tf.zeros((self.n_trackers, self.batch_size, np.prod(self.object_shape))),
            appearance=tf.zeros((self.n_trackers, self.batch_size, 3*np.prod(self.object_shape))),
            order=tf.zeros((self.n_trackers, self.batch_size, 1), dtype=tf.int32),
        )
        tensor_arrays = make_tensor_arrays(structure, self.dynamic_n_frames)

        loop_vars = [f, conf, hidden_states, *tensor_arrays]

        result = tf.while_loop(self._loop_cond, self._loop_body, loop_vars)

        first_ta_idx = min(i for i, ta in enumerate(result) if isinstance(ta, tf.TensorArray))
        tensor_arrays = result[first_ta_idx:]

        def finalize_ta(ta):
            t = ta.stack()
            # reshape from (n_frames, n_trackers, batch_size, *other) to (batch_size, n_frames, n_trackers, *other)
            return tf.transpose(t, (2, 0, 1, *range(3, len(t.shape))))

        tensors = map_structure(finalize_ta, tensor_arrays, is_leaf=lambda t: isinstance(t, tf.TensorArray))
        tensors = apply_keys(structure, tensors)

        self._tensors.update(tensors)

        pprint.pprint(self._tensors)

        # --- render/decode ---

        ys, xs, yt, xt = tf.split(tensors["pose"], 4, axis=-1)
        self.record_tensors(conf=tensors["conf"], ys=ys, xs=xs, yt=yt, xt=xt,)

        yt_normed = (yt + 1) / 2
        xt_normed = (xt + 1) / 2
        ys_normed = 1 + self.eta[0] * ys
        xs_normed = 1 + self.eta[1] * xs

        normalized_box = tf.concat([yt_normed, xt_normed, ys_normed, xs_normed], axis=-1)

        # expose values for plotting

        self._tensors.update(
            normalized_box=normalized_box,
            mask=tf.reshape(tensors["mask"], (batch_size, d_n_frames, n_trackers, *self.object_shape)),
            appearance=tf.reshape(
                tensors["appearance"], (batch_size, d_n_frames, n_trackers, *self.object_shape, self.image_depth)),
            conf=tensors["conf"],
            layer=tensors["layer"],
            order=tensors["order"],
        )

        # --- reshape values ---

        N = batch_size * d_n_frames * n_trackers
        ys_normed = tf.reshape(ys_normed, (N, 1))
        xs_normed = tf.reshape(xs_normed, (N, 1))
        yt_normed = tf.reshape(yt_normed, (N, 1))
        xt_normed = tf.reshape(xt_normed, (N, 1))

        _yt, _xt, _ys, _xs = tba_coords_to_image_space(
            yt_normed, xt_normed, ys_normed, xs_normed,
            (self.image_height, self.image_width), self.anchor_box, top_left=False)

        mask = tf.reshape(tensors["mask"], (N, *self.object_shape, 1))
        appearance = tf.reshape(tensors["appearance"], (N, *self.object_shape, self.image_depth))

        transform_constraints = snt.AffineWarpConstraints.no_shear_2d()
        warper = snt.AffineGridWarper(
            (self.image_height, self.image_width), self.object_shape, transform_constraints)
        inverse_warper = warper.inverse()
        transforms = tf.concat([_xs, _xt, _ys, _yt], axis=-1)
        grid_coords = inverse_warper(transforms)

        transformed_masks = tf.contrib.resampler.resampler(mask, grid_coords)
        transformed_masks = tf.reshape(
            transformed_masks,
            (batch_size, d_n_frames, n_trackers, self.image_height, self.image_width, 1))

        transformed_appearances = tf.contrib.resampler.resampler(appearance, grid_coords)
        transformed_appearances = tf.reshape(
            transformed_appearances,
            (batch_size, d_n_frames, n_trackers, self.image_height, self.image_width, self.image_depth))

        layer_masks = []
        layer_appearances = []

        conf = tensors["conf"][:, :, :, :, None, None]

        # TODO: currently assuming a black background

        final_frames = tf.zeros((batch_size, d_n_frames, self.image_height, self.image_width, self.image_depth))

        # For each layer, create a mask image and an appearance image
        for layer_idx in range(self.n_layers):
            layer_weight = tensors["layer"][:, :, :, layer_idx, None, None, None]

            # (batch_size, n_frames, self.image_height, self.image_width, 1)
            layer_mask = tf.reduce_sum(conf * layer_weight * transformed_masks, axis=2)
            layer_mask = tf.minimum(1.0, layer_mask)

            # (batch_size, n_frames, self.image_height, self.image_width, 3)
            layer_appearance = tf.reduce_sum(conf * layer_weight * transformed_masks * transformed_appearances, axis=2)

            if self.clamp_appearance:
                layer_appearance = tf.minimum(1.0, layer_appearance)

            final_frames = (1 - layer_mask) * final_frames + layer_appearance

            layer_masks.append(layer_mask)
            layer_appearances.append(layer_appearance)

        self._tensors["output"] = final_frames

        # --- losses ---

        self._tensors['per_pixel_reconstruction_loss'] = (self.inp - final_frames)**2

        self.losses['reconstruction'] = (
            tf.reduce_sum(self._tensors["per_pixel_reconstruction_loss"])
            / tf.cast(d_n_frames * self.batch_size, tf.float32)
        )
        # self.losses['reconstruction'] = tf.reduce_mean(self._tensors["per_pixel_reconstruction_loss"])

        self.losses['area'] = self.lmbda * tf.reduce_mean(ys_normed * xs_normed)


class TBA_RenderHook(RenderHook):
    N = 4
    linewidth = 4
    gt_color = "xkcd:yellow"
    gt_color2 = "xkcd:fire engine red"

    def build_fetches(self, updater):
        fetches = ("inp output normalized_box appearance mask conf layer order "
                   "tracker_output attention_result memory_activation attention_weights ")
        if "n_annotations" in updater.network._tensors:
            fetches += " annotations n_annotations"
        return fetches.split()

    def __call__(self, updater):
        fetched = self._fetch(updater)
        self._plot_reconstruction(updater, fetched)

    @staticmethod
    def normalize_images(images):
        mx = images.reshape(*images.shape[:-3], -1).max(axis=-1)
        return images / mx[..., None, None, None]

    def _plot_reconstruction(self, updater, fetched):
        inp = fetched['inp']
        output = fetched['output']

        nb = np.split(fetched['normalized_box'], 4, axis=-1)
        pixel_space_box = tba_coords_to_pixel_space(
            *nb, (updater.image_height, updater.image_width),
            updater.network.anchor_box, top_left=True)
        pixel_space_box = np.concatenate(pixel_space_box, axis=-1)

        conf = fetched['conf']
        layer = fetched['layer']
        order = fetched['order']

        H, W = updater.network.H, updater.network.W
        attention_weights = fetched['attention_weights']
        attention_weights = attention_weights.reshape(*attention_weights.shape[:-1], H, W)
        memory_activation = fetched['memory_activation']
        memory_activation = memory_activation.reshape(*memory_activation.shape[:-1], H, W)

        mask = fetched['mask']
        appearance = fetched['appearance']

        annotations = fetched.get('annotations', None)
        n_annotations = fetched.get('n_annotations', np.zeros(inp.shape[0], dtype='i'))

        diff = self.normalize_images(np.abs(inp - output).sum(axis=-1, keepdims=True) / output.shape[-1])
        xent = self.normalize_images(xent_loss(pred=output, label=inp, tf=False).sum(axis=-1, keepdims=True))

        B, T = inp.shape[:2]
        print("Plotting for {} data points...".format(B))
        n_base_images = 8
        n_images_per_obj = 4
        n_images = n_base_images + n_images_per_obj * updater.network.n_trackers

        fig_unit_size = 4
        fig_height = T * fig_unit_size
        fig_width = n_images * fig_unit_size

        colours = plt.get_cmap('Set3').colors[:updater.network.n_trackers]
        rgb_colours = np.array([to_rgb(c) for c in colours])

        for b in range(B):
            fig, axes = plt.subplots(T, n_images, figsize=(fig_width, fig_height))
            for ax in axes.flatten():
                ax.set_axis_off()

            for t in range(T):
                ax = axes[t, 0]
                self.imshow(ax, inp[b, t])
                if t == 0:
                    ax.set_title('input')

                ax = axes[t, 1]
                self.imshow(ax, output[b, t])
                if t == 0:
                    ax.set_title('reconstruction')

                ax = axes[t, 2]
                self.imshow(ax, diff[b, t])
                if t == 0:
                    ax.set_title('abs error')

                ax = axes[t, 3]
                self.imshow(ax, xent[b, t])
                if t == 0:
                    ax.set_title('xent')

                ax = axes[t, 4]
                array = np.concatenate(
                    [rgb_colours[:, None, :],
                     conf[b, t, :, :, None] * (1., 1., 1.),
                     layer[b, t, :, :, None] * (1., 1., 1.)],
                    axis=1)
                self.imshow(ax, array)
                if t == 0:
                    ax.set_title('conf & layer')

                ax = axes[t, 5]
                array = rgb_colours[order[b, t, :, 0]][None, :, :]
                self.imshow(ax, array)
                if t == 0:
                    ax.set_title('order')

                gt_ax = axes[t, 6]
                self.imshow(gt_ax, inp[b, t])
                if t == 0:
                    gt_ax.set_title('gt with boxes')

                rec_ax = axes[t, 7]
                self.imshow(rec_ax, output[b, t])
                if t == 0:
                    rec_ax.set_title('rec with boxes')

                # Plot true bounding boxes
                for k in range(n_annotations[b]):
                    valid, _, _, top, bottom, left, right = annotations[b, t, k]

                    if not valid:
                        continue

                    height = bottom - top
                    width = right - left

                    # make a striped rectangle by superimposing two rectangles with different linestyles

                    rect = patches.Rectangle(
                        (left, top), width, height, linewidth=self.linewidth,
                        edgecolor=self.gt_color, facecolor='none', linestyle="-")
                    gt_ax.add_patch(rect)
                    rect = patches.Rectangle(
                        (left, top), width, height, linewidth=self.linewidth,
                        edgecolor=self.gt_color2, facecolor='none', linestyle=":")
                    gt_ax.add_patch(rect)

                    rect = patches.Rectangle(
                        (left, top), width, height, linewidth=self.linewidth,
                        edgecolor=self.gt_color, facecolor='none', linestyle="-")
                    rec_ax.add_patch(rect)
                    rect = patches.Rectangle(
                        (left, top), width, height, linewidth=self.linewidth,
                        edgecolor=self.gt_color2, facecolor='none', linestyle=":")
                    rec_ax.add_patch(rect)

                for i in range(updater.network.n_trackers):
                    top, left, height, width = pixel_space_box[b, t, i]
                    rect = patches.Rectangle(
                        (left, top), width, height, linewidth=6, edgecolor=colours[i], facecolor='none')
                    gt_ax.add_patch(rect)
                    rect = patches.Rectangle(
                        (left, top), width, height, linewidth=6, edgecolor=colours[i], facecolor='none')
                    rec_ax.add_patch(rect)

                    ax_appearance = axes[t, n_base_images+n_images_per_obj*i]
                    self.imshow(ax_appearance, appearance[b, t, i])
                    rect = patches.Rectangle(
                        (-0.05, -0.05), 1.1, 1.1, clip_on=False, linewidth=20,
                        transform=ax_appearance.transAxes, edgecolor=colours[i], facecolor='none')
                    ax_appearance.add_patch(rect)
                    if t == 0:
                        ax_appearance.set_title('appearance {}'.format(i))

                    ax_mask = axes[t, n_base_images+n_images_per_obj*i+1]
                    self.imshow(ax_mask, mask[b, t, i])
                    rect = patches.Rectangle(
                        (-0.05, -0.05), 1.1, 1.1, clip_on=False, linewidth=20,
                        transform=ax_mask.transAxes, edgecolor=colours[i], facecolor='none')
                    ax_mask.add_patch(rect)
                    if t == 0:
                        ax_mask.set_title('mask {}'.format(i))

                    ax_mem = axes[t, n_base_images+n_images_per_obj*i+2]
                    self.imshow(ax_mem, memory_activation[b, t, i])
                    rect = patches.Rectangle(
                        (-0.05, -0.05), 1.1, 1.1, clip_on=False, linewidth=20,
                        transform=ax_mem.transAxes, edgecolor=colours[i], facecolor='none')
                    ax_mem.add_patch(rect)
                    if t == 0:
                        ax_mem.set_title('memory_activation {}'.format(i))

                    ax_att = axes[t, n_base_images+n_images_per_obj*i+3]
                    self.imshow(ax_att, attention_weights[b, t, i])
                    rect = patches.Rectangle(
                        (-0.05, -0.05), 1.1, 1.1, clip_on=False, linewidth=20,
                        transform=ax_att.transAxes, edgecolor=colours[i], facecolor='none')
                    ax_att.add_patch(rect)
                    if t == 0:
                        ax_att.set_title('attention_weights {}'.format(i))

            plt.subplots_adjust(left=0.02, right=.98, top=.98, bottom=0.02, wspace=0.1, hspace=0.1)
            self.savefig("tba/" + str(b), fig, updater)
