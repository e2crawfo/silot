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

from spair_video.core import VideoNetwork, MOTMetrics


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
    keys_accessed = (
        ["resampled_" + name for name in "where_coords presence_prob num_steps_per_sample".split()]
        + "annotations n_annotations".split()
    )

    def _process_data(self, tensors, updater):
        obj = tensors["resampled_presence_prob"]
        predicted_n_digits = tensors['resampled_num_steps_per_sample']
        w, h, x, y = np.split(tensors['resampled_where_coords'], 4, axis=3)

        transformed_x = 0.5 * (x + 1.) * updater.network.image_width
        transformed_y = 0.5 * (y + 1.) * updater.network.image_height

        height = h * updater.network.image_height
        width = w * updater.network.image_width

        top = transformed_y - height / 2
        left = transformed_x - width / 2

        annotations = tensors["annotations"]
        n_annotations = tensors["n_annotations"]

        return obj, predicted_n_digits, top, left, height, width, annotations, n_annotations


class TBA_MOTMetrics(MOTMetrics):
    keys_accessed = (
        ["resampled_" + name for name in "obj_id where_coords num_steps_per_sample".split()]
        + "annotations n_annotations dynamic_n_frames".split()
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

        transformed_x = 0.5 * (x + 1.) * updater.network.image_width
        transformed_y = 0.5 * (y + 1.) * updater.network.image_height

        height = h * updater.network.image_height
        width = w * updater.network.image_width

        top = transformed_y - height / 2
        left = transformed_x - width / 2

        return obj, pred_n_objects, obj_id, top, left, height, width


def coords_to_pixel_space(y, x, h, w, image_shape, anchor_box, top_left):
    h = h * anchor_box[0]
    w = w * anchor_box[1]

    y = y * image_shape[0] - 0.5
    x = x * image_shape[1] - 0.5

    if top_left:
        y -= h / 2
        x -= w / 2

    return y, x, h, w


class TrackingByAnimation(VideoNetwork):
    build_backbone = Param()
    build_cell = Param()
    build_key_network = Param()
    build_write_network = Param()
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

    backbone = None
    cell = None
    key_network = None
    write_network = None
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

    def _loop_cond(self, f, *_):
        return f < self.dynamic_n_frames

    def _loop_body(self, f, conf, hidden_states, *tensor_arrays):
        batch_size = self.batch_size

        memory = self.encoded_frames[:, f]  # (batch_size, H*W, S)

        order = tf.cond(
            tf.logical_or(tf.equal(f, 0), not self.prioritize),
            lambda: tf.tile(tf.range(self.n_trackers)[:, None, None], (1, batch_size, 1)),
            lambda: tf.contrib.framework.argsort(conf, axis=0, direction='DESCENDING')
        )  # (n_trackers, batch_size, 1)
        order = tf.reshape(order, (self.n_trackers, batch_size, 1))

        inverse_order = tf.contrib.framework.argsort(order, axis=0, direction='ASCENDING')

        tensors = defaultdict(list)

        for i in range(self.n_trackers):
            # --- apply ordering if applicable ---

            indices = order[i]  # (batch_size, 1)
            indexor = tf.concat([indices, tf.range(batch_size)[:, None]], axis=1)  # (batch_size, 2)
            _hidden_states = tf.gather_nd(hidden_states, indexor)  # (batch_size, n_hidden)

            # --- access the memory using spatial attention ---

            keys = self.key_network(_hidden_states, self.S+1, self.is_training)  # (batch_size, self.S+1)
            keys, temperature_logit = tf.split(keys, [self.S, 1], axis=-1)
            temperature = 1 + tf.math.softplus(temperature_logit)
            key_activation = temperature * tf_cosine_similarity(memory, keys[:, None, :])  # (batch_size, H*W)
            normed_key_activation = tf.nn.softmax(key_activation, axis=1)[:, :, None]  # (batch_size, H*W, 1)
            attention_result = tf.reduce_sum(normed_key_activation * memory, axis=1)  # (batch_size, S)

            # --- update tracker hidden state and output ---

            tracker_output, new_hidden = self.cell(attention_result, _hidden_states)

            # --- update the memory for the next trackers ---

            write = self.write_network(tracker_output, 2*self.S, self.is_training)
            erase, write = tf.split(write, 2, axis=-1)
            erase = tf.nn.sigmoid(erase)

            memory = (
                (1 - normed_key_activation * erase[:, None, :]) * memory
                + normed_key_activation * write[:, None, :]
            )

            tensors["hidden_states"].append(new_hidden)
            tensors["tracker_output"].append(tracker_output)
            tensors["attention_result"].append(attention_result)

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
        # conf = tf.nn.sigmoid(conf)

        conf = (
            self.float_is_training * conf
            + (1 - self.float_is_training) * tf.round(conf)
        )

        layer = tfp.distributions.RelaxedOneHotCategorical(self.layer_temperature, logits=layer).sample()

        pose = tf.nn.tanh(pose)

        mask = tfp.distributions.RelaxedBernoulli(self.mask_temperature, logits=mask).sample()
        appearance = tf.nn.sigmoid(appearance)

        output = dict(
            conf=conf, layer=layer, pose=pose,
            mask=mask, appearance=appearance, order=order,
        )
        output.update(tensors)

        tensor_arrays = append_to_tensor_arrays(f, output, tensor_arrays)

        f += 1

        return [f, conf, tensors["hidden_states"], *tensor_arrays]

    def build_representation(self):
        # --- init modules ---

        if self.backbone is None:
            self.backbone = self.build_backbone(scope="backbone")

            if "backbone" in self.fixed_weights:
                self.backbone.fix_variables()

        if self.cell is None:
            self.cell = cfg.build_cell(self.n_hidden, name="cell")

            # self.cell must be a Sonnet RNNCore

            if self.learn_initial_state:
                self.initial_hidden_state = snt.trainable_initial_state(
                    1, self.cell.state_size, tf.float32, name="initial_hidden_state")

        if self.key_network is None:
            self.key_network = cfg.build_key_network(scope="key_network")
            if "key_network" in self.fixed_weights:
                self.key_network.fix_variables()

        if self.write_network is None:
            self.write_network = cfg.build_write_network(scope="write_network")
            if "write_network" in self.fixed_weights:
                self.write_network.fix_variables()

        if self.output_network is None:
            self.output_network = cfg.build_output_network(scope="output_network")
            if "output_network" in self.fixed_weights:
                self.output_network.fix_variables()

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

        _, H, W, _ = encoded_frames.shape
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
        xt_normed = (yt + 1) / 2
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
        yt = tf.reshape(yt, (N, 1))
        xt = tf.reshape(xt, (N, 1))

        mask = tf.reshape(tensors["mask"], (N, *self.object_shape, 1))
        appearance = tf.reshape(tensors["appearance"], (N, *self.object_shape, self.image_depth))

        transform_constraints = snt.AffineWarpConstraints.no_shear_2d()
        warper = snt.AffineGridWarper(
            (self.image_height, self.image_width), self.object_shape, transform_constraints)
        inverse_warper = warper.inverse()
        transforms = tf.concat([xs_normed, xt, ys_normed, yt], axis=-1)
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
        partial_frames = []

        conf = tensors["conf"][:, :, :, :, None, None]

        # TODO: currently assuming a black background

        final_frames = tf.zeros((batch_size, d_n_frames, self.image_height, self.image_width, self.image_depth))

        # TODO: on mnist, the shape/mask is replaced with all 1's...

        # For each layer, create a mask image and an appearance image
        for layer_idx in range(self.n_layers):
            layer_weight = tensors["layer"][:, :, :, layer_idx, None, None, None]

            # (batch_size, n_frames, self.image_height, self.image_width, 1)
            layer_mask = tf.minimum(1.0, tf.reduce_sum(conf * layer_weight * transformed_masks, axis=2))

            # vvv in pytorch code, layer_appearance is clamped to < 1 for MNIST only...

            # (batch_size, n_frames, self.image_height, self.image_width, 3)
            layer_appearance = tf.reduce_sum(conf * layer_weight * transformed_masks * transformed_appearances, axis=2)

            final_frames = (1 - layer_mask) * final_frames + layer_appearance

            layer_masks.append(layer_mask)
            layer_appearances.append(layer_appearance)
            partial_frames.append(final_frames)

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

    def build_fetches(self, updater):
        return "inp output normalized_box appearance mask conf layer order tracker_output attention_result".split()

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

        yt, xt, ys, xs = np.split(fetched['normalized_box'], 4, axis=-1)
        pixel_space_box = coords_to_pixel_space(
            yt, xt, ys, xs, (updater.image_height, updater.image_width),
            updater.network.anchor_box, top_left=True)
        pixel_space_box = np.concatenate(pixel_space_box, axis=-1)

        conf = fetched['conf']
        layer = fetched['layer']
        order = fetched['order']

        mask = fetched['mask']
        appearance = fetched['appearance']

        diff = self.normalize_images(np.abs(inp - output).sum(axis=-1, keepdims=True) / output.shape[-1])
        xent = self.normalize_images(xent_loss(pred=output, label=inp, tf=False).sum(axis=-1, keepdims=True))

        B, T = inp.shape[:2]
        print("Plotting for {} data points...".format(B))
        n_base_images = 7
        n_images = n_base_images + 2 * updater.network.n_trackers

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

                ax = axes[t, 6]
                self.imshow(ax, output[b, t])
                if t == 0:
                    ax.set_title('rec with boxes')

                for i in range(updater.network.n_trackers):
                    top, left, height, width = pixel_space_box[b, t, i]
                    rect = patches.Rectangle(
                        (left, top), width, height, linewidth=6, edgecolor=colours[i], facecolor='none')
                    ax.add_patch(rect)

                    ax_appearance = axes[t, n_base_images+2*i]
                    self.imshow(ax_appearance, appearance[b, t, i])
                    rect = patches.Rectangle(
                        (-0.05, -0.05), 1.1, 1.1, clip_on=False, linewidth=20,
                        transform=ax_appearance.transAxes, edgecolor=colours[i], facecolor='none')
                    ax_appearance.add_patch(rect)
                    if t == 0:
                        ax_appearance.set_title('appearance {}'.format(i))

                    ax_mask = axes[t, n_base_images+2*i+1]
                    self.imshow(ax_mask, mask[b, t, i])
                    rect = patches.Rectangle(
                        (-0.05, -0.05), 1.1, 1.1, clip_on=False, linewidth=20,
                        transform=ax_mask.transAxes, edgecolor=colours[i], facecolor='none')
                    ax_mask.add_patch(rect)
                    if t == 0:
                        ax_mask.set_title('mask {}'.format(i))

            plt.subplots_adjust(left=0.02, right=.98, top=.98, bottom=0.02, wspace=0.1, hspace=0.1)
            self.savefig("tba/" + str(b), fig, updater)
