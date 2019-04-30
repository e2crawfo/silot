import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from collections import defaultdict
import sonnet as snt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb

from dps import cfg
from dps.utils.tf import build_scheduled_value, ConvNet, RenderHook
from dps.utils import Param

from auto_yolo.models.core import xent_loss

from spair_video.core import VideoNetwork


def cosine_similarity(a, b, keepdims=False):
    """ Supports broadcasting. """
    normalize_a = tf.nn.l2_normalize(a, axis=-1)
    normalize_b = tf.nn.l2_normalize(b, axis=-1)
    return tf.reduce_sum(normalize_a * normalize_b, axis=-1, keepdims=keepdims)


class TbaBackbone(ConvNet):
    def __init__(self, check_output_shape=False, **kwargs):
        layers = [
            dict(kernel_size=5, strides=1, filters=32, pool=True),
            dict(kernel_size=3, strides=1, filters=64, pool=True),
            dict(kernel_size=1, strides=1, filters=128, pool=True),
            dict(kernel_size=3, strides=1, filters=256, pool=False),
            dict(kernel_size=1, strides=1, filters=20, pool=False),
        ]
        super().__init__(layers, check_output_shape=False)


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

    needs_background = True

    backbone = None
    cell = None
    key_network = None
    write_network = None
    output_network = None
    background_encoder = None
    background_decoder = None

    def __init__(self, env, updater, scope=None, **kwargs):
        super().__init__(env, updater, scope=scope)

        self.lmbda = build_scheduled_value(self.lmbda, "lmbda")

        self.output_size_per_object = np.prod(self.object_shape) * (self.image_depth + 1) + 1 + 4 + self.n_layers

    def build_representation(self):
        # --- init modules ---

        if self.backbone is None:
            self.backbone = self.build_backbone(scope="backbone")
            self.backbone.layers[-1]['filters'] = self.S
            if "backbone" in self.fixed_weights:
                self.backbone.fix_variables()

        if self.cell is None:
            self.cell = cfg.build_cell(scope="cell", n_hidden=self.n_hidden)
            if "cell" in self.fixed_weights:
                self.cell.fix_variables()

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

        n_frames, n_trackers, batch_size = self.n_frames, self.n_trackers, self.batch_size

        # --- encode ---

        video = tf.reshape(self.inp, (batch_size * n_frames, *self.obs_shape[1:]))

        zero_to_one_Y = tf.linspace(0., 1., self.image_height)
        zero_to_one_X = tf.linspace(0., 1., self.image_width)
        X, Y = tf.meshgrid(zero_to_one_X, zero_to_one_Y)
        X = tf.tile(X[None, :, :, None], (batch_size * n_frames, 1, 1, 1))
        Y = tf.tile(Y[None, :, :, None], (batch_size * n_frames, 1, 1, 1))
        video = tf.concat([video, Y, X], axis=-1)

        encoded_frames = self.backbone(video, (1, 1, self.S), self.is_training)
        _, H, W, _ = encoded_frames.shape
        encoded_frames = tf.reshape(encoded_frames, (batch_size, n_frames, H*W, self.S))

        tracker_values = defaultdict(list)

        tracker_values["hidden_states"].append(
            tf.stack(
                [self.cell.zero_state(batch_size, tf.float32)
                 for i in range(n_trackers)],
                axis=0)
        )

        cts = tf.minimum(1., self.float_is_training + (1 - float(self.discrete_eval)))
        mask_temperature = cts * 1.0 + (1 - cts) * 1e-5
        layer_temperature = cts * 1.0 + (1 - cts) * 1e-5

        for f in range(n_frames):
            memory = encoded_frames[:, f]  # (batch_size, H*W, S)

            if f == 0 or not self.prioritize:
                order = tf.tile(tf.range(self.n_trackers)[:, None, None], (1, batch_size, 1))
            else:
                conf = tracker_values["conf"][-1]
                order = tf.contrib.framework.argsort(conf, axis=0, direction='DESCENDING')  # (n_trackers, batch_size, 1)

            inverse_order = tf.contrib.framework.argsort(order, axis=0, direction='ASCENDING')

            new = defaultdict(list)
            hidden_states = tracker_values["hidden_states"][f]

            for i in range(n_trackers):
                # --- apply ordering if applicable ---

                indices = order[i]  # (batch_size, 1)
                indexor = tf.concat([indices, tf.range(batch_size)[:, None]], axis=1)  # (batch_size, 2)
                _hidden_states = tf.gather_nd(hidden_states, indexor)  # (batch_size, n_hidden)

                # --- access the memory using spatial attention ---

                keys = self.key_network(_hidden_states, self.S+1, self.is_training)
                keys, temperature_logit = tf.split(keys, [self.S, 1], axis=-1)
                temperature = 1 + tf.math.softplus(temperature_logit)
                key_activation = temperature * cosine_similarity(memory, keys[:, None, :])  # (batch_size, H*W)
                normed_key_activation = tf.nn.softmax(key_activation, axis=1)[:, :, None]  # (batch_size, H*W, 1)
                attention_result = tf.reduce_sum(normed_key_activation * memory, axis=1)  # (batch_size, S)

                # --- update tracker hidden state and output ---

                raw_output, new_hidden = self.cell(attention_result, _hidden_states)

                # --- update the memory for the next trackers ---

                write = self.write_network(raw_output, 2*self.S, self.is_training)
                erase, write = tf.split(write, 2, axis=-1)
                erase = tf.nn.sigmoid(erase)

                memory = (
                    (1 - normed_key_activation * erase[:, None, :]) * memory
                    + normed_key_activation * write[:, None, :]
                )

                # --- compute the output values ---

                output = self.output_network(raw_output, self.output_size_per_object, self.is_training)
                conf, layer, pose, mask, appearance = tf.split(
                    output,
                    [1, self.n_layers, 4, np.prod(self.object_shape), self.image_depth * np.prod(self.object_shape)],
                    axis=-1)

                conf = tf.nn.sigmoid(conf)

                if "conf" in self.fixed_values:
                    conf = self.fixed_values['conf'] * tf.ones_like(conf)

                conf = (
                    self.float_is_training * conf
                    + (1 - self.float_is_training) * tf.round(conf)
                )

                layer = tfp.distributions.RelaxedOneHotCategorical(layer_temperature, logits=layer).sample()

                pose = tf.nn.tanh(pose)
                ys, xs, yt, xt = tf.split(pose, 4, axis=-1)
                if "ys" in self.fixed_values:
                    ys = self.fixed_values['ys'] * tf.ones_like(ys)
                if "xs" in self.fixed_values:
                    xs = self.fixed_values['xs'] * tf.ones_like(xs)
                if "yt" in self.fixed_values:
                    yt = self.fixed_values['yt'] * tf.ones_like(yt)
                if "xt" in self.fixed_values:
                    xt = self.fixed_values['xt'] * tf.ones_like(xt)
                pose = tf.concat([ys, xs, yt, xt], axis=-1)

                mask = tfp.distributions.RelaxedBernoulli(mask_temperature, logits=mask).sample()
                if "mask" in self.fixed_values:
                    mask = self.fixed_values['mask'] * tf.ones_like(mask)

                appearance = tf.nn.sigmoid(appearance)
                if "appearance" in self.fixed_values:
                    appearance = self.fixed_values['appearance'] * tf.ones_like(appearance)

                new["hidden_states"].append(new_hidden)
                new["tracker_output"].append(raw_output)
                new["conf"].append(conf)
                new["layer"].append(layer)
                new["pose"].append(pose)
                new["mask"].append(mask)
                new["appearance"].append(appearance)

            new = {k: tf.stack(v, axis=0) for k, v in new.items()}

            # reverse the ordering
            batch_indices = tf.tile(tf.range(batch_size)[None, :, None], (n_trackers, 1, 1))
            inverse_indexor = tf.concat([inverse_order, batch_indices], axis=2)  # (n_trackers, batch_size, 2)
            new = {k: tf.gather_nd(v, inverse_indexor) for k, v in new.items()}

            for k, v in new.items():
                tracker_values[k].append(v)
            tracker_values["order"].append(order)

        # get into shape (batch_size, n_frames, n_trackers, other)
        def transpose(tensor):
            return tf.transpose(tensor, (2, 0, 1, *range(3, len(tensor.shape))))

        tracker_values = {k: transpose(tf.stack(v, axis=0)) for k, v in tracker_values.items()}

        pose = tracker_values["pose"]
        self.record_tensors(
            conf=tracker_values["conf"],
            pose_ys=pose[..., 0],
            pose_xs=pose[..., 1],
            pose_yt=pose[..., 2],
            pose_xt=pose[..., 3]
        )

        # --- render/decode ---

        ys, xs, yt, xt = tf.split(tracker_values["pose"], 4, axis=-1)

        # scales normalized to (0, 1) by dividing by image size
        ys_normed = (1 + self.eta[0] * ys) * self.anchor_box[0] / self.image_height
        xs_normed = (1 + self.eta[1] * xs) * self.anchor_box[1] / self.image_width

        # expose values for plotting

        self._tensors.update(
            ys_normed=ys_normed,
            xs_normed=xs_normed,
            yt=yt,
            xt=xt,
            mask=tf.reshape(tracker_values["mask"], (batch_size, n_frames, n_trackers, *self.object_shape)),
            appearance=tf.reshape(
                tracker_values["appearance"], (batch_size, n_frames, n_trackers, *self.object_shape, self.image_depth)),
            conf=tracker_values["conf"],
            layer=tracker_values["layer"],
            order=tracker_values["order"],
        )

        # --- reshape values ---

        N = batch_size * n_frames * n_trackers
        ys_normed = tf.reshape(ys_normed, (N, 1))
        xs_normed = tf.reshape(xs_normed, (N, 1))
        yt = tf.reshape(yt, (N, 1))
        xt = tf.reshape(xt, (N, 1))

        mask = tf.reshape(tracker_values["mask"], (N, *self.object_shape, 1))
        appearance = tf.reshape(tracker_values["appearance"], (N, *self.object_shape, self.image_depth))

        transform_constraints = snt.AffineWarpConstraints.no_shear_2d()
        warper = snt.AffineGridWarper(
            (self.image_height, self.image_width), self.object_shape, transform_constraints)
        inverse_warper = warper.inverse()
        transforms = tf.concat([xs_normed, xt, ys_normed, yt], axis=-1)
        grid_coords = inverse_warper(transforms)

        transformed_masks = tf.contrib.resampler.resampler(mask, grid_coords)
        transformed_masks = tf.reshape(
            transformed_masks,
            (batch_size, n_frames, n_trackers, self.image_height, self.image_width, 1))

        transformed_appearances = tf.contrib.resampler.resampler(appearance, grid_coords)
        transformed_appearances = tf.reshape(
            transformed_appearances,
            (batch_size, n_frames, n_trackers, self.image_height, self.image_width, self.image_depth))

        layer_masks = []
        layer_appearances = []
        partial_frames = []

        conf = tracker_values["conf"][:, :, :, :, None, None]

        # TODO: currently assuming a black background
        final_frames = tf.zeros((batch_size, n_frames, self.image_height, self.image_width, self.image_depth))

        # For each layer, create a mask image and an appearance image
        for layer_idx in range(self.n_layers):
            layer_weight = tracker_values["layer"][:, :, :, layer_idx, None, None, None]

            # (batch_size, n_frames, self.image_height, self.image_width, 1)
            layer_mask = tf.minimum(1.0, tf.reduce_sum(conf * layer_weight * transformed_masks, axis=2))

            # (batch_size, n_frames, self.image_height, self.image_width, 3)
            layer_appearance = tf.reduce_sum(conf * layer_weight * transformed_masks * transformed_appearances, axis=2)

            final_frames = (1 - layer_mask) * final_frames + layer_appearance

            layer_masks.append(layer_mask)
            layer_appearances.append(layer_appearance)
            partial_frames.append(final_frames)

        self._tensors["output"] = final_frames

        # --- losses ---

        self._tensors['per_pixel_reconstruction_loss'] = (self.inp - final_frames)**2
        self.losses['reconstruction'] = tf.reduce_mean(self._tensors["per_pixel_reconstruction_loss"])
        self.losses['area'] = self.lmbda * tf.reduce_mean((1 + self.eta[0] * ys) * (1 + self.eta[1] * xs))


class TBA_RenderHook(RenderHook):
    N = 4

    def __call__(self, updater):
        self.fetches = "inp output ys_normed xs_normed yt xt appearance mask conf layer order"

        fetched = self._fetch(updater)
        self._plot_reconstruction(updater, fetched)

    @staticmethod
    def normalize_images(images):
        mx = images.reshape(*images.shape[:-3], -1).max(axis=-1)
        return images / mx[..., None, None, None]

    def _plot_reconstruction(self, updater, fetched):
        inp = fetched['inp']
        output = fetched['output']
        ys = fetched['ys_normed'] * updater.image_height
        xs = fetched['xs_normed'] * updater.image_width
        yt = (fetched['yt'] + 1) / 2 * updater.image_height
        xt = (fetched['xt'] + 1) / 2 * updater.image_width
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
                    top = yt[b, t, i] - ys[b, t, i] / 2
                    height = ys[b, t, i]
                    left = xt[b, t, i] - xs[b, t, i] / 2
                    width = xs[b, t, i]
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
            self.savefig("sampled_patches/" + str(b), fig, updater)
