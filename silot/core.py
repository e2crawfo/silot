import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from tensorflow.python.ops.rnn import dynamic_rnn
import sonnet as snt
from orderedattrdict import AttrDict
import motmetrics as mm

from dps import cfg
from dps.utils import Param, animate
from dps.utils.tf import (
    build_scheduled_value, FIXED_COLLECTION, tf_mean_sum, MLP,
    RenderHook, tf_shape, ConvNet, RecurrentGridConvNet,
)

from auto_yolo.models.core import normal_vae, TensorRecorder, xent_loss, coords_to_pixel_space


def get_object_ids(obj, is_new, threshold=0.5, on_only=True):
    shape = obj.shape[:3]
    obj = obj.reshape(shape)
    is_new = is_new.reshape(shape)
    B, F, n_objects = shape
    pred_ids = np.zeros((B, F), dtype=np.object)

    for b in range(B):
        next_id = 0
        ids = [-1] * n_objects
        for f in range(F):
            _pred_ids = []
            for i in range(n_objects):
                if obj[b, f, i] > threshold:
                    if is_new[b, f, i]:
                        ids[i] = next_id
                        next_id += 1
                    _pred_ids.append(ids[i])
                elif not on_only:
                    _pred_ids.append(ids[i])

            pred_ids[b, f] = _pred_ids
    return pred_ids


class MOTMetrics:
    keys_accessed = "is_new normalized_box obj annotations n_annotations"

    def __init__(self, start_frame=0, end_frame=np.inf, is_training=False):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.is_training = is_training

    def get_feed_dict(self, updater):
        if self.is_training:
            return {updater.network.is_training: True}
        else:
            return {}

    def _process_data(self, tensors, updater):
        obj = tensors['obj']

        shape = obj.shape[:3]
        obj = obj.reshape(shape)

        nb = np.split(tensors['normalized_box'], 4, axis=-1)
        top, left, height, width = coords_to_pixel_space(
            *nb, (updater.image_height, updater.image_width),
            updater.network.anchor_box, top_left=True)

        top = top.reshape(shape)
        left = left.reshape(shape)
        height = height.reshape(shape)
        width = width.reshape(shape)

        is_new = tensors['is_new']
        B, F, n_objects = shape
        pred_ids = np.zeros((B, F), dtype=np.object)

        for b in range(B):
            next_id = 0
            ids = [-1] * n_objects
            for f in range(F):
                _pred_ids = []
                for i in range(n_objects):
                    if obj[b, f, i] > cfg.obj_threshold:
                        if is_new[b, f, i]:
                            ids[i] = next_id
                            next_id += 1
                        _pred_ids.append(ids[i])
                pred_ids[b, f] = _pred_ids

        pred_n_objects = n_objects * np.ones((B, F), dtype=np.int32)

        return obj, pred_n_objects, pred_ids, top, left, height, width

    def __call__(self, tensors, updater):
        obj, pred_n_objects, pred_ids, top, left, height, width = self._process_data(tensors, updater)

        annotations = tensors["annotations"]
        batch_size, n_frames, n_objects = obj.shape[:3]

        accumulators = []

        for b in range(batch_size):
            acc = mm.MOTAccumulator(auto_id=True)

            for f in range(self.start_frame, min(self.end_frame, n_frames)):
                gt_ids = [int(_id) for valid, _, _id, *_ in annotations[b, f] if float(valid) > 0.5]
                gt_boxes = [
                    (top, left, bottom-top, right-left)
                    for valid, _, _, top, bottom, left, right in annotations[b, f]
                    if float(valid) > 0.5]

                _pred_ids = [int(j) for j in pred_ids[b, f] if j >= 0]
                pred_boxes = []

                for i in range(int(pred_n_objects[b, f])):
                    if obj[b, f, i] > cfg.obj_threshold:
                        pred_boxes.append([top[b, f, i], left[b, f, i], height[b, f, i], width[b, f, i]])

                # Speed things up for really bad trackers
                if len(pred_boxes) > 2 * len(gt_boxes):
                    pred_boxes = []
                    _pred_ids = []

                distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
                acc.update(gt_ids, _pred_ids, distances)
            accumulators.append(acc)

        mh = mm.metrics.create()

        summary = mh.compute_many(
            accumulators,
            metrics=['mota'],
            # metrics=['mota', 'idf1'],
            names=[str(i) for i in range(len(accumulators))],
            generate_overall=True
        )
        return dict(summary.loc['OVERALL'])


class VideoNetwork(TensorRecorder):
    attr_prior_mean = Param()
    attr_prior_std = Param()
    noisy = Param()
    stage_steps = Param()
    initial_n_frames = Param()
    n_frames_scale = Param()

    background_encoder = None
    background_decoder = None

    eval_funcs = dict()

    def __init__(self, env, updater, scope=None, **kwargs):
        self.updater = updater

        self.obs_shape = env.datasets['train'].obs_shape
        self.n_frames, self.image_height, self.image_width, self.image_depth = self.obs_shape

        super(VideoNetwork, self).__init__(scope=scope, **kwargs)

    def std_nonlinearity(self, std_logit):
        std = 2 * tf.nn.sigmoid(tf.clip_by_value(std_logit, -10, 10))
        if not self.noisy:
            std = tf.zeros_like(std)
        return std

    @property
    def inp(self):
        return self._tensors["inp"]

    @property
    def batch_size(self):
        return self._tensors["batch_size"]

    @property
    def is_training(self):
        return self._tensors["is_training"]

    @property
    def float_is_training(self):
        return self._tensors["float_is_training"]

    def _call(self, data, is_training):
        self.data = data

        inp = data["image"]
        self._tensors = AttrDict(
            inp=inp,
            is_training=is_training,
            float_is_training=tf.to_float(is_training),
            batch_size=tf.shape(inp)[0],
        )

        if "annotations" in data:
            self._tensors.update(
                annotations=data["annotations"]["data"],
                n_annotations=data["annotations"]["shapes"][:, 1],
                n_valid_annotations=tf.to_int32(
                    tf.reduce_sum(
                        data["annotations"]["data"][:, :, :, 0]
                        * tf.to_float(data["annotations"]["mask"][:, :, :, 0]),
                        axis=2
                    )
                )
            )

        if "label" in data:
            self._tensors.update(
                targets=data["label"],
            )

        if "background" in data:
            self._tensors.update(
                ground_truth_background=data["background"],
            )

        if "offset" in data:
            self._tensors.update(
                offset=data["offset"],
            )

        max_n_frames = tf_shape(inp)[1]

        if self.stage_steps is None:
            self.current_stage = tf.constant(0, tf.int32)
            dynamic_n_frames = max_n_frames
        else:
            self.current_stage = tf.cast(tf.train.get_or_create_global_step(), tf.int32) // self.stage_steps
            dynamic_n_frames = tf.minimum(
                self.initial_n_frames + self.n_frames_scale * self.current_stage, max_n_frames)

        dynamic_n_frames = tf.cast(dynamic_n_frames, tf.float32)
        dynamic_n_frames = (
            self.float_is_training * tf.cast(dynamic_n_frames, tf.float32)
            + (1-self.float_is_training) * tf.cast(max_n_frames, tf.float32)
        )
        self.dynamic_n_frames = tf.cast(dynamic_n_frames, tf.int32)

        self._tensors.current_stage = self.current_stage
        self._tensors.dynamic_n_frames = self.dynamic_n_frames

        self._tensors.inp = self._tensors.inp[:, :self.dynamic_n_frames]

        if 'annotations' in self._tensors:
            self._tensors.annotations = self._tensors.annotations[:, :self.dynamic_n_frames]
            # self._tensors.n_annotations = self._tensors.n_annotations[:, :self.dynamic_n_frames]
            self._tensors.n_valid_annotations = self._tensors.n_valid_annotations[:, :self.dynamic_n_frames]

        self.record_tensors(
            batch_size=tf.to_float(self.batch_size),
            float_is_training=self.float_is_training,
            current_stage=self.current_stage,
            dynamic_n_frames=self.dynamic_n_frames,
        )

        self.losses = dict()

        with tf.variable_scope("representation", reuse=self.initialized):
            self.build_representation()

        return dict(
            tensors=self._tensors,
            recorded_tensors=self.recorded_tensors,
            losses=self.losses,
        )

    def build_background(self):
        if cfg.background_cfg.mode == "colour":
            rgb = np.array(to_rgb(cfg.background_cfg.colour))[None, None, None, :]
            background = rgb * tf.ones_like(self.inp)

        elif cfg.background_cfg.mode == "learn_solid":
            # Learn a solid colour for the background
            self.solid_background_logits = tf.get_variable("solid_background", initializer=[0.0, 0.0, 0.0])
            if "background" in self.fixed_weights:
                tf.add_to_collection(FIXED_COLLECTION, self.solid_background_logits)
            solid_background = tf.nn.sigmoid(10 * self.solid_background_logits)
            background = solid_background[None, None, None, :] * tf.ones_like(self.inp)

        elif cfg.background_cfg.mode == "scalor":
            pass

        elif cfg.background_cfg.mode == "learn":
            self.maybe_build_subnet("background_encoder")
            self.maybe_build_subnet("background_decoder")

            # Here I'm just encoding the first frame...
            bg_attr = self.background_encoder(self.inp[:, 0], 2 * cfg.background_cfg.A, self.is_training)
            bg_attr_mean, bg_attr_log_std = tf.split(bg_attr, 2, axis=-1)
            bg_attr_std = tf.exp(bg_attr_log_std)
            if not self.noisy:
                bg_attr_std = tf.zeros_like(bg_attr_std)

            bg_attr, bg_attr_kl = normal_vae(bg_attr_mean, bg_attr_std, self.attr_prior_mean, self.attr_prior_std)

            self._tensors.update(
                bg_attr_mean=bg_attr_mean,
                bg_attr_std=bg_attr_std,
                bg_attr_kl=bg_attr_kl,
                bg_attr=bg_attr)

            # --- decode ---

            _, T, H, W, _ = tf_shape(self.inp)

            background = self.background_decoder(bg_attr, 3, self.is_training)
            assert len(background.shape) == 2 and background.shape[1] == 3
            background = tf.nn.sigmoid(tf.clip_by_value(background, -10, 10))
            background = tf.tile(background[:, None, None, None, :], (1, T, H, W, 1))

        elif cfg.background_cfg.mode == "learn_and_transform":
            self.maybe_build_subnet("background_encoder")
            self.maybe_build_subnet("background_decoder")

            # --- encode ---

            n_transform_latents = 4
            n_latents = (2 * cfg.background_cfg.A, 2 * n_transform_latents)

            bg_attr, bg_transform_params = self.background_encoder(self.inp, n_latents, self.is_training)

            # --- bg attributes ---

            bg_attr_mean, bg_attr_log_std = tf.split(bg_attr, 2, axis=-1)
            bg_attr_std = self.std_nonlinearity(bg_attr_log_std)

            bg_attr, bg_attr_kl = normal_vae(bg_attr_mean, bg_attr_std, self.attr_prior_mean, self.attr_prior_std)

            # --- bg location ---

            bg_transform_params = tf.reshape(
                bg_transform_params,
                (self.batch_size, self.dynamic_n_frames, 2*n_transform_latents))

            mean, log_std = tf.split(bg_transform_params, 2, axis=2)
            std = self.std_nonlinearity(log_std)

            logits, kl = normal_vae(mean, std, 0.0, 1.0)

            # integrate across timesteps
            logits = tf.cumsum(logits, axis=1)
            logits = tf.reshape(logits, (self.batch_size*self.dynamic_n_frames, n_transform_latents))

            y, x, h, w = tf.split(logits, n_transform_latents, axis=1)
            h = (0.9 - 0.5) * tf.nn.sigmoid(h) + 0.5
            w = (0.9 - 0.5) * tf.nn.sigmoid(w) + 0.5
            y = (1 - h) * tf.nn.tanh(y)
            x = (1 - w) * tf.nn.tanh(x)

            # --- decode ---

            background = self.background_decoder(bg_attr, self.image_depth, self.is_training)
            bg_shape = cfg.background_cfg.bg_shape
            background = background[:, :bg_shape[0], :bg_shape[1], :]
            assert background.shape[1:3] == bg_shape
            background_raw = tf.nn.sigmoid(tf.clip_by_value(background, -10, 10))

            transform_constraints = snt.AffineWarpConstraints.no_shear_2d()

            warper = snt.AffineGridWarper(
                bg_shape, (self.image_height, self.image_width), transform_constraints)

            transforms = tf.concat([w, x, h, y], axis=-1)
            grid_coords = warper(transforms)

            grid_coords = tf.reshape(
                grid_coords,
                (self.batch_size, self.dynamic_n_frames, *tf_shape(grid_coords)[1:]))

            background = tf.contrib.resampler.resampler(background_raw, grid_coords)

            self._tensors.update(
                bg_attr_mean=bg_attr_mean,
                bg_attr_std=bg_attr_std,
                bg_attr_kl=bg_attr_kl,
                bg_attr=bg_attr,
                bg_y=tf.reshape(y, (self.batch_size, self.dynamic_n_frames, 1)),
                bg_x=tf.reshape(x, (self.batch_size, self.dynamic_n_frames, 1)),
                bg_h=tf.reshape(h, (self.batch_size, self.dynamic_n_frames, 1)),
                bg_w=tf.reshape(w, (self.batch_size, self.dynamic_n_frames, 1)),
                bg_transform_kl=kl,
                bg_raw=background_raw,
            )

        elif cfg.background_cfg.mode == "data":
            background = self._tensors["ground_truth_background"]

        else:
            raise Exception("Unrecognized background mode: {}.".format(cfg.background_cfg.mode))

        self._tensors["background"] = background[:, :self.dynamic_n_frames]


class BackgroundExtractor(RecurrentGridConvNet):
    bidirectional = True

    bg_head = None
    transform_head = None

    def _call(self, inp, output_size, is_training):
        if self.bg_head is None:
            self.bg_head = ConvNet(
                layers=[
                    dict(filters=None, kernel_size=1, strides=1, padding="SAME"),
                    dict(filters=None, kernel_size=1, strides=1, padding="SAME"),
                ],
                scope="bg_head"
            )

        if self.transform_head is None:
            self.transform_head = MLP(n_units=[64, 64], scope="transform_head")

        n_attr_channels, n_transform_values = output_size
        processed = super()._call(inp, n_attr_channels, is_training)
        B, F, H, W, C = tf_shape(processed)

        # Map processed to shapes (B, H, W, C) and (B, F, 2)

        bg_attrs = self.bg_head(tf.reduce_mean(processed, axis=1), None, is_training)

        transform_values = self.transform_head(
            tf.reshape(processed, (B*F, H*W*C)),
            n_transform_values, is_training)

        transform_values = tf.reshape(transform_values, (B, F, n_transform_values))

        return bg_attrs, transform_values


class SimpleVideoVAE(VideoNetwork):
    """ Encode each frame with an encoder, use a recurrent network to link between latent
        representations of different frames (in a causal direction), apply a decoder to the
        frame-wise latest representations to come up with reconstructions of each frame.

    """
    attr_prior_mean = Param()
    attr_prior_std = Param()

    A = Param()

    train_reconstruction = Param()
    reconstruction_weight = Param()

    train_kl = Param()
    kl_weight = Param()
    noisy = Param()

    build_encoder = Param()
    build_decoder = Param()
    build_cell = Param()
    flat_latent = Param()

    encoder = None
    decoder = None
    cell = None

    def __init__(self, env, updater, scope=None, **kwargs):
        super().__init__(env, updater, scope=scope)

        self.attr_prior_mean = build_scheduled_value(self.attr_prior_mean, "attr_prior_mean")
        self.attr_prior_std = build_scheduled_value(self.attr_prior_std, "attr_prior_std")

        self.reconstruction_weight = build_scheduled_value(
            self.reconstruction_weight, "reconstruction_weight")
        self.kl_weight = build_scheduled_value(self.kl_weight, "kl_weight")

        if not self.noisy and self.train_kl:
            raise Exception("If `noisy` is False, `train_kl` must also be False.")

    def build_representation(self):
        # --- init modules ---

        if self.encoder is None:
            self.encoder = self.build_encoder(scope="encoder")
            if "encoder" in self.fixed_weights:
                self.encoder.fix_variables()

        if self.cell is None and self.build_cell is not None:
            self.cell = cfg.build_cell(scope="cell")
            if "cell" in self.fixed_weights:
                self.cell.fix_variables()

        if self.decoder is None:
            self.decoder = cfg.build_decoder(scope="decoder")
            if "decoder" in self.fixed_weights:
                self.decoder.fix_variables()

        # --- encode ---

        inp_trailing_shape = tf_shape(self.inp)[2:]
        video = tf.reshape(self.inp, (self.batch_size * self.dynamic_n_frames, *inp_trailing_shape))
        encoder_output = self.encoder(video, 2 * self.A, self.is_training)

        eo_trailing_shape = tf_shape(encoder_output)[1:]
        encoder_output = tf.reshape(
            encoder_output, (self.batch_size, self.dynamic_n_frames, *eo_trailing_shape))

        if self.cell is None:
            attr = encoder_output
        else:

            if self.flat_latent:
                n_trailing_dims = int(np.prod(eo_trailing_shape))
                encoder_output = tf.reshape(
                    encoder_output, (self.batch_size, self.dynamic_n_frames, n_trailing_dims))
            else:
                raise Exception("NotImplemented")

                n_objects = int(np.prod(eo_trailing_shape[:-1]))
                D = eo_trailing_shape[-1]
                encoder_output = tf.reshape(
                    encoder_output, (self.batch_size, self.dynamic_n_frames, n_objects, D))

            encoder_output = tf.layers.flatten(encoder_output)

            attr, final_state = dynamic_rnn(
                self.cell, encoder_output, initial_state=self.cell.zero_state(self.batch_size, tf.float32),
                parallel_iterations=1, swap_memory=False, time_major=False)

        attr_mean, attr_log_std = tf.split(attr, 2, axis=-1)
        attr_std = tf.math.softplus(attr_log_std)

        if not self.noisy:
            attr_std = tf.zeros_like(attr_std)

        attr, attr_kl = normal_vae(attr_mean, attr_std, self.attr_prior_mean, self.attr_prior_std)

        self._tensors.update(attr_mean=attr_mean, attr_std=attr_std, attr_kl=attr_kl, attr=attr)

        # --- decode ---

        decoder_input = tf.reshape(attr, (self.batch_size*self.dynamic_n_frames, *tf_shape(attr)[2:]))

        reconstruction = self.decoder(decoder_input, tf_shape(self.inp)[2:], self.is_training)
        reconstruction = reconstruction[:, :self.obs_shape[1], :self.obs_shape[2], :]
        reconstruction = tf.reshape(reconstruction, (self.batch_size, self.dynamic_n_frames, *self.obs_shape[1:]))

        reconstruction = tf.nn.sigmoid(tf.clip_by_value(reconstruction, -10, 10))
        self._tensors["output"] = reconstruction

        # --- losses ---

        if self.train_kl:
            self.losses['attr_kl'] = tf_mean_sum(self._tensors["attr_kl"])

        if self.train_reconstruction:
            self._tensors['per_pixel_reconstruction_loss'] = xent_loss(pred=reconstruction, label=self.inp)
            self.losses['reconstruction'] = tf_mean_sum(self._tensors['per_pixel_reconstruction_loss'])


class SimpleVAE_RenderHook(RenderHook):
    fetches = "inp output".split()

    def __call__(self, updater):
        fetched = self._fetch(updater)
        self._plot_reconstruction(updater, fetched)

    def _plot_reconstruction(self, updater, fetched):
        inp = fetched['inp']
        output = fetched['output']

        fig_height = 20
        fig_width = 4.5 * fig_height

        diff = self.normalize_images(np.abs(inp - output).sum(axis=-1, keepdims=True) / output.shape[-1])
        xent = self.normalize_images(xent_loss(pred=output, label=inp, tf=False).sum(axis=-1, keepdims=True))

        path = self.path_for("animation", updater, ext=None)
        fig, axes, anim, path = animate(
            inp, output, diff.astype('f'), xent.astype('f'),
            figsize=(fig_width, fig_height), path=path)
        plt.close()
