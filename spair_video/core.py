import tensorflow as tf
import numpy as np
from matplotlib.colors import to_rgb

from dps import cfg
from dps.utils import Param
from dps.utils.tf import build_scheduled_value, FIXED_COLLECTION

from auto_yolo.models.core import normal_vae, TensorRecorder


class VideoVAE(TensorRecorder):
    fixed_weights = Param()
    fixed_values = Param()
    no_gradient = Param()

    attr_prior_mean = Param()
    attr_prior_std = Param()

    A = Param()

    train_reconstruction = Param()
    reconstruction_weight = Param()

    train_kl = Param()
    kl_weight = Param()
    noisy = Param()

    needs_background = True

    eval_funcs = dict()

    background_encoder = None
    background_decoder = None

    def __init__(self, env, updater, scope=None, **kwargs):
        self.updater = updater

        self.obs_shape = env.datasets['train'].obs_shape
        self.image_height, self.image_width, self.image_depth = self.obs_shape

        if isinstance(self.fixed_weights, str):
            self.fixed_weights = self.fixed_weights.split()

        self.attr_prior_mean = build_scheduled_value(self.attr_prior_mean, "attr_prior_mean")
        self.attr_prior_std = build_scheduled_value(self.attr_prior_std, "attr_prior_std")

        self.reconstruction_weight = build_scheduled_value(
            self.reconstruction_weight, "reconstruction_weight")
        self.kl_weight = build_scheduled_value(self.kl_weight, "kl_weight")

        if not self.noisy and self.train_kl:
            raise Exception("If `noisy` is False, `train_kl` must also be False.")

        if isinstance(self.fixed_weights, str):
            self.fixed_weights = self.fixed_weights.split()

        if isinstance(self.no_gradient, str):
            self.no_gradient = self.no_gradient.split()

        super(VideoVAE, self).__init__(scope=scope, **kwargs)

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

        inp = data["image"]

        self._tensors = dict(
            inp=inp,
            is_training=is_training,
            float_is_training=tf.to_float(is_training),
            batch_size=tf.shape(inp)[0],
        )

        if "annotations" in data:
            import pdb; pdb.set_trace()
            self._tensors.update(
                annotations=data["annotations"]["data"],
                n_annotations=data["annotations"]["shapes"][:, 0],
                n_valid_annotations=tf.reduce_sum(
                    data["annotations"]["data"][:, :, 0]
                    * data["annotations"]["mask"][:, :, 0],
                    axis=1
                )
            )

        if "label" in data:
            self._tensors.update(
                targets=data["label"],
            )

        if "background" in data:
            self._tensors.update(
                background=data["background"],
            )

        self.recorded_tensors = dict(
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

    def build_background(self):
        if self.needs_background:
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

            elif cfg.background_cfg.mode == "learn":
                if self.background_encoder is None:
                    self.background_encoder = cfg.build_background_encoder(scope="background_encoder")
                    if "background_encoder" in self.fixed_weights:
                        self.background_encoder.fix_variables()

                if self.background_decoder is None:
                    self.background_decoder = cfg.build_background_decoder(scope="background_decoder")
                    if "background_decoder" in self.fixed_weights:
                        self.background_decoder.fix_variables()

                bg_attr = self.background_encoder(self.inp, 2 * cfg.background_cfg.A, self.is_training)
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

                background = self.background_decoder(bg_attr, self.inp.shape[1:], self.is_training)

                if len(background.shape) == 2:
                    # background decoder predicts a solid colour
                    assert background.shape[1] == 3

                    background = tf.nn.sigmoid(tf.clip_by_value(background, -10, 10))
                    background = background[:, None, None, :]
                    background = tf.tile(background, (1, self.inp.shape[1], self.inp.shape[2], 1))
                else:
                    background = background[:, :self.inp.shape[1], :self.inp.shape[2], :]
                    background = tf.nn.sigmoid(tf.clip_by_value(background, -10, 10))

            elif cfg.background_cfg.mode == "data":
                background = self._tensors["background"]

            else:
                raise Exception("Unrecognized background mode: {}.".format(cfg.background_cfg.mode))

            self._tensors["background"] = background
