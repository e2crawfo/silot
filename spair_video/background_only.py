import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from auto_yolo.models.core import xent_loss

from dps.utils import Param, Config, animate
from dps.utils.tf import build_scheduled_value, tf_mean_sum, RenderHook

from spair_video.core import VideoNetwork


class BackgroundOnly(VideoNetwork):
    train_reconstruction = Param()
    reconstruction_weight = Param()

    train_kl = Param()
    kl_weight = Param()
    noisy = Param()

    needs_background = True

    def __init__(self, env, updater, scope=None, **kwargs):
        super().__init__(env, updater, scope=scope)

        self.reconstruction_weight = build_scheduled_value(self.reconstruction_weight, "reconstruction_weight")
        self.kl_weight = build_scheduled_value(self.kl_weight, "kl_weight")

        if not self.noisy and self.train_kl:
            raise Exception("If `noisy` is False, `train_kl` must also be False.")

    def build_representation(self):
        self._tensors["output"] = reconstruction = self._tensors["background"]

        # --- losses ---

        if self.train_reconstruction:
            self._tensors['per_pixel_reconstruction_loss'] = xent_loss(pred=reconstruction, label=self.inp)
            self.losses['reconstruction'] = tf_mean_sum(self._tensors['per_pixel_reconstruction_loss'])

        if "bg_attr_kl" in self._tensors:
            self.losses.update(
                bg_attr_kl=self.kl_weight * tf_mean_sum(self._tensors["bg_attr_kl"]),
                bg_transform_kl=self.kl_weight * tf_mean_sum(self._tensors["bg_transform_kl"]),
            )


class BackgroundOnly_RenderHook(RenderHook):
    def build_fetches(self, updater):
        return "inp output"

    @staticmethod
    def normalize_images(images):
        mx = images.reshape(*images.shape[:-3], -1).max(axis=-1)
        return images / mx[..., None, None, None]

    def __call__(self, updater):
        self.fetches = "inp output"

        fetched = self._fetch(updater)
        fetched = Config(fetched)

        inp = fetched['inp']
        output = fetched['output']
        T = inp.shape[1]
        mean_image = np.tile(inp.mean(axis=1, keepdims=True), (1, T, 1, 1, 1))

        B = inp.shape[0]

        fig_unit_size = 3

        fig_height = B * fig_unit_size
        fig_width = 7 * fig_unit_size

        diff = self.normalize_images(np.abs(inp - output).sum(axis=-1, keepdims=True))
        xent = self.normalize_images(xent_loss(pred=output, label=inp, tf=False).sum(axis=-1, keepdims=True))

        diff_mean = self.normalize_images(np.abs(mean_image - output).sum(axis=-1, keepdims=True))
        xent_mean = self.normalize_images(xent_loss(pred=mean_image, label=inp, tf=False).sum(axis=-1, keepdims=True))

        path = self.path_for("animation", updater, ext=None)

        fig, axes, anim, path = animate(
            inp, output, diff.astype('f'), xent.astype('f'), mean_image, diff_mean.astype('f'), xent_mean.astype('f'),
            figsize=(fig_width, fig_height), path=path, square_grid=False)
        plt.close()