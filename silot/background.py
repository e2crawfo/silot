import tensorflow as tf
from auto_yolo.models.core import normal_vae

from dps.utils import Param
from dps.utils.tf import ScopedFunction


class ScalorBackground(ScopedFunction):
    build_background_encoder = Param()
    build_background_decoder = Param()
    n_latents_per_channel = Param()

    def _call(self, inp, mask, is_training):
        self.maybe_build_subnet('background_encoder')
        self.maybe_build_subnet('background_decoder')

        combined = tf.concat([inp, mask], axis=-1)
        latent = self.background_encoder(combined, 2*self.n_latents_per_channel, is_training)
        mean, std = tf.split(latent, 2, axis=-1)
        sample, kl = normal_vae(mean, std, 0, 1)
        background = self.background_decoder(sample, None, is_training)
        return background, kl
