import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt
from orderedattrdict import AttrDict

Normal = tfp.distributions.Normal

from dps import cfg
from dps.utils import Param
from dps.utils.tf import build_scheduled_value, FIXED_COLLECTION, ScopedFunction, tf_shape

from auto_yolo.tf_ops import render_sprites, resampler_edge
from auto_yolo.models.core import (
    concrete_binary_pre_sigmoid_sample, concrete_binary_sample_kl, tf_safe_log)
from auto_yolo.models.object_layer import ObjectLayer


def extract_affine_glimpse(image, object_shape, yt, xt, ys, xs, unit_square=False):
    """
    unit_square: whether (yt, xt) are in unit square coordinates, and need to be switched.

    """
    _, *image_shape, image_depth = tf_shape(image)
    transform_constraints = snt.AffineWarpConstraints.no_shear_2d()
    warper = snt.AffineGridWarper(image_shape, object_shape, transform_constraints)

    if unit_square:
        xt = 2 * xt - 1
        yt = 2 * yt - 1

    leading_shape = tf_shape(yt)[:-1]

    _boxes = tf.concat([xs, xt, ys, yt], axis=-1)
    _boxes = tf.reshape(_boxes, (-1, 4))

    grid_coords = warper(_boxes)

    grid_coords = tf.reshape(grid_coords, (*leading_shape, *object_shape, 2))

    glimpses = resampler_edge.resampler_edge(image, grid_coords)
    glimpses = tf.reshape(glimpses, (*leading_shape, *object_shape, image_depth))

    return glimpses


class ObjectPropagationLayer(ObjectLayer):

    def __init__(self):
        pass

    def _independent_prior(self):
        return dict(
            cell_y_logit=Normal(loc=self.yx_prior_mean, scale=self.yx_prior_std),
            cell_x_logit=Normal(loc=self.yx_prior_mean, scale=self.yx_prior_std),
            h_logit=Normal(loc=self.hw_prior_mean, scale=self.hw_prior_std),
            w_logit=Normal(loc=self.hw_prior_mean, scale=self.hw_prior_std),
            attr=Normal(loc=self.attr_prior_mean, scale=self.attr_prior_std),
            z_logit=Normal(loc=self.z_prior_mean, scale=self.z_prior_std),
        )

    def _compute_kl(self, tensors, prior):
        d_yt_kl = tensors["d_yt_logit_dist"].kl_divergence(prior["d_yt_logit"])
        d_xt_kl = tensors["d_xt_logit_dist"].kl_divergence(prior["d_xt_logit"])
        d_ys_kl = tensors["d_ys_logit_dist"].kl_divergence(prior["d_ys_logit"])
        d_xs_kl = tensors["d_xs_logit_dist"].kl_divergence(prior["d_xs_logit"])

        if "d_yt" in self.no_gradient:
            d_yt_kl = tf.stop_gradient(d_yt_kl)

        if "d_xt" in self.no_gradient:
            d_xt_kl = tf.stop_gradient(d_xt_kl)

        if "d_ys" in self.no_gradient:
            d_ys_kl = tf.stop_gradient(d_ys_kl)

        if "d_xs" in self.no_gradient:
            d_xs_kl = tf.stop_gradient(d_xs_kl)

        d_box_kl = tf.concat([d_yt_kl, d_xt_kl, d_ys_kl, d_xs_kl], axis=-1)

        # --- d_attr ---

        d_attr_kl = tensors["d_attr_dist"].kl_divergence(prior["d_attr"])

        if "attr" in self.no_gradient:
            d_attr_kl = tf.stop_gradient(d_attr_kl)

        # --- z ---

        d_z_kl = tensors["z_logit_dist"].kl_divergence(prior["z_logit"])

        if "z" in self.no_gradient:
            d_z_kl = tf.stop_gradient(d_z_kl)

        if "z" in self.fixed_values:
            d_z_kl = tf.zeros_like(d_z_kl)

        # --- obj ---

        # TODO: Get object KL if appropriate

        # obj_kl_tensors = self._compute_obj_kl(tensors)

        return dict(
            d_yt_kl=d_yt_kl,
            d_xt_kl=d_xt_kl,
            d_ys_kl=d_ys_kl,
            d_xs_kl=d_xs_kl,
            d_box_kl=d_box_kl,
            d_attr_kl=d_attr_kl,
            d_z_kl=d_z_kl,
        )

    def _call(self, inp, features, objects, is_training, is_posterior):
        self.maybe_build_subnet("d_box_network", key="d_box", builder=cfg.build_lateral)
        self.maybe_build_subnet("d_attr_network", key="d_attr", builder=cfg.build_lateral)
        self.maybe_build_subnet("d_z_network", key="d_z", builder=cfg.build_lateral)
        self.maybe_build_subnet("d_obj", key="d_obj", builder=cfg.build_lateral)
        self.maybe_build_subnet("glimpse_prime_network", key="glimpse_prime", builder=cfg.build_lateral)

        self.maybe_build_subnet("glimpse_prime_encoder", builder=cfg.build_object_encoder)
        self.maybe_build_subnet("glimpse_encoder", builder=cfg.build_object_encoder)

        if not self.initialized:
            # Note this limits the re-usability of this module to images
            # with a fixed shape (the shape of the first image it is used on)
            self.image_height = int(inp.shape[-3])
            self.image_width = int(inp.shape[-2])
            self.image_depth = int(inp.shape[-1])
            self.batch_size = tf.shape(inp)[0]
            self.is_training = is_training
            self.float_is_training = tf.to_float(is_training)

        batch_size, *obj_leading_shape, _ = tf_shape(features)

        new_objects = AttrDict()

        is_posterior_tf = tf.ones_like(features[..., 0:2])
        if is_posterior:
            is_posterior_tf = is_posterior_tf * [1, 0]
        else:
            is_posterior_tf = is_posterior_tf * [0, 1]

        yt, xt, ys, xs = tf.split(objects.normalized_box, 4, axis=-1)

        if is_posterior:
            # --- predict parameters for glimpse prime ---

            glimpse_prime_params = self.glimpse_prime_network(features, 4, self.is_training)
            _yt, _xt, _ys, _xs = tf.split(glimpse_prime_params, 4, axis=-1)

            # --- obtain final parameters for glimpse prime by modifying current pose ---

            g_yt = yt + _yt
            g_xt = xt + _xt

            g_ys = ys * (1 + tf.nn.tanh(_ys))
            g_xs = xs * (1 + tf.nn.tanh(_xs))

            # --- extract glimpse prime ---

            glimpse_prime = extract_affine_glimpse(inp, self.object_shape, g_yt, g_xt, g_ys, g_xs, unit_square=True)

            # --- encode glimpse ---

            encoded_glimpse_prime = self.glimpse_prime_encoder_network(
                glimpse_prime, self.n_glimpse_features, self.is_training)

        else:
            glimpse_prime = tf.zeros((batch_size, *obj_leading_shape, *self.object_shape, self.image_depth))
            encoded_glimpse_prime = tf.zeros((batch_size, *obj_leading_shape, self.n_glimpse_features))

        # --- predict distribution for d_box ---

        d_box_inp = tf.concat([features, encoded_glimpse_prime, is_posterior_tf], axis=-1)
        d_box_params = self.d_box_network(d_box_inp, 8, self.is_training)

        d_box_mean, d_box_log_std = tf.split(d_box_params, 2, axis=-1)

        d_box_std = self.std_nonlinearity(d_box_log_std)

        d_box_mean = self.training_wheels * tf.stop_gradient(d_box_mean) + (1-self.training_wheels) * d_box_mean
        d_box_std = self.training_wheels * tf.stop_gradient(d_box_std) + (1-self.training_wheels) * d_box_std

        d_yt_mean, d_xt_mean, d_ys_mean, d_xs_mean = tf.split(d_box_mean, 4, axis=-1)
        d_yt_std, d_xt_std, d_ys_std, d_xs_std = tf.split(d_box_std, 4, axis=-1)

        d_yt_dist = Normal(loc=d_yt_mean, scale=d_yt_std)
        d_yt = d_yt_dist.sample()

        d_xt_dist = Normal(loc=d_xt_mean, scale=d_xt_std)
        d_xt = d_xt_dist.sample()

        d_ys_dist = Normal(loc=d_ys_mean, scale=d_ys_std)
        d_ys = d_ys_dist.sample()

        d_xs_dist = Normal(loc=d_xs_mean, scale=d_xs_std)
        d_xs = d_xs_dist.sample()

        if "d_yt" in self.no_gradient:
            d_yt = tf.stop_gradient(d_yt)

        if "d_xt" in self.no_gradient:
            d_xt = tf.stop_gradient(d_xt)

        if "d_ys" in self.no_gradient:
            d_ys = tf.stop_gradient(d_ys)

        if "d_xs" in self.no_gradient:
            d_xs = tf.stop_gradient(d_xs)

        new_yt = yt + d_yt
        new_xt = xt + d_xt

        new_ys = ys * (1 + tf.nn.tanh(d_ys))
        new_xs = xs * (1 + tf.nn.tanh(d_xs))

        new_box = tf.concat([new_yt, new_xt, new_ys, new_xs], axis=-1)

        new_objects.update(
            yt=new_yt,
            xt=new_xt,
            ys=new_ys,
            xs=new_xs,
            normalized_box=new_box,
            d_yt=d_yt,
            d_xt=d_xt,
            d_ys=d_ys,
            d_xs=d_xs,
            glimpse_prime=glimpse_prime,
        )

        # --- attributes ---

        # --- extract a glimpse using new box ---

        if is_posterior:
            glimpse = extract_affine_glimpse(inp, self.object_shape, yt, xt, ys, xs, unit_square=True)
            encoded_glimpse = self.glimpse_encoder_network(
                glimpse, self.n_glimpse_features, self.is_training)
        else:
            glimpse = tf.zeros((batch_size, *obj_leading_shape, *self.object_shape, self.image_depth))
            encoded_glimpse = tf.zeros((batch_size, *obj_leading_shape, self.n_glimpse_features))

        # --- predict change in attributes ---

        d_attr_inp = tf.concat([features, new_box, encoded_glimpse, is_posterior_tf], axis=-1)
        d_attr_params = self.d_attr_network(d_attr_inp, 2*self.A, self.is_training)

        d_attr_mean, d_attr_log_std = tf.split(d_attr_params, 2, axis=-1)
        d_attr_std = self.std_nonlinearity(d_attr_log_std)

        d_attr_dist = Normal(loc=d_attr_mean, scale=d_attr_std)
        d_attr = d_attr_dist.sample()

        # --- apply change in attributes ---

        new_attr = objects.attr + d_attr

        if "attr" in self.no_gradient:
            new_attr = tf.stop_gradient(new_attr)

        new_objects.update(attr=new_attr, d_attr=d_attr, glimpse=glimpse)

        # --- z ---

        d_z_inp = tf.concat([features, new_box, new_attr, encoded_glimpse, is_posterior_tf], axis=-1)
        d_z_params = self.d_z_network(d_z_inp, 2, self.is_training)

        d_z_mean, d_z_log_std = tf.split(d_z_params, 2, axis=-1)
        d_z_std = self.std_nonlinearity(d_z_log_std)

        d_z_mean = self.training_wheels * tf.stop_gradient(d_z_mean) + (1-self.training_wheels) * d_z_mean
        d_z_std = self.training_wheels * tf.stop_gradient(d_z_std) + (1-self.training_wheels) * d_z_std

        d_z_logit_dist = Normal(loc=d_z_mean, scale=d_z_std)
        d_z_logits = d_z_logit_dist.sample()

        old_z_logits = self.z_nonlinearity_inverse(objects.z)
        new_z_logits = old_z_logits + d_z_logits
        new_z = self.z_nonlinearity(new_z_logits)

        if "z" in self.no_gradient:
            new_z = tf.stop_gradient(new_z)

        if "z" in self.fixed_values:
            new_z = self.fixed_values['z'] * tf.ones_like(new_z)

        new_objects.update(
            d_z_mean=d_z_mean,
            d_z_std=d_z_std,
            z=new_z,
        )

        # --- obj ---

        d_obj_inp = tf.concat([features, new_box, new_attr, new_z, encoded_glimpse, is_posterior_tf], axis=-1)
        d_obj_logits = self.d_obj_network(d_obj_inp, 1, self.is_training)

        d_obj_logits = self.training_wheels * tf.stop_gradient(d_obj_logits) + (1-self.training_wheels) * d_obj_logits
        d_obj_logits = d_obj_logits / self.obj_temp

        d_obj_log_odds = tf.clip_by_value(d_obj_logits, -10., 10.)

        d_obj_pre_sigmoid = concrete_binary_pre_sigmoid_sample(d_obj_log_odds, self.obj_concrete_temp)
        raw_d_obj = tf.nn.sigmoid(d_obj_pre_sigmoid)

        if self.noisy:
            d_obj = (
                self.float_is_training * raw_d_obj
                + (1 - self.float_is_training) * tf.round(raw_d_obj)
            )
        else:
            d_obj = tf.round(raw_d_obj)

        new_obj = objects.obj * d_obj

        if "obj" in self.no_gradient:
            d_obj = tf.stop_gradient(d_obj)

        if "obj" in self.fixed_values:
            new_obj = self.fixed_values['obj'] * tf.ones_like(new_obj)

        new_objects.update(
            obj=new_obj,
            d_obj=d_obj,
            raw_d_obj=raw_d_obj,
            d_obj_pre_sigmoid=d_obj_pre_sigmoid,
            d_obj_log_odds=d_obj_log_odds,
            d_obj_prob=tf.nn.sigmoid(d_obj_log_odds),
        )

        # --- final ---

        new_objects.all = tf.concat(
            [new_objects.box, new_objects.attr, new_objects.z, new_objects.obj], axis=-1)

        return new_objects
