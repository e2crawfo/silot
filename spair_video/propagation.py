import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt
from orderedattrdict import AttrDict

Normal = tfp.distributions.Normal

from dps import cfg
from dps.utils import Param
from dps.utils.tf import tf_shape, apply_object_wise, MLP

from auto_yolo.tf_ops import resampler_edge
from auto_yolo.models.core import concrete_binary_pre_sigmoid_sample, concrete_binary_sample_kl, coords_to_image_space
from auto_yolo.models.object_layer import ObjectLayer
from auto_yolo.models.networks import SpatialAttentionLayerV2


def extract_affine_glimpse(image, object_shape, cyt, cxt, ys, xs):
    """ (yt, xt) are rectangle center. (ys, xs) are rectangle height/width """
    _, *image_shape, image_depth = tf_shape(image)
    transform_constraints = snt.AffineWarpConstraints.no_shear_2d()
    warper = snt.AffineGridWarper(image_shape, object_shape, transform_constraints)

    # change coordinate system
    cyt = 2 * cyt - 1
    cxt = 2 * cxt - 1

    leading_shape = tf_shape(cyt)[:-1]

    _boxes = tf.concat([xs, cxt, ys, cyt], axis=-1)
    _boxes = tf.reshape(_boxes, (-1, 4))

    grid_coords = warper(_boxes)

    grid_coords = tf.reshape(grid_coords, (*leading_shape, *object_shape, 2))

    glimpses = resampler_edge.resampler_edge(image, grid_coords)
    glimpses = tf.reshape(glimpses, (*leading_shape, *object_shape, image_depth))

    return glimpses


class ObjectPropagationLayer(ObjectLayer):
    n_propagated_objects = Param()
    use_glimpse = Param()
    learn_glimpse_prime = Param()
    where_t_scale = Param()
    where_s_scale = Param()

    d_yx_prior_mean = Param()
    d_yx_prior_std = Param()
    d_hw_prior_mean = Param()
    d_hw_prior_std = Param()
    d_attr_prior_mean = Param()
    d_attr_prior_std = Param()
    d_z_prior_mean = Param()
    d_z_prior_std = Param()
    d_obj_log_odds_prior = Param()

    anchor_box = Param()

    do_lateral = Param()
    n_hidden = Param()

    lateral_network = None

    def null_object_set(self, batch_size):
        new_objects = AttrDict(
            normalized_box=tf.zeros((batch_size, self.n_propagated_objects, 4)),
            attr=tf.zeros((batch_size, self.n_propagated_objects, self.A)),
            z=tf.zeros((batch_size, self.n_propagated_objects, 1)),
            obj=tf.zeros((batch_size, self.n_propagated_objects, 1)),
        )

        yt, xt, ys, xs = tf.split(new_objects.normalized_box, 4, axis=-1)

        new_objects.update(
            yt=yt,
            xt=xt,
            ys=ys,
            xs=xs,

            d_yt=yt + 0.0,
            d_xt=xt + 0.0,
            d_ys=ys + 0.0,
            d_xs=xs + 0.0,

            d_box=tf.zeros_like(new_objects.normalized_box),

            d_attr=new_objects.attr + 0.0,
            d_z=new_objects.z + 0.0,
        )

        new_objects.all = tf.concat(
            [new_objects.normalized_box, new_objects.attr, new_objects.z, new_objects.obj], axis=-1)

        return new_objects

    def _independent_prior(self):
        return dict(
            d_yt_logit_mean=self.d_yx_prior_mean,
            d_xt_logit_mean=self.d_yx_prior_mean,
            d_ys_logit_mean=self.d_hw_prior_mean,
            d_xs_logit_mean=self.d_hw_prior_mean,
            d_attr_mean=self.d_attr_prior_mean,
            d_z_logit_mean=self.d_z_prior_mean,

            d_yt_logit_std=self.d_yx_prior_std,
            d_xt_logit_std=self.d_yx_prior_std,
            d_ys_logit_std=self.d_hw_prior_std,
            d_xs_logit_std=self.d_hw_prior_std,
            d_attr_std=self.d_attr_prior_std,
            d_z_logit_std=self.d_z_prior_std,

            d_obj_log_odds=self.d_obj_log_odds_prior,
        )

    def compute_kl(self, tensors, prior=None):
        if prior is None:
            prior = self._independent_prior()

        # if "d_yt_logit_mean" not in tensors:
        #     return dict(
        #         d_yt_kl=tf.zeros_like(tensors.d_yt),
        #         d_xt_kl=tf.zeros_like(tensors.d_xt),
        #         d_ys_kl=tf.zeros_like(tensors.d_ys),
        #         d_xs_kl=tf.zeros_like(tensors.d_xs),
        #         d_attr_kl=tf.zeros_like(tensors.d_attr),
        #         d_z_kl=tf.zeros_like(tensors.d_z),
        #     )

        def normal_kl(name):
            loc_name = name + "_mean"
            scale_name = name + "_std"
            _post = Normal(loc=tensors[loc_name], scale=tensors[scale_name])
            _prior = Normal(loc=prior[loc_name], scale=prior[scale_name])
            return _post.kl_divergence(_prior)

        d_yt_kl = normal_kl("d_yt_logit")
        d_xt_kl = normal_kl("d_xt_logit")
        d_ys_kl = normal_kl("d_ys_logit")
        d_xs_kl = normal_kl("d_xs_logit")

        if "d_yt" in self.no_gradient:
            d_yt_kl = tf.stop_gradient(d_yt_kl)

        if "d_xt" in self.no_gradient:
            d_xt_kl = tf.stop_gradient(d_xt_kl)

        if "d_ys" in self.no_gradient:
            d_ys_kl = tf.stop_gradient(d_ys_kl)

        if "d_xs" in self.no_gradient:
            d_xs_kl = tf.stop_gradient(d_xs_kl)

        # --- d_attr ---

        d_attr_kl = normal_kl("d_attr")

        if "d_attr" in self.no_gradient:
            d_attr_kl = tf.stop_gradient(d_attr_kl)

        # --- z ---

        d_z_kl = normal_kl("d_z_logit")

        if "d_z" in self.no_gradient:
            d_z_kl = tf.stop_gradient(d_z_kl)

        if "d_z" in self.fixed_values:
            d_z_kl = tf.zeros_like(d_z_kl)

        # --- obj ---

        d_obj_kl = concrete_binary_sample_kl(
            tensors["d_obj_pre_sigmoid"],
            tensors["d_obj_log_odds"], self.obj_concrete_temp,
            prior["d_obj_log_odds"], self.obj_concrete_temp,
        )

        if "d_obj" in self.no_gradient:
            d_obj_kl = tf.stop_gradient(d_obj_kl)

        return dict(
            d_yt_kl=d_yt_kl,
            d_xt_kl=d_xt_kl,
            d_ys_kl=d_ys_kl,
            d_xs_kl=d_xs_kl,
            d_attr_kl=d_attr_kl,
            d_z_kl=d_z_kl,
            d_obj_kl=d_obj_kl,
        )

    def _call(self, inp, features, objects, is_training, is_posterior):
        print("\n" + "-" * 10 + " PropagationLayer(is_posterior={}) ".format(is_posterior) + "-" * 10)

        # TODO: possibly allow adjacent objects to depend on one another...this is the purpose of
        # SequentialSpair's "relational" network. Of course, maybe we hope that the input feature processing
        # is enough...but that doesn't allow taking into account the random values that are sampled.
        # But in SPAIR, I found that the lateral connections didn't matter so much...and these are analogous to those

        self.maybe_build_subnet("d_box_network", key="d_box", builder=cfg.build_lateral)
        self.maybe_build_subnet("d_attr_network", key="d_attr", builder=cfg.build_lateral)
        self.maybe_build_subnet("d_z_network", key="d_z", builder=cfg.build_lateral)
        self.maybe_build_subnet("d_obj_network", key="d_obj", builder=cfg.build_lateral)

        if self.do_lateral and self.lateral_network is None:
            self.lateral_network = SpatialAttentionLayerV2(
                n_hidden=self.n_hidden,
                build_mlp=lambda scope: MLP(n_units=[self.n_hidden, self.n_hidden], scope=scope),
                do_object_wise=True,
                scope="lateral_network",
            )

        if self.use_glimpse:
            if self.learn_glimpse_prime:
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

        if self.do_lateral:
            batch_size, n_objects, _ = tf_shape(features)

            new_objects = []

            for i in range(n_objects):
                # apply lateral to running objects with the feature vector for
                # the current object

                _features = features[:, i:i+1, :]

                if i > 0:
                    normalized_box = tf.concat([o.normalized_box for o in new_objects], axis=1)
                    attr = tf.concat([o.attr for o in new_objects], axis=1)
                    z = tf.concat([o.z for o in new_objects], axis=1)
                    obj = tf.concat([o.obj for o in new_objects], axis=1)
                    completed_features = tf.concat([normalized_box[:, :, 2:], attr, z, obj], axis=2)
                    completed_locs = normalized_box[:, :, :2]

                    current_features = tf.concat(
                        [objects.normalized_box[:, i:i+1, 2:],
                         objects.attr[:, i:i+1],
                         objects.z[:, i:i+1],
                         objects.obj[:, i:i+1]],
                        axis=2)
                    current_locs = objects.normalized_box[:, i:i+1, :2]

                    # if i > max_completed_objects:
                    #     # top_k_indices
                    #     # squared_distances = tf.reduce_sum((completed_locs - current_locs)**2, axis=2)
                    #     # _, top_k_indices = tf.nn.top_k(squared_distances, k=max_completed_objects, sorted=False)

                    _features = self.lateral_network(
                        completed_locs, completed_features,
                        current_locs, current_features,
                        is_training)

                _objects = AttrDict(
                    normalized_box=objects.normalized_box[:, i:i+1],
                    attr=objects.attr[:, i:i+1],
                    z=objects.z[:, i:i+1],
                    obj=objects.obj[:, i:i+1],
                )

                _new_objects = self._body(inp, _features, _objects, is_posterior)
                new_objects.append(_new_objects)

            _new_objects = AttrDict()
            for k in new_objects[0]:
                _new_objects[k] = tf.concat([no[k] for no in new_objects], axis=1)
            return _new_objects

        else:
            return self._body(inp, features, objects, is_posterior)

    def _body(self, inp, features, objects, is_posterior):
        batch_size, n_objects, _ = tf_shape(features)

        new_objects = AttrDict()

        is_posterior_tf = tf.ones_like(features[..., 0:2])
        if is_posterior:
            is_posterior_tf = is_posterior_tf * [1, 0]
        else:
            is_posterior_tf = is_posterior_tf * [0, 1]

        base_features = tf.concat([features, is_posterior_tf], axis=-1)

        cyt, cxt, ys, xs = tf.split(objects.normalized_box, 4, axis=-1)

        if self.use_glimpse and self.learn_glimpse_prime:
            # Do this regardless of is_posterior, otherwise ScopedFunction gets messed up
            glimpse_prime_params = apply_object_wise(self.glimpse_prime_network, base_features, 4, self.is_training)
        else:
            glimpse_prime_params = tf.zeros_like(base_features[..., :4])

        if is_posterior and self.use_glimpse:

            if self.learn_glimpse_prime:
                # --- obtain final parameters for glimpse prime by modifying current pose ---
                _yt, _xt, _ys, _xs = tf.split(glimpse_prime_params, 4, axis=-1)

                g_yt = cyt + self.where_t_scale * tf.nn.tanh(_yt)
                g_xt = cxt + self.where_t_scale * tf.nn.tanh(_xt)

                g_ys = ys * (1 + self.where_s_scale * tf.nn.tanh(_ys))
                g_xs = xs * (1 + self.where_s_scale * tf.nn.tanh(_xs))
            else:
                g_yt = cyt
                g_xt = cxt
                g_ys = 2.0 * ys
                g_xs = 2.0 * xs

            # --- extract glimpse prime ---

            _, image_height, image_width, _ = tf_shape(inp)
            g_yt, g_xt, g_ys, g_xs = coords_to_image_space(
                g_yt, g_xt, g_ys, g_xs, (image_height, image_width), self.anchor_box, top_left=False)
            glimpse_prime = extract_affine_glimpse(inp, self.object_shape, g_yt, g_xt, g_ys, g_xs)
        else:
            g_yt = tf.zeros_like(cyt)
            g_xt = tf.zeros_like(cxt)
            g_ys = tf.zeros_like(ys)
            g_xs = tf.zeros_like(xs)
            glimpse_prime = tf.zeros((batch_size, n_objects, *self.object_shape, self.image_depth))

        new_objects.update(
            g_yt=g_yt,
            g_xt=g_xt,
            g_ys=g_ys,
            g_xs=g_xs,
        )

        # --- encode glimpse ---

        if self.use_glimpse:
            encoded_glimpse_prime = apply_object_wise(
                self.glimpse_prime_encoder, glimpse_prime, self.A, self.is_training, n_trailing_dims=3)

        if not (self.use_glimpse and is_posterior):
            encoded_glimpse_prime = tf.zeros((batch_size, n_objects, self.A), dtype=tf.float32)

        # --- predict distribution for d_box ---

        d_box_inp = tf.concat([base_features, encoded_glimpse_prime], axis=-1)
        d_box_params = apply_object_wise(self.d_box_network, d_box_inp, 8, self.is_training)

        d_box_mean, d_box_log_std = tf.split(d_box_params, 2, axis=-1)

        d_box_std = self.std_nonlinearity(d_box_log_std)

        d_box_mean = self.training_wheels * tf.stop_gradient(d_box_mean) + (1-self.training_wheels) * d_box_mean
        d_box_std = self.training_wheels * tf.stop_gradient(d_box_std) + (1-self.training_wheels) * d_box_std

        d_yt_mean, d_xt_mean, d_ys_mean, d_xs_mean = tf.split(d_box_mean, 4, axis=-1)
        d_yt_std, d_xt_std, d_ys_std, d_xs_std = tf.split(d_box_std, 4, axis=-1)

        d_yt_logit = Normal(loc=d_yt_mean, scale=d_yt_std).sample()
        d_xt_logit = Normal(loc=d_xt_mean, scale=d_xt_std).sample()
        d_ys_logit = Normal(loc=d_ys_mean, scale=d_ys_std).sample()
        d_xs_logit = Normal(loc=d_xs_mean, scale=d_xs_std).sample()

        if "d_yt" in self.no_gradient:
            d_yt_logit = tf.stop_gradient(d_yt_logit)

        if "d_xt" in self.no_gradient:
            d_xt_logit = tf.stop_gradient(d_xt_logit)

        if "d_ys" in self.no_gradient:
            d_ys_logit = tf.stop_gradient(d_ys_logit)

        if "d_xs" in self.no_gradient:
            d_xs_logit = tf.stop_gradient(d_xs_logit)

        new_cyt = cyt + self.where_t_scale * tf.nn.tanh(d_yt_logit)
        new_cxt = cxt + self.where_t_scale * tf.nn.tanh(d_xt_logit)

        new_ys = ys * (1 + self.where_s_scale * tf.nn.tanh(d_ys_logit))
        new_xs = xs * (1 + self.where_s_scale * tf.nn.tanh(d_xs_logit))

        d_box = tf.concat([d_yt_logit, d_xt_logit, d_ys_logit, d_xs_logit], axis=-1)

        new_box = tf.concat([new_cyt, new_cxt, new_ys, new_xs], axis=-1)

        new_objects.update(
            yt=new_cyt,
            xt=new_cxt,
            ys=new_ys,
            xs=new_xs,
            normalized_box=new_box,

            d_yt_logit=d_yt_logit,
            d_xt_logit=d_xt_logit,
            d_ys_logit=d_ys_logit,
            d_xs_logit=d_xs_logit,

            glimpse_prime=glimpse_prime,

            d_yt_logit_mean=d_yt_mean,
            d_xt_logit_mean=d_xt_mean,
            d_ys_logit_mean=d_ys_mean,
            d_xs_logit_mean=d_xs_mean,

            d_yt_logit_std=d_yt_std,
            d_xt_logit_std=d_xt_std,
            d_ys_logit_std=d_ys_std,
            d_xs_logit_std=d_xs_std,
        )

        # --- attributes ---

        # --- extract a glimpse using new box ---

        if is_posterior and self.use_glimpse:
            _, image_height, image_width, _ = tf_shape(inp)
            _new_cyt, _new_cxt, _new_ys, _new_xs = coords_to_image_space(
                new_cyt, new_cxt, new_ys, new_xs, (image_height, image_width), self.anchor_box, top_left=False)

            glimpse = extract_affine_glimpse(inp, self.object_shape, _new_cyt, _new_cxt, _new_ys, _new_xs)

        else:
            glimpse = tf.zeros((batch_size, n_objects, *self.object_shape, self.image_depth))

        if self.use_glimpse:
            encoded_glimpse = apply_object_wise(
                self.glimpse_encoder, glimpse, self.A, self.is_training, n_trailing_dims=3)

        if not (self.use_glimpse and is_posterior):
            encoded_glimpse = tf.zeros((batch_size, n_objects, self.A), dtype=tf.float32)

        # --- predict change in attributes ---

        # We shouldn't condition on new_box, not spatially invariant
        # d_attr_inp = tf.concat([base_features, new_box, encoded_glimpse], axis=-1)
        d_attr_inp = tf.concat([base_features, d_box, encoded_glimpse], axis=-1)
        d_attr_params = apply_object_wise(self.d_attr_network, d_attr_inp, 2*self.A, self.is_training)

        d_attr_mean, d_attr_log_std = tf.split(d_attr_params, 2, axis=-1)
        d_attr_std = self.std_nonlinearity(d_attr_log_std)

        d_attr = Normal(loc=d_attr_mean, scale=d_attr_std).sample()

        # --- apply change in attributes ---

        new_attr = objects.attr + d_attr

        if "d_attr" in self.no_gradient:
            new_attr = tf.stop_gradient(new_attr)

        new_objects.update(
            attr=new_attr,
            d_attr=d_attr,
            d_attr_mean=d_attr_mean,
            d_attr_std=d_attr_std,
            glimpse=glimpse
        )

        # --- z ---

        # We shouldn't condition on new_box, not spatially invariant
        # d_z_inp = tf.concat([base_features, new_box, new_attr, encoded_glimpse], axis=-1)
        d_z_inp = tf.concat([base_features, d_box, new_attr, encoded_glimpse], axis=-1)
        d_z_params = apply_object_wise(self.d_z_network, d_z_inp, 2, self.is_training)

        d_z_mean, d_z_log_std = tf.split(d_z_params, 2, axis=-1)
        d_z_std = self.std_nonlinearity(d_z_log_std)

        d_z_mean = self.training_wheels * tf.stop_gradient(d_z_mean) + (1-self.training_wheels) * d_z_mean
        d_z_std = self.training_wheels * tf.stop_gradient(d_z_std) + (1-self.training_wheels) * d_z_std

        d_z_logits = Normal(loc=d_z_mean, scale=d_z_std).sample()

        if "d_z" in self.no_gradient:
            d_z_logits = tf.stop_gradient(d_z_logits)

        if "d_z" in self.fixed_values:
            d_z_logits = self.fixed_values['d_z'] * tf.ones_like(d_z_logits)

        old_z_logits = self.z_nonlinearity_inverse(objects.z)
        new_z_logits = old_z_logits + d_z_logits
        new_z = self.z_nonlinearity(new_z_logits)

        new_objects.update(
            z=new_z,
            d_z_logit=d_z_logits,
            d_z_logit_mean=d_z_mean,
            d_z_logit_std=d_z_std,
        )

        # --- obj ---

        # We shouldn't condition on new_box, not spatially invariant
        # d_obj_inp = tf.concat([base_features, new_box, new_attr, new_z, encoded_glimpse], axis=-1)
        d_obj_inp = tf.concat([base_features, d_box, new_attr, new_z, encoded_glimpse], axis=-1)
        d_obj_logits = apply_object_wise(self.d_obj_network, d_obj_inp, 1, self.is_training)

        d_obj_logits = self.training_wheels * tf.stop_gradient(d_obj_logits) + (1-self.training_wheels) * d_obj_logits
        d_obj_log_odds = tf.clip_by_value(d_obj_logits / self.obj_temp, -10., 10.)

        if self.noisy:
            d_obj_pre_sigmoid = concrete_binary_pre_sigmoid_sample(d_obj_log_odds, self.obj_concrete_temp)
        else:
            d_obj_pre_sigmoid = d_obj_log_odds

        d_obj = tf.nn.sigmoid(d_obj_pre_sigmoid)

        new_obj = objects.obj * d_obj
        new_render_obj = (
            self.float_is_training * new_obj
            + (1 - self.float_is_training) * tf.round(new_obj)
        )

        new_objects.update(
            d_obj_log_odds=d_obj_log_odds,
            d_obj_prob=tf.nn.sigmoid(d_obj_log_odds),
            d_obj_pre_sigmoid=d_obj_pre_sigmoid,
            d_obj=d_obj,
            obj=new_obj,
            render_obj=new_render_obj,
        )

        # --- final ---

        new_objects.all = tf.concat(
            [new_objects.normalized_box, new_objects.attr, new_objects.z, new_objects.obj], axis=-1)

        return new_objects
