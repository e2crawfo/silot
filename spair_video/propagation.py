import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt
from orderedattrdict import AttrDict

Normal = tfp.distributions.Normal

from dps import cfg
from dps.utils import Param
from dps.utils.tf import tf_shape, apply_object_wise, MLP

from auto_yolo.tf_ops import resampler_edge
from auto_yolo.models.core import concrete_binary_pre_sigmoid_sample, coords_to_image_space, concrete_binary_sample_kl
from auto_yolo.models.object_layer import ObjectLayer
from auto_yolo.models.networks import SpatialAttentionLayerV2


def extract_affine_glimpse(image, object_shape, cyt, cxt, ys, xs, edge_resampler):
    """ (cyt, cxt) are rectangle center. (ys, xs) are rectangle height/width """
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

    if edge_resampler:
        glimpses = resampler_edge.resampler_edge(image, grid_coords)
    else:
        glimpses = tf.contrib.resampler.resampler(image, grid_coords)

    glimpses = tf.reshape(glimpses, (*leading_shape, *object_shape, image_depth))

    return glimpses


class ObjectPropagationLayer(ObjectLayer):
    n_prop_objects = Param()
    learn_glimpse_prime = Param()
    where_t_scale = Param()
    glimpse_prime_scale = Param()

    d_yx_prior_mean = Param()
    d_yx_prior_std = Param()

    hw_prior_mean = Param()
    hw_prior_std = Param()
    min_hw = Param()
    max_hw = Param()

    d_attr_prior_mean = Param()
    d_attr_prior_std = Param()
    gate_d_attr = Param()

    d_z_prior_mean = Param()
    d_z_prior_std = Param()

    anchor_box = Param()

    do_lateral = Param()
    n_hidden = Param()
    use_abs_posn = Param()
    edge_resampler = Param()

    lateral_network = None

    def __init__(self, cell, **kwargs):
        self.cell = cell
        super().__init__(**kwargs)

    def null_object_set(self, batch_size):
        n_prop_objects = self.n_prop_objects

        new_objects = AttrDict(
            normalized_box=tf.zeros((batch_size, n_prop_objects, 4)),
            attr=tf.zeros((batch_size, n_prop_objects, self.A)),
            z=tf.zeros((batch_size, n_prop_objects, 1)),
            obj=tf.zeros((batch_size, n_prop_objects, 1)),
        )

        yt, xt, ys, xs = tf.split(new_objects.normalized_box, 4, axis=-1)

        new_objects.update(
            abs_posn=new_objects.normalized_box[..., :2] + 0.0,

            yt=yt,
            xt=xt,
            ys=ys,
            xs=xs,

            ys_logit=ys + 0.0,
            xs_logit=xs + 0.0,

            d_yt=yt + 0.0,
            d_xt=xt + 0.0,
            d_ys=ys + 0.0,
            d_xs=xs + 0.0,

            d_attr=new_objects.attr + 0.0,
            d_z=new_objects.z + 0.0,
            z_logit=new_objects.z + 0.0,
        )

        prop_state = self.cell.initial_state(batch_size*n_prop_objects, tf.float32)
        trailing_shape = tf_shape(prop_state)[1:]
        new_objects.prop_state = tf.reshape(prop_state, (batch_size, n_prop_objects, *trailing_shape))
        new_objects.prior_prop_state = new_objects.prop_state

        return new_objects

    def _independent_prior(self):
        return dict(
            d_yt_logit_mean=self.d_yx_prior_mean,
            d_xt_logit_mean=self.d_yx_prior_mean,
            ys_logit_mean=self.hw_prior_mean,
            xs_logit_mean=self.hw_prior_mean,
            d_attr_mean=self.d_attr_prior_mean,
            d_z_logit_mean=self.d_z_prior_mean,

            d_yt_logit_std=self.d_yx_prior_std,
            d_xt_logit_std=self.d_yx_prior_std,
            ys_logit_std=self.hw_prior_std,
            xs_logit_std=self.hw_prior_std,
            d_attr_std=self.d_attr_prior_std,
            d_z_logit_std=self.d_z_prior_std,
        )

    def compute_kl(self, tensors, prior=None, do_obj=True):
        simple_obj = prior is not None

        if prior is None:
            prior = self._independent_prior()

        def normal_kl(name):
            loc_name = name + "_mean"
            scale_name = name + "_std"
            _post = Normal(loc=tensors[loc_name], scale=tensors[scale_name])
            _prior = Normal(loc=prior[loc_name], scale=prior[scale_name])
            return _post.kl_divergence(_prior)

        d_yt_kl = normal_kl("d_yt_logit")
        d_xt_kl = normal_kl("d_xt_logit")
        ys_kl = normal_kl("ys_logit")
        xs_kl = normal_kl("xs_logit")
        d_attr_kl = normal_kl("d_attr")
        d_z_kl = normal_kl("d_z_logit")

        kl = dict(
            d_yt_kl=d_yt_kl,
            d_xt_kl=d_xt_kl,
            ys_kl=ys_kl,
            xs_kl=xs_kl,
            d_attr_kl=d_attr_kl,
            d_z_kl=d_z_kl,
        )

        if do_obj:
            if simple_obj:
                kl['d_obj_kl'] = concrete_binary_sample_kl(
                    tensors["d_obj_pre_sigmoid"],
                    tensors["d_obj_log_odds"], self.obj_concrete_temp,
                    prior["d_obj_log_odds"], self.obj_concrete_temp,
                )
            else:
                kl['d_obj_kl'] = self._compute_obj_kl(tensors)

        return kl

    def _build_networks(self):
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

        if self.learn_glimpse_prime:
            self.maybe_build_subnet("glimpse_prime_network", key="glimpse_prime", builder=cfg.build_lateral)
        self.maybe_build_subnet("glimpse_prime_encoder", builder=cfg.build_object_encoder)
        self.maybe_build_subnet("glimpse_encoder", builder=cfg.build_object_encoder)

    def _call(self, inp, features, objects, is_training, is_posterior):
        print("\n" + "-" * 10 + " PropagationLayer(is_posterior={}) ".format(is_posterior) + "-" * 10)

        self._build_networks()

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
            # hasn't been updated to make use of abs_posn
            raise Exception("NotImplemented.")

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

        if self.learn_glimpse_prime:
            # Do this regardless of is_posterior, otherwise ScopedFunction gets messed up
            glimpse_prime_params = apply_object_wise(
                self.glimpse_prime_network, base_features, output_size=4, is_training=self.is_training)
        else:
            glimpse_prime_params = tf.zeros_like(base_features[..., :4])

        if is_posterior:

            if self.learn_glimpse_prime:
                # --- obtain final parameters for glimpse prime by modifying current pose ---
                _yt, _xt, _ys, _xs = tf.split(glimpse_prime_params, 4, axis=-1)

                # This is how it is done in SQAIR
                g_yt = cyt + 0.1 * _yt
                g_xt = cxt + 0.1 * _xt
                g_ys = ys + 0.1 * _ys
                g_xs = xs + 0.1 * _xs

                # g_yt = cyt + self.glimpse_prime_scale * tf.nn.tanh(_yt)
                # g_xt = cxt + self.glimpse_prime_scale * tf.nn.tanh(_xt)
                # g_ys = ys + self.glimpse_prime_scale * tf.nn.tanh(_ys)
                # g_xs = xs + self.glimpse_prime_scale * tf.nn.tanh(_xs)
            else:
                g_yt = cyt
                g_xt = cxt
                g_ys = self.glimpse_prime_scale * ys
                g_xs = self.glimpse_prime_scale * xs

            # --- extract glimpse prime ---

            _, image_height, image_width, _ = tf_shape(inp)
            g_yt, g_xt, g_ys, g_xs = coords_to_image_space(
                g_yt, g_xt, g_ys, g_xs, (image_height, image_width), self.anchor_box, top_left=False)
            glimpse_prime = extract_affine_glimpse(inp, self.object_shape, g_yt, g_xt, g_ys, g_xs, self.edge_resampler)
        else:
            g_yt = tf.zeros_like(cyt)
            g_xt = tf.zeros_like(cxt)
            g_ys = tf.zeros_like(ys)
            g_xs = tf.zeros_like(xs)
            glimpse_prime = tf.zeros((batch_size, n_objects, *self.object_shape, self.image_depth))

        new_objects.update(
            glimpse_prime_box=tf.concat([g_yt, g_xt, g_ys, g_xs], axis=-1),
        )

        # --- encode glimpse ---

        encoded_glimpse_prime = apply_object_wise(
            self.glimpse_prime_encoder, glimpse_prime,
            n_trailing_dims=3, output_size=self.A, is_training=self.is_training)

        if not is_posterior:
            encoded_glimpse_prime = tf.zeros((batch_size, n_objects, self.A), dtype=tf.float32)

        # --- position and scale ---

        d_box_inp = tf.concat([base_features, encoded_glimpse_prime], axis=-1)
        d_box_params = apply_object_wise(self.d_box_network, d_box_inp, output_size=8, is_training=self.is_training)

        d_box_mean, d_box_log_std = tf.split(d_box_params, 2, axis=-1)

        d_box_std = self.std_nonlinearity(d_box_log_std)

        d_box_mean = self.training_wheels * tf.stop_gradient(d_box_mean) + (1-self.training_wheels) * d_box_mean
        d_box_std = self.training_wheels * tf.stop_gradient(d_box_std) + (1-self.training_wheels) * d_box_std

        d_yt_mean, d_xt_mean, d_ys, d_xs = tf.split(d_box_mean, 4, axis=-1)
        d_yt_std, d_xt_std, ys_std, xs_std = tf.split(d_box_std, 4, axis=-1)

        # --- position ---

        # We predict position a bit differently from scale. For scale we want to put a prior on the actual value of
        # the scale, whereas for position we want to put a prior on the difference in position over timesteps.

        d_yt_logit = Normal(loc=d_yt_mean, scale=d_yt_std).sample()
        d_xt_logit = Normal(loc=d_xt_mean, scale=d_xt_std).sample()

        d_yt = self.where_t_scale * tf.nn.tanh(d_yt_logit)
        d_xt = self.where_t_scale * tf.nn.tanh(d_xt_logit)

        new_cyt = cyt + d_yt
        new_cxt = cxt + d_xt

        new_abs_posn = objects.abs_posn + tf.concat([d_yt, d_xt], axis=-1)

        # --- scale ---

        new_ys_mean = objects.ys_logit + d_ys
        new_xs_mean = objects.xs_logit + d_xs

        new_ys_logit = Normal(loc=new_ys_mean, scale=ys_std).sample()
        new_xs_logit = Normal(loc=new_xs_mean, scale=xs_std).sample()

        new_ys = float(self.max_hw - self.min_hw) * tf.nn.sigmoid(tf.clip_by_value(new_ys_logit, -10, 10)) + self.min_hw
        new_xs = float(self.max_hw - self.min_hw) * tf.nn.sigmoid(tf.clip_by_value(new_xs_logit, -10, 10)) + self.min_hw

        # Used for conditioning
        if self.use_abs_posn:
            box_params = tf.concat([new_abs_posn, d_yt_logit, d_xt_logit, new_ys_logit, new_xs_logit], axis=-1)
        else:
            box_params = tf.concat([d_yt_logit, d_xt_logit, new_ys_logit, new_xs_logit], axis=-1)

        new_objects.update(
            abs_posn=new_abs_posn,

            yt=new_cyt,
            xt=new_cxt,
            ys=new_ys,
            xs=new_xs,
            normalized_box=tf.concat([new_cyt, new_cxt, new_ys, new_xs], axis=-1),

            d_yt_logit=d_yt_logit,
            d_xt_logit=d_xt_logit,
            ys_logit=new_ys_logit,
            xs_logit=new_xs_logit,

            d_yt_logit_mean=d_yt_mean,
            d_xt_logit_mean=d_xt_mean,
            ys_logit_mean=new_ys_mean,
            xs_logit_mean=new_xs_mean,

            d_yt_logit_std=d_yt_std,
            d_xt_logit_std=d_xt_std,
            ys_logit_std=ys_std,
            xs_logit_std=xs_std,

            glimpse_prime=glimpse_prime,
        )

        # --- attributes ---

        # --- extract a glimpse using new box ---

        if is_posterior:
            _, image_height, image_width, _ = tf_shape(inp)
            _new_cyt, _new_cxt, _new_ys, _new_xs = coords_to_image_space(
                new_cyt, new_cxt, new_ys, new_xs, (image_height, image_width), self.anchor_box, top_left=False)

            glimpse = extract_affine_glimpse(
                inp, self.object_shape, _new_cyt, _new_cxt, _new_ys, _new_xs, self.edge_resampler)

        else:
            glimpse = tf.zeros((batch_size, n_objects, *self.object_shape, self.image_depth))

        encoded_glimpse = apply_object_wise(
            self.glimpse_encoder, glimpse, n_trailing_dims=3, output_size=self.A, is_training=self.is_training)

        if not is_posterior:
            encoded_glimpse = tf.zeros((batch_size, n_objects, self.A), dtype=tf.float32)

        # --- predict change in attributes ---

        d_attr_inp = tf.concat([base_features, box_params, encoded_glimpse], axis=-1)
        d_attr_params = apply_object_wise(
            self.d_attr_network, d_attr_inp, output_size=2*self.A+1, is_training=self.is_training)

        d_attr_mean, d_attr_log_std, gate_logit = tf.split(d_attr_params, [self.A, self.A, 1], axis=-1)
        d_attr_std = self.std_nonlinearity(d_attr_log_std)

        gate = tf.nn.sigmoid(gate_logit)

        if self.gate_d_attr:
            d_attr_mean *= gate

        d_attr = Normal(loc=d_attr_mean, scale=d_attr_std).sample()

        # --- apply change in attributes ---

        new_attr = objects.attr + d_attr

        new_objects.update(
            attr=new_attr,
            d_attr=d_attr,
            d_attr_mean=d_attr_mean,
            d_attr_std=d_attr_std,
            glimpse=glimpse,
            d_attr_gate=gate,
        )

        # --- z ---

        d_z_inp = tf.concat([base_features, box_params, new_attr, encoded_glimpse], axis=-1)
        d_z_params = apply_object_wise(self.d_z_network, d_z_inp, output_size=2, is_training=self.is_training)

        d_z_mean, d_z_log_std = tf.split(d_z_params, 2, axis=-1)
        d_z_std = self.std_nonlinearity(d_z_log_std)

        d_z_mean = self.training_wheels * tf.stop_gradient(d_z_mean) + (1-self.training_wheels) * d_z_mean
        d_z_std = self.training_wheels * tf.stop_gradient(d_z_std) + (1-self.training_wheels) * d_z_std

        d_z_logit = Normal(loc=d_z_mean, scale=d_z_std).sample()

        new_z_logit = objects.z_logit + d_z_logit
        new_z = self.z_nonlinearity(new_z_logit)

        new_objects.update(
            z=new_z,
            z_logit=new_z_logit,
            d_z_logit=d_z_logit,
            d_z_logit_mean=d_z_mean,
            d_z_logit_std=d_z_std,
        )

        # --- obj ---

        d_obj_inp = tf.concat([base_features, box_params, new_attr, new_z, encoded_glimpse], axis=-1)
        d_obj_logit = apply_object_wise(self.d_obj_network, d_obj_inp, output_size=1, is_training=self.is_training)

        d_obj_logit = self.training_wheels * tf.stop_gradient(d_obj_logit) + (1-self.training_wheels) * d_obj_logit
        d_obj_log_odds = tf.clip_by_value(d_obj_logit / self.obj_temp, -10., 10.)

        d_obj_pre_sigmoid = (
            self._noisy * concrete_binary_pre_sigmoid_sample(d_obj_log_odds, self.obj_concrete_temp)
            + (1 - self._noisy) * d_obj_log_odds
        )

        d_obj = tf.nn.sigmoid(d_obj_pre_sigmoid)

        new_obj = objects.obj * d_obj
        new_render_obj = new_obj

        new_objects.update(
            d_obj_log_odds=d_obj_log_odds,
            d_obj_prob=tf.nn.sigmoid(d_obj_log_odds),
            d_obj_pre_sigmoid=d_obj_pre_sigmoid,
            d_obj=d_obj,
            obj=new_obj,
            render_obj=new_render_obj,
        )

        # --- update each object's hidden state --

        cell_input = tf.concat([box_params, new_attr, new_z, new_obj], axis=-1)

        if is_posterior:
            _, new_objects.prop_state = apply_object_wise(self.cell, cell_input, objects.prop_state)
            new_objects.prior_prop_state = new_objects.prop_state
        else:
            _, new_objects.prior_prop_state = apply_object_wise(self.cell, cell_input, objects.prior_prop_state)
            new_objects.prop_state = new_objects.prior_prop_state

        return new_objects


class SQAIRPropagationLayer(ObjectPropagationLayer):
    """ Reimplementation of SQAIR's propagation system, made to be compatible with SILOT """

    def _build_networks(self):
        self.maybe_build_subnet("d_box_network", key="d_box", builder=cfg.build_lateral)
        self.maybe_build_subnet("predict_attr_inp", builder=cfg.build_lateral)
        self.maybe_build_subnet("predict_attr_temp", builder=cfg.build_lateral)
        self.maybe_build_subnet("d_z_network", key="d_z", builder=cfg.build_lateral)
        self.maybe_build_subnet("d_obj_network", key="d_obj", builder=cfg.build_lateral)

        if self.do_lateral and self.lateral_network is None:
            self.lateral_network = SpatialAttentionLayerV2(
                n_hidden=self.n_hidden,
                build_mlp=lambda scope: MLP(n_units=[self.n_hidden, self.n_hidden], scope=scope),
                do_object_wise=True,
                scope="lateral_network",
            )

        self.maybe_build_subnet("glimpse_prime_network", key="glimpse_prime", builder=cfg.build_lateral)
        self.maybe_build_subnet("glimpse_prime_encoder", builder=cfg.build_object_encoder)
        self.maybe_build_subnet("glimpse_encoder", builder=cfg.build_object_encoder)

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

        # Do this regardless of is_posterior, otherwise ScopedFunction gets messed up
        glimpse_dim = self.object_shape[0] * self.object_shape[1]
        glimpse_prime_params = apply_object_wise(
            self.glimpse_prime_network, base_features, output_size=4+2*glimpse_dim,
            is_training=self.is_training)

        glimpse_prime_params, glimpse_prime_mask_logit, glimpse_mask_logit = \
            tf.split(glimpse_prime_params, [4, glimpse_dim, glimpse_dim], axis=-1)

        if is_posterior:
            # --- obtain final parameters for glimpse prime by modifying current pose ---
            _yt, _xt, _ys, _xs = tf.split(glimpse_prime_params, 4, axis=-1)

            g_yt = cyt + 0.1 * _yt
            g_xt = cxt + 0.1 * _xt
            g_ys = ys + 0.1 * _ys
            g_xs = xs + 0.1 * _xs

            # --- extract glimpse prime ---

            _, image_height, image_width, _ = tf_shape(inp)
            g_yt, g_xt, g_ys, g_xs = coords_to_image_space(
                g_yt, g_xt, g_ys, g_xs, (image_height, image_width), self.anchor_box, top_left=False)
            glimpse_prime = extract_affine_glimpse(inp, self.object_shape, g_yt, g_xt, g_ys, g_xs, self.edge_resampler)
        else:
            g_yt = tf.zeros_like(cyt)
            g_xt = tf.zeros_like(cxt)
            g_ys = tf.zeros_like(ys)
            g_xs = tf.zeros_like(xs)
            glimpse_prime = tf.zeros((batch_size, n_objects, *self.object_shape, self.image_depth))

        glimpse_prime_mask = tf.nn.sigmoid(glimpse_prime_mask_logit + 1.)
        leading_mask_shape = tf_shape(glimpse_prime)[:-1]
        glimpse_prime_mask = tf.reshape(glimpse_prime_mask, (*leading_mask_shape, 1))

        glimpse_prime *= glimpse_prime_mask

        new_objects.update(
            glimpse_prime_box=tf.concat([g_yt, g_xt, g_ys, g_xs], axis=-1),
            glimpse_prime=glimpse_prime,
            glimpse_prime_mask=glimpse_prime_mask,
        )

        # --- encode glimpse ---

        encoded_glimpse_prime = apply_object_wise(
            self.glimpse_prime_encoder, glimpse_prime,
            n_trailing_dims=3, output_size=self.A, is_training=self.is_training)

        if not is_posterior:
            encoded_glimpse_prime = tf.zeros((batch_size, n_objects, self.A), dtype=tf.float32)

        # --- position and scale ---

        # roughly:
        # base_features == temporal_state, encoded_glimpse_prime == hidden_output
        # hidden_output conditions on encoded_glimpse, and that's the only place encoded_glimpse_prime is used.

        # Here SQAIR conditions on the actual location values from the previous timestep, but we leave that out for now.
        d_box_inp = tf.concat([base_features, encoded_glimpse_prime], axis=-1)
        d_box_params = apply_object_wise(self.d_box_network, d_box_inp, output_size=8, is_training=self.is_training)

        d_box_mean, d_box_log_std = tf.split(d_box_params, 2, axis=-1)

        d_box_std = self.std_nonlinearity(d_box_log_std)

        d_box_mean = self.training_wheels * tf.stop_gradient(d_box_mean) + (1-self.training_wheels) * d_box_mean
        d_box_std = self.training_wheels * tf.stop_gradient(d_box_std) + (1-self.training_wheels) * d_box_std

        d_yt_mean, d_xt_mean, d_ys, d_xs = tf.split(d_box_mean, 4, axis=-1)
        d_yt_std, d_xt_std, ys_std, xs_std = tf.split(d_box_std, 4, axis=-1)

        # --- position ---

        # We predict position a bit differently from scale. For scale we want to put a prior on the actual value of
        # the scale, whereas for position we want to put a prior on the difference in position over timesteps.

        d_yt_logit = Normal(loc=d_yt_mean, scale=d_yt_std).sample()
        d_xt_logit = Normal(loc=d_xt_mean, scale=d_xt_std).sample()

        d_yt = self.where_t_scale * tf.nn.tanh(d_yt_logit)
        d_xt = self.where_t_scale * tf.nn.tanh(d_xt_logit)

        new_cyt = cyt + d_yt
        new_cxt = cxt + d_xt

        new_abs_posn = objects.abs_posn + tf.concat([d_yt, d_xt], axis=-1)

        # --- scale ---

        new_ys_mean = objects.ys_logit + d_ys
        new_xs_mean = objects.xs_logit + d_xs

        new_ys_logit = Normal(loc=new_ys_mean, scale=ys_std).sample()
        new_xs_logit = Normal(loc=new_xs_mean, scale=xs_std).sample()

        new_ys = float(self.max_hw - self.min_hw) * tf.nn.sigmoid(tf.clip_by_value(new_ys_logit, -10, 10)) + self.min_hw
        new_xs = float(self.max_hw - self.min_hw) * tf.nn.sigmoid(tf.clip_by_value(new_xs_logit, -10, 10)) + self.min_hw

        if self.use_abs_posn:
            box_params = tf.concat([new_abs_posn, d_yt_logit, d_xt_logit, new_ys_logit, new_xs_logit], axis=-1)
        else:
            box_params = tf.concat([d_yt_logit, d_xt_logit, new_ys_logit, new_xs_logit], axis=-1)

        new_objects.update(
            abs_posn=new_abs_posn,

            yt=new_cyt,
            xt=new_cxt,
            ys=new_ys,
            xs=new_xs,
            normalized_box=tf.concat([new_cyt, new_cxt, new_ys, new_xs], axis=-1),

            d_yt_logit=d_yt_logit,
            d_xt_logit=d_xt_logit,
            ys_logit=new_ys_logit,
            xs_logit=new_xs_logit,

            d_yt_logit_mean=d_yt_mean,
            d_xt_logit_mean=d_xt_mean,
            ys_logit_mean=new_ys_mean,
            xs_logit_mean=new_xs_mean,

            d_yt_logit_std=d_yt_std,
            d_xt_logit_std=d_xt_std,
            ys_logit_std=ys_std,
            xs_logit_std=xs_std,
        )

        # --- attributes ---

        # --- extract a glimpse using new box ---

        if is_posterior:
            _, image_height, image_width, _ = tf_shape(inp)
            _new_cyt, _new_cxt, _new_ys, _new_xs = coords_to_image_space(
                new_cyt, new_cxt, new_ys, new_xs, (image_height, image_width), self.anchor_box, top_left=False)

            glimpse = extract_affine_glimpse(
                inp, self.object_shape, _new_cyt, _new_cxt, _new_ys, _new_xs, self.edge_resampler)

        else:
            glimpse = tf.zeros((batch_size, n_objects, *self.object_shape, self.image_depth))

        glimpse_mask = tf.nn.sigmoid(glimpse_mask_logit + 1.)
        leading_mask_shape = tf_shape(glimpse)[:-1]
        glimpse_mask = tf.reshape(glimpse_mask, (*leading_mask_shape, 1))

        glimpse *= glimpse_mask

        encoded_glimpse = apply_object_wise(
            self.glimpse_encoder, glimpse, n_trailing_dims=3, output_size=self.A, is_training=self.is_training)

        if not is_posterior:
            encoded_glimpse = tf.zeros((batch_size, n_objects, self.A), dtype=tf.float32)

        # --- predict change in attributes ---

        # so under sqair we mix between three different values for the attributes:
        # 1. value from previous timestep
        # 2. value predicted directly from glimpse
        # 3. value predicted based on update of temporal cell...this update conditions on hidden_output,
        # the prediction in #2., and the where values.

        # How to do this given that we are predicting the change in attr? We could just directly predict
        # the attr instead, but call it d_attr. After all, it is in this function that we control
        # whether d_attr is added to attr.

        # So, make a prediction based on just the input:

        attr_from_inp = apply_object_wise(
            self.predict_attr_inp, encoded_glimpse, output_size=2*self.A, is_training=self.is_training)
        attr_from_inp_mean, attr_from_inp_log_std = tf.split(attr_from_inp, [self.A, self.A], axis=-1)

        attr_from_inp_std = self.std_nonlinearity(attr_from_inp_log_std)

        # And then a prediction which takes the past into account (predicting gate values at the same time):

        attr_from_temp_inp = tf.concat([base_features, box_params, encoded_glimpse], axis=-1)
        attr_from_temp = apply_object_wise(
            self.predict_attr_temp, attr_from_temp_inp, output_size=5*self.A, is_training=self.is_training)

        (attr_from_temp_mean, attr_from_temp_log_std,
         f_gate_logit, i_gate_logit, t_gate_logit) = tf.split(attr_from_temp, 5, axis=-1)

        attr_from_temp_std = self.std_nonlinearity(attr_from_temp_log_std)

        # bias the gates
        f_gate = tf.nn.sigmoid(f_gate_logit + 1) * .9999
        i_gate = tf.nn.sigmoid(i_gate_logit + 1) * .9999
        t_gate = tf.nn.sigmoid(t_gate_logit + 1) * .9999

        attr_mean = f_gate * objects.attr + (1 - i_gate) * attr_from_inp_mean + (1 - t_gate) * attr_from_temp_mean
        attr_std = (1 - i_gate) * attr_from_inp_std + (1 - t_gate) * attr_from_temp_std

        new_attr = Normal(loc=attr_mean, scale=attr_std).sample()

        # --- apply change in attributes ---

        new_objects.update(
            attr=new_attr,
            d_attr=new_attr - objects.attr,
            d_attr_mean=attr_mean - objects.attr,
            d_attr_std=attr_std,
            f_gate=f_gate,
            i_gate=i_gate,
            t_gate=t_gate,
            glimpse=glimpse,
            glimpse_mask=glimpse_mask,
        )

        # --- z ---

        d_z_inp = tf.concat([base_features, box_params, new_attr, encoded_glimpse], axis=-1)
        d_z_params = apply_object_wise(self.d_z_network, d_z_inp, output_size=2, is_training=self.is_training)

        d_z_mean, d_z_log_std = tf.split(d_z_params, 2, axis=-1)
        d_z_std = self.std_nonlinearity(d_z_log_std)

        d_z_mean = self.training_wheels * tf.stop_gradient(d_z_mean) + (1-self.training_wheels) * d_z_mean
        d_z_std = self.training_wheels * tf.stop_gradient(d_z_std) + (1-self.training_wheels) * d_z_std

        d_z_logit = Normal(loc=d_z_mean, scale=d_z_std).sample()

        new_z_logit = objects.z_logit + d_z_logit
        new_z = self.z_nonlinearity(new_z_logit)

        new_objects.update(
            z=new_z,
            z_logit=new_z_logit,
            d_z_logit=d_z_logit,
            d_z_logit_mean=d_z_mean,
            d_z_logit_std=d_z_std,
        )

        # --- obj ---

        d_obj_inp = tf.concat([base_features, box_params, new_attr, new_z, encoded_glimpse], axis=-1)
        d_obj_logit = apply_object_wise(self.d_obj_network, d_obj_inp, output_size=1, is_training=self.is_training)

        d_obj_logit = self.training_wheels * tf.stop_gradient(d_obj_logit) + (1-self.training_wheels) * d_obj_logit
        d_obj_log_odds = tf.clip_by_value(d_obj_logit / self.obj_temp, -10., 10.)

        d_obj_pre_sigmoid = (
            self._noisy * concrete_binary_pre_sigmoid_sample(d_obj_log_odds, self.obj_concrete_temp)
            + (1 - self._noisy) * d_obj_log_odds
        )

        d_obj = tf.nn.sigmoid(d_obj_pre_sigmoid)

        new_obj = objects.obj * d_obj
        new_render_obj = new_obj

        new_objects.update(
            d_obj_log_odds=d_obj_log_odds,
            d_obj_prob=tf.nn.sigmoid(d_obj_log_odds),
            d_obj_pre_sigmoid=d_obj_pre_sigmoid,
            d_obj=d_obj,
            obj=new_obj,
            render_obj=new_render_obj,
        )

        # --- update each object's hidden state --

        cell_input = tf.concat([box_params, new_attr, new_z, new_obj], axis=-1)

        if is_posterior:
            _, new_objects.prop_state = apply_object_wise(self.cell, cell_input, objects.prop_state)
            new_objects.prior_prop_state = new_objects.prop_state
        else:
            _, new_objects.prior_prop_state = apply_object_wise(self.cell, cell_input, objects.prior_prop_state)
            new_objects.prop_state = new_objects.prior_prop_state

        return new_objects
