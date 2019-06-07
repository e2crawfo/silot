import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from itertools import product
from orderedattrdict import AttrDict
import pprint
import shutil
import os
import sonnet as snt

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.colors import to_rgb

from dps import cfg
from dps.utils import Param, map_structure, Config
from dps.utils.tf import RenderHook, tf_mean_sum, tf_shape, MLP

from auto_yolo.models.core import AP, xent_loss, coords_to_pixel_space
from auto_yolo.models.object_layer import GridObjectLayer, ConvGridObjectLayer, ObjectRenderer
from auto_yolo.models.networks import SpatialAttentionLayerV2, DummySpatialAttentionLayer

from spair_video.core import VideoNetwork, MOTMetrics
from spair_video.propagation import ObjectPropagationLayer, SQAIRPropagationLayer


def select_top_k_objects(prop, disc):
    batch_size, *prop_other, final = tf_shape(prop.obj)
    assert final == 1
    n_prop_objects = np.product(prop_other)

    _, *disc_other, _ = tf_shape(disc.obj)
    n_disc_objects = np.product(disc_other)
    n_disc_obj_dim = len(disc_other)

    prop_presence = tf.reshape(prop.obj, (batch_size, n_prop_objects))
    disc_presence = tf.reshape(disc.obj, (batch_size, n_disc_objects))

    all_presence = tf.concat([prop_presence, disc_presence], axis=1)

    _, top_k_indices = tf.nn.top_k(all_presence, k=n_prop_objects, sorted=False)
    top_k_indices = tf.sort(top_k_indices, axis=1)
    top_k_indices = tf.reshape(top_k_indices, (batch_size, n_prop_objects))

    from_prop = tf.cast(top_k_indices < n_prop_objects, tf.int32)
    n_from_prop = tf.reduce_sum(from_prop, axis=1)

    scatter_indices = tf.concat(
        [tf.tile(tf.range(batch_size)[:, None, None], (1, n_prop_objects, 1)),
         top_k_indices[:, :, None]],
        axis=2
    )

    # create an array of shape (batch_size, n_prop_objects+n_disc_objects) that
    # has a 1 for every index that is in the top_k for that batch element
    in_top_k = tf.scatter_nd(
        scatter_indices, tf.ones((batch_size, n_prop_objects), dtype=tf.int32),
        (batch_size, n_prop_objects+n_disc_objects))

    from_disc_idx = n_from_prop

    new_indices = []
    is_new = []
    for i in range(n_prop_objects):
        # indices to use for gather if i is not present in top_k
        gather_indices = tf.concat([tf.range(batch_size)[:, None], from_disc_idx[:, None]], axis=1)
        other = tf.gather_nd(top_k_indices, gather_indices)

        i_present = in_top_k[:, i]

        indices = tf.where(tf.cast(i_present, tf.bool), i * tf.ones_like(other), other)

        from_disc_idx += 1 - i_present

        new_indices.append(indices)
        is_new.append(1 - i_present)

    top_k_indices = tf.stack(new_indices, axis=1)
    is_new = tf.stack(is_new, axis=1)

    batch_indices = tf.tile(tf.range(batch_size)[:, None, None], (1, n_prop_objects, 1))
    index_array = tf.concat([batch_indices, top_k_indices[:, :, None]], axis=2)

    selected_objects = AttrDict()

    shared_keys = disc.keys() & prop.keys()
    for key in shared_keys:
        trailing_dims = tf_shape(disc[key])[1+n_disc_obj_dim:]
        disc_value = tf.reshape(disc[key], (batch_size, n_disc_objects, *trailing_dims))
        values = tf.concat([prop[key], disc_value], axis=1)
        selected_objects[key] = tf.gather_nd(values, index_array)

    selected_objects.update(
        pred_n_objects=tf.reduce_sum(selected_objects.obj, axis=(1, 2)),
        pred_n_objects_hard=tf.reduce_sum(tf.round(selected_objects.render_obj), axis=(1, 2)),
        final_weights=tf.one_hot(top_k_indices, n_prop_objects + n_disc_objects, axis=-1),
        is_new=is_new,
    )

    return selected_objects


def apply_keys(d, values):
    new = type(d)()
    for (k, v), _v in zip(sorted(d.items()), values):
        if isinstance(_v, tf.Tensor):
            new[k] = _v
        else:
            new[k] = apply_keys(v, _v)
    return new


def append_to_tensor_arrays(f, structured, tensor_arrays):
    new_tensor_arrays = []
    for (k, v), ta in zip(sorted(structured.items()), tensor_arrays):
        if isinstance(v, tf.Tensor):
            result = ta.write(f, v)
        else:
            result = append_to_tensor_arrays(f, v, ta)
        new_tensor_arrays.append(result)
    return new_tensor_arrays


def make_tensor_arrays(structure, n_frames):
    tas = []
    for k, v in sorted(structure.items()):
        if isinstance(v, tf.Tensor):
            ta = tf.TensorArray(v.dtype, n_frames, dynamic_size=False, element_shape=v.shape)
            tas.append(ta)
        else:
            _tas = make_tensor_arrays(v, n_frames)
            tas.append(_tas)
    return tas


class InterpretableSequentialSpair(VideoNetwork):
    build_backbone = Param()
    build_discovery_feature_fuser = Param()
    build_mlp = Param()
    build_prop_cell = Param()

    n_backbone_features = Param()
    n_objects_per_cell = Param()

    train_reconstruction = Param()
    reconstruction_weight = Param()
    train_kl = Param()
    kl_weight = Param()

    prior_start_step = Param()
    n_hidden = Param()
    learn_prior = Param()
    disc_dropout_prob = Param()
    anchor_box = Param()
    independent_prop = Param()
    use_sqair_prop = Param()
    conv_discovery = Param()

    disc_layer = None
    disc_feature_extractor = None
    prop_layer = None
    prop_feature_extractor = None
    prop_cell = None

    object_renderer = None

    @property
    def eval_funcs(self):
        if getattr(self, '_eval_funcs', None) is None:
            if "annotations" in self._tensors:
                ap_iou_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                eval_funcs = {"AP_at_point_{}".format(int(10 * v)): AP(v) for v in ap_iou_values}
                eval_funcs["AP"] = AP(ap_iou_values)
                eval_funcs["MOT"] = MOTMetrics()
                self._eval_funcs = eval_funcs
            else:
                self._eval_funcs = {}

        return self._eval_funcs

    def _loop_cond(self, f, *_):
        return f < self.dynamic_n_frames

    def _loop_body(self, f, normalized_box, attr, z, obj, _all, prop_state, ys_logit, xs_logit, z_logit, *tensor_arrays):
        objects = AttrDict(
            normalized_box=normalized_box,
            ys_logit=ys_logit,
            xs_logit=xs_logit,
            attr=attr,
            z=z,
            z_logit=z_logit,
            obj=obj,
            all=_all,
            prop_state=prop_state,
        )

        structured_result = self._inner_loop_body(f, objects)
        tensor_arrays = append_to_tensor_arrays(f, structured_result, tensor_arrays)
        selected_objects = structured_result.selected_objects

        f += 1

        return [
            f,
            selected_objects.normalized_box,
            selected_objects.attr,
            selected_objects.z,
            selected_objects.obj,
            selected_objects.all,
            selected_objects.prop_state,
            selected_objects.ys_logit,
            selected_objects.xs_logit,
            selected_objects.z_logit,
            *tensor_arrays]

    def _inner_loop_body(self, f, objects):

        # --- prop ---

        object_features = tf.concat([objects.all[..., 2:], objects.prop_state], axis=2)
        object_locs = objects.all[..., :2]

        object_features_for_prop = self.prop_feature_extractor(
            object_locs, object_features, object_locs, object_features, self.is_training)

        prop_args = (self.inp[:, f], object_features_for_prop, objects, self.is_training,)

        post_prop_objects = self.prop_layer(*prop_args, is_posterior=True)
        prop_objects = post_prop_objects

        # if self.learn_prior:
        #     prior_prop_objects = self.prop_layer(*prop_args, is_posterior=False)
        # else:
        #     prior_prop_objects = None

        # prop_objects = tf.cond(
        #     use_prior_objects,
        #     lambda: prior_prop_objects,
        #     lambda: post_prop_objects)

        # --- get features of the propagated objects for the purposes of discovery ---

        object_locs = prop_objects.all[..., :2]
        object_features = tf.concat([prop_objects.all[..., 2:], prop_objects.prop_state], axis=2)

        object_features_for_disc = self.disc_feature_extractor(
            object_locs, object_features, self.grid_cell_centers, None, self.is_training)

        object_features_for_disc = tf.reshape(
            object_features_for_disc, (self.batch_size, self.H, self.W, self.n_hidden))

        # if self.learn_prior:
        #     dummy_backbone_output = tf.zeros_like(self.backbone_output[:, f])

        #     is_prior_tf = tf.ones_like(prop_object_features[..., 0:2]) * [0, 1]
        #     prior_features_inp = tf.concat(
        #         [prop_object_features, dummy_backbone_output, is_prior_tf], axis=-1)

        #     prior_disc_features = self.disc_feature_fuser(
        #         prior_features_inp, self.B*self.n_backbone_features, self.is_training)

        #     prior_disc_features = tf.reshape(
        #         prior_disc_features, (self.batch_size, self.H, self.W, self.B, self.n_backbone_features))

        #     prior_disc_objects = self.disc_layer(
        #         self.inp[:, f], prior_disc_features, self.is_training, is_posterior=False)

        #     prior_selected_objects = select_top_k_objects(prior_prop_objects, prior_disc_objects)

        #     prior_render_tensors = self.object_renderer(
        #         prior_selected_objects, self._tensors["background"][:, f], self.is_training)

        # else:
        #     prior_disc_objects = None

        # --- fuse features of the propagated objects with bottom-up features from the current frame ---

        is_posterior_tf = tf.ones_like(object_features_for_disc[..., 0:2]) * [1, 0]
        post_disc_features_inp = tf.concat(
            [object_features_for_disc, self.backbone_output[:, f], is_posterior_tf], axis=-1)

        post_disc_features = self.discovery_feature_fuser(
            post_disc_features_inp, self.B*self.n_backbone_features, self.is_training)

        post_disc_features = tf.reshape(
            post_disc_features, (self.batch_size, self.H, self.W, self.B, self.n_backbone_features))

        # --- discovery ---

        post_disc_objects = self.disc_layer(
            self.inp[:, f], post_disc_features, self.is_training,
            is_posterior=True, prop_state=self.initial_prop_state)

        # --- discovery dropout ---

        disc_mask_dist = tfp.distributions.Bernoulli(
            (1.-self.disc_dropout_prob) * tf.ones(self.batch_size))
        disc_mask = tf.cast(disc_mask_dist.sample(), tf.float32)
        do_mask = self.float_is_training * tf.cast(f > 0, tf.float32)
        disc_mask = do_mask * disc_mask + (1 - do_mask) * tf.ones(self.batch_size)
        disc_mask = disc_mask[:, None, None]

        post_disc_objects.obj = disc_mask * post_disc_objects.obj
        post_disc_objects.render_obj = disc_mask * post_disc_objects.render_obj

        # --- object selection ---

        post_selected_objects = select_top_k_objects(post_prop_objects, post_disc_objects)

        # --- rendering ---

        post_render_tensors = self.object_renderer(
            post_selected_objects, self._tensors["background"][:, f], self.is_training)

        # ---

        render_tensors = post_render_tensors
        selected_objects = post_selected_objects

        # render_tensors = tf.cond(
        #     use_prior_objects,
        #     lambda: prior_render_tensors,
        #     lambda: post_render_tensors)

        # selected_objects = tf.cond(
        #     use_prior_objects,
        #     lambda: prior_selected_objects,
        #     lambda: post_selected_objects)

        # --- appearance of object sets for plotting ---

        post_prop_objects.update(
            self.object_renderer(post_prop_objects, None, self.is_training, appearance_only=True))
        post_disc_objects.update(
            self.object_renderer(post_disc_objects, None, self.is_training, appearance_only=True))

        # --- kl ---

        prop_indep_prior_kl = self.prop_layer.compute_kl(post_prop_objects, do_obj=False)
        disc_indep_prior_kl = self.disc_layer.compute_kl(post_disc_objects, do_obj=False)

        obj_for_kl = AttrDict()
        for name in "obj_pre_sigmoid obj_log_odds obj_prob".split():
            obj_for_kl[name] = tf.concat(
                [post_prop_objects["d_" + name], post_disc_objects[name]], axis=1)
        obj_for_kl['obj'] = tf.concat([post_prop_objects['obj'], post_disc_objects['obj']], axis=1)

        indep_prior_obj_kl = self.disc_layer._compute_obj_kl(obj_for_kl)

        _tensors = AttrDict(
            post=AttrDict(
                prop=post_prop_objects,
                disc=post_disc_objects,
                select=post_selected_objects,
                render=post_render_tensors,
            ),

            selected_objects=selected_objects,

            prop_indep_prior_kl=prop_indep_prior_kl,
            disc_indep_prior_kl=disc_indep_prior_kl,

            indep_prior_obj_kl=indep_prior_obj_kl,

            **render_tensors,
        )

        # --- prior ---

        # if self.learn_prior:
        #     prior_prop_objects.update(
        #         self.object_renderer(prior_prop_objects, None, self.is_training, appearance_only=True))
        #     prior_disc_objects.update(
        #         self.object_renderer(prior_disc_objects, None, self.is_training, appearance_only=True))

        #     prop_learned_prior_kl = self.prop_layer.compute_kl(
        #         post_prop_objects, prior=prior_prop_objects)
        #     disc_learned_prior_kl = self.disc_layer.compute_kl(
        #         post_disc_objects, prior=prior_disc_objects)

        #     _tensors.update(
        #         prior=AttrDict(
        #             prop=prior_prop_objects,
        #             disc=prior_disc_objects,
        #             select=prior_selected_objects,
        #             render=prior_render_tensors,
        #         ),
        #         disc_learned_prior_kl=disc_learned_prior_kl,
        #         prop_learned_prior_kl=prop_learned_prior_kl,
        #     )

        return _tensors

    def build_representation(self):
        # --- init modules ---

        self.maybe_build_subnet("backbone")
        self.maybe_build_subnet("discovery_feature_fuser")

        self.B = self.n_objects_per_cell

        self.backbone_output, n_grid_cells, grid_cell_size = self.backbone(
            self.inp, self.B*self.n_backbone_features, self.is_training)

        self.H, self.W = [int(i) for i in n_grid_cells]
        self.HWB = self.H * self.W * self.B
        self.pixels_per_cell = tuple(int(i) for i in grid_cell_size)
        H, W = self.H, self.W

        if self.disc_layer is None:
            if self.conv_discovery:
                self.disc_layer = ConvGridObjectLayer(self.pixels_per_cell, scope="discovery")
            else:
                self.disc_layer = GridObjectLayer(self.pixels_per_cell, scope="discovery")

        if self.prop_cell is None:
            self.prop_cell = self.build_prop_cell(2*self.n_hidden, name="prop_cell")
            # self.prop_cell must be a Sonnet RNNCore
            self.initial_prop_state = snt.trainable_initial_state(
                1, self.prop_cell.state_size, tf.float32, name="prop_cell_initial_state")

        if self.prop_layer is None:
            prop_class = SQAIRPropagationLayer if self.use_sqair_prop else ObjectPropagationLayer
            self.prop_layer = prop_class(cell=self.prop_cell, scope="propagation")

        if self.object_renderer is None:
            self.object_renderer = ObjectRenderer(scope="renderer")

        if self.disc_feature_extractor is None:
            self.disc_feature_extractor = SpatialAttentionLayerV2(
                n_hidden=self.n_hidden,
                build_mlp=lambda scope: MLP(n_units=[self.n_hidden, self.n_hidden], scope=scope),
                do_object_wise=False,
                scope="discovery_feature_extractor",
            )

        if self.prop_feature_extractor is None:
            if self.independent_prop:
                self.prop_feature_extractor = DummySpatialAttentionLayer(
                    n_hidden=self.n_hidden,
                    build_mlp=lambda scope: MLP(n_units=[self.n_hidden, self.n_hidden], scope=scope),
                    do_object_wise=True,
                    scope="propagation_feature_extractor",
                )
            else:
                self.prop_feature_extractor = SpatialAttentionLayerV2(
                    n_hidden=self.n_hidden,
                    build_mlp=lambda scope: MLP(n_units=[self.n_hidden, self.n_hidden], scope=scope),
                    do_object_wise=True,
                    scope="propagation_feature_extractor",
                )

        # centers of the grid cells in normalized (anchor box) space.

        y = (np.arange(H, dtype='f') + 0.5) / H * (self.image_height / self.anchor_box[0])
        x = (np.arange(W, dtype='f') + 0.5) / W * (self.image_width / self.anchor_box[1])
        x, y = np.meshgrid(x, y)
        self.grid_cell_centers = tf.constant(np.concatenate([y.flatten()[:, None], x.flatten()[:, None]], axis=1))

        tensors = []
        objects = self.prop_layer.null_object_set(self.batch_size)

        f = tf.constant(0, dtype=tf.int32)
        structure = self._inner_loop_body(f, objects)

        tensor_arrays = make_tensor_arrays(structure, self.dynamic_n_frames)

        loop_vars = [
            f, objects.normalized_box, objects.attr, objects.z, objects.obj, objects.all,
            objects.prop_state, objects.ys_logit, objects.xs_logit, objects.z_logit, *tensor_arrays]

        result = tf.while_loop(self._loop_cond, self._loop_body, loop_vars)

        first_ta = min(i for i, ta in enumerate(result) if isinstance(ta, tf.TensorArray))
        tensor_arrays = result[first_ta:]

        tensors = map_structure(lambda ta: ta.stack(), tensor_arrays, is_leaf=lambda t: isinstance(t, tf.TensorArray))
        tensors = map_structure(
            lambda t: tf.transpose(t, (1, 0, *range(2, len(t.shape)))),
            tensors, is_leaf=lambda t: isinstance(t, tf.Tensor))
        tensors = apply_keys(structure, tensors)

        self._tensors.update(tensors)
        self._tensors.update(**self._tensors['selected_objects'])

        pprint.pprint(self._tensors)

        # --- specify values to record ---

        self.record_tensors(
            batch_size=self.batch_size,
            float_is_training=self.float_is_training,
        )

        prop_to_record = (
            "yt xt ys xs z attr obj d_yt_logit d_xt_logit ys_logit xs_logit d_z_logit d_attr d_obj".split())

        post_prop = self._tensors.post.prop
        self.record_tensors(**{"post_prop_{}".format(k): post_prop[k] for k in prop_to_record})
        self.record_tensors(
            **{"post_prop_{}".format(k): v for k, v in post_prop.items() if k.endswith('_std')})

        if "d_attr_gate" in post_prop:
            self.record_tensors(post_prop_d_attr_gate=post_prop["d_attr_gate"])

        if "f_gate" in post_prop:
            self.record_tensors(post_prop_d_attr_f_gate=post_prop["f_gate"])
            self.record_tensors(post_prop_d_attr_i_gate=post_prop["i_gate"])
            self.record_tensors(post_prop_d_attr_t_gate=post_prop["t_gate"])

        disc_to_record = "cell_y cell_x height width yt xt ys xs z attr obj pred_n_objects".split()

        post_disc = self._tensors.post.disc
        self.record_tensors(**{"post_disc_{}".format(k): post_disc[k] for k in disc_to_record})
        self.record_tensors(
            **{"post_disc_{}".format(k): v for k, v in post_disc.items() if k.endswith('_std')})

        if self.learn_prior:
            prior_prop = self._tensors.prior.prop
            self.record_tensors(**{"prior_prop_{}".format(k): prior_prop[k] for k in prop_to_record})
            self.record_tensors(
                **{"prior_prop_{}".format(k): v for k, v in prior_prop.items() if k.endswith('_std')})

            prior_disc = self._tensors.prior.disc
            self.record_tensors(**{"prior_disc_{}".format(k): prior_disc[k] for k in disc_to_record})
            self.record_tensors(
                **{"prior_disc_{}".format(k): v for k, v in prior_disc.items() if k.endswith('_std')})

        # --- losses ---

        if self.train_reconstruction:
            output = self._tensors['output']
            inp = self._tensors['inp']
            self._tensors['per_pixel_reconstruction_loss'] = xent_loss(pred=output, label=inp)
            self.losses['reconstruction'] = (
                self.reconstruction_weight * tf_mean_sum(self._tensors['per_pixel_reconstruction_loss'])
            )

        if self.train_kl:
            kl_weight = 0.5 * self.kl_weight if self.learn_prior else self.kl_weight

            prop_obj = self._tensors.post.prop.obj
            prop_indep_prior_kl = self._tensors["prop_indep_prior_kl"]

            self.losses.update(
                **{"prop_indep_prior_{}".format(k): kl_weight * tf_mean_sum(prop_obj * kl)
                   for k, kl in prop_indep_prior_kl.items()
                   if "obj" not in k}
            )

            disc_obj = self._tensors.post.disc.obj
            disc_indep_prior_kl = self._tensors["disc_indep_prior_kl"]

            self.losses.update(
                **{"disc_indep_prior_{}".format(k): kl_weight * tf_mean_sum(disc_obj * kl)
                   for k, kl in disc_indep_prior_kl.items()
                   if "obj" not in k}
            )

            self.losses.update(
                indep_prior_obj_kl=kl_weight * tf_mean_sum(self._tensors["indep_prior_obj_kl"]),
            )

            if self.learn_prior:
                prop_learned_prior_kl = self._tensors["prop_learned_prior_kl"]
                self.losses.update(
                    **{"prop_learned_prior_{}".format(k): kl_weight * tf_mean_sum(prop_obj * kl)
                       for k, kl in prop_learned_prior_kl.items()
                       if "obj" not in k}
                )

                disc_learned_prior_kl = self._tensors["disc_learned_prior_kl"]
                self.losses.update(
                    **{"disc_learned_prior_{}".format(k): kl_weight * tf_mean_sum(disc_obj * kl)
                       for k, kl in disc_learned_prior_kl.items()
                       if "obj" not in k}
                )

                self.losses.update(
                    learned_prior_obj_kl=kl_weight * tf_mean_sum(self._tensors["learned_prior_obj_kl"]),
                )

            # Don't multiply by 0.5 here, because there is no learned prior
            if cfg.background_cfg.mode in ("learn_and_transform", "learn"):
                self.losses.update(
                    bg_attr_kl=self.kl_weight * tf_mean_sum(self._tensors["bg_attr_kl"]),
                )
            if cfg.background_cfg.mode == "learn_and_transform":
                self.losses.update(
                    bg_transform_kl=self.kl_weight * tf_mean_sum(self._tensors["bg_transform_kl"]),
                )

        # --- other evaluation metrics ---

        if "n_annotations" in self._tensors:
            count_1norm = tf.to_float(
                tf.abs(
                    tf.to_int32(self._tensors["pred_n_objects_hard"])
                    - self._tensors["n_valid_annotations"]))

            count_1norm_relative = (
                count_1norm / tf.maximum(tf.cast(self._tensors["n_valid_annotations"], tf.float32), 1e-6))

            self.record_tensors(
                count_1norm_relative=count_1norm_relative,
                count_1norm=count_1norm,
                count_error=count_1norm > 0.5,
            )


class ISSPAIR_RenderHook(RenderHook):
    N = 4
    linewidth = 2
    on_color = np.array(to_rgb("xkcd:azure"))
    off_color = np.array(to_rgb("xkcd:red"))
    selected_color = np.array(to_rgb("xkcd:neon green"))
    unselected_color = np.array(to_rgb("xkcd:fire engine red"))
    gt_color = "xkcd:yellow"
    glimpse_color = "xkcd:orange"
    cutoff = 0.5

    def build_fetches(self, updater):
        prop_names = (
            "d_obj xs_logit d_xt_logit ys_logit d_yt_logit d_z_logit xs xt ys yt "
            "glimpse normalized_box obj glimpse_prime z appearance glimpse_prime_box"
        ).split()

        if updater.network.use_sqair_prop:
            prop_names.extend(['glimpse_prime_mask', 'glimpse_mask'])

        disc_names = "obj render_obj z appearance normalized_box glimpse".split()
        select_names = "obj z normalized_box final_weights yt xt ys xs".split()
        render_names = "output".split()

        _fetches = Config(
            post=Config(
                disc=Config(**{n: 0 for n in disc_names}),
                prop=Config(**{n: 0 for n in prop_names}),
                select=Config(**{n: 0 for n in select_names}),
                render=Config(**{n: 0 for n in render_names}),
            ),
        )

        if updater.network.learn_prior:
            _fetches["prior"] = Config(
                disc=Config(**{n: 0 for n in disc_names}),
                prop=Config(**{n: 0 for n in prop_names}),
                select=Config(**{n: 0 for n in select_names}),
                render=Config(**{n: 0 for n in render_names}),
            )

        fetches = ' '.join(list(_fetches.keys()))
        fetches += " inp background"

        network = updater.network
        if "n_annotations" in network._tensors:
            fetches += " annotations n_annotations"

        if 'prediction' in network._tensors:
            fetches += " prediction targets"

        if "actions" in network._tensors:
            fetches += " actions"

        if "bg_y" in network._tensors:
            fetches += " bg_y bg_x bg_h bg_w bg_raw"

        return fetches

    def __call__(self, updater):
        fetched = self._fetch(updater)
        fetched = Config(fetched)
        self._prepare_fetched(updater, fetched)
        self._plot_patches(updater, fetched)

    @staticmethod
    def normalize_images(images):
        mx = images.reshape(*images.shape[:-3], -1).max(axis=-1)
        return images / mx[..., None, None, None]

    def _prepare_fetched(self, updater, fetched):
        inp = fetched['inp']
        prediction = fetched.get("prediction", None)
        targets = fetched.get("targets", None)

        N, T, image_height, image_width, _ = inp.shape

        background = fetched['background']

        modes = "prior post" if updater.network.learn_prior else "post"

        for mode in modes.split():
            for kind in "disc prop select".split():
                yt, xt, ys, xs = np.split(fetched[mode][kind].normalized_box, 4, axis=-1)
                pixel_space_box = coords_to_pixel_space(
                    yt, xt, ys, xs, (image_height, image_width), updater.network.anchor_box, top_left=True)
                fetched[mode][kind].pixel_space_box = np.concatenate(pixel_space_box, axis=-1)

            g_yt, g_xt, g_ys, g_xs = np.split(fetched[mode]["prop"].glimpse_prime_box, 4, axis=-1)
            glimpse_prime_pixel_space_box = coords_to_pixel_space(
                g_yt, g_xt, g_ys, g_xs, (image_height, image_width), updater.network.anchor_box, top_left=True)
            fetched[mode]["prop"].glimpse_prime_pixel_space_box = np.concatenate(glimpse_prime_pixel_space_box, axis=-1)

            output = fetched[mode].render.output
            fetched[mode].render.diff = self.normalize_images(np.abs(inp - output).mean(axis=-1, keepdims=True))
            fetched[mode].render.xent = self.normalize_images(
                xent_loss(pred=output, label=inp, tf=False).mean(axis=-1, keepdims=True))

        n_annotations = fetched.get("n_annotations", np.zeros(N, dtype='i'))
        annotations = fetched.get("annotations", None)
        # actions = fetched.get("actions", None)

        learned_bg = "bg_y" in fetched
        bg_y = fetched.get("bg_y", None)
        bg_x = fetched.get("bg_x", None)
        bg_h = fetched.get("bg_h", None)
        bg_w = fetched.get("bg_w", None)
        bg_raw = fetched.get("bg_raw", None)

        fetched.update(
            prediction=prediction,
            targets=targets,
            background=background,
            n_annotations=n_annotations,
            annotations=annotations,
            learned_bg=learned_bg,
            bg_y=bg_y,
            bg_x=bg_x,
            bg_h=bg_h,
            bg_w=bg_w,
            bg_raw=bg_raw,
        )

    def _plot_patches(self, updater, fetched):
        # Create a plot showing what each object is generating

        def flt(main=None, **floats):
            if main is not None:
                s = main + ": "
            else:
                s = ''
            s += ', '.join("{}={:.2f}".format(k, v) for k, v in floats.items())
            return s

        N, T, image_height, image_width, _ = fetched['inp'].shape
        H, W, B = updater.network.H, updater.network.W, updater.network.B

        fig_unit_size = 3
        n_other_plots = 10

        # number of objects per image
        M = 5
        if updater.network.use_sqair_prop:
            M += 2  # for masks

        fig_half_width = max(M*W, n_other_plots)
        fig_width = 2 * fig_half_width + 1 if updater.network.learn_prior else fig_half_width
        n_prop_objects = updater.network.prop_layer.n_prop_objects
        n_prop_rows = int(np.ceil(n_prop_objects / W))
        fig_height = B * H + 4 + 2*n_prop_rows + 2

        for idx in range(N):

            # --- set up figure and axes ---

            fig = plt.figure(figsize=(fig_unit_size*fig_width, fig_unit_size*fig_height))
            time_text = fig.suptitle('', fontsize=20, fontweight='bold')

            gs = gridspec.GridSpec(fig_height, fig_width, figure=fig)

            post_disc_axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(M*W)] for i in range(B*H)])
            post_prop_axes = np.array([
                [fig.add_subplot(gs[B*H+4+i, j]) for j in range(M*W)]
                for i in range(n_prop_rows)])
            post_prop_axes = post_prop_axes.flatten()
            post_select_axes = np.array([
                [fig.add_subplot(gs[B*H+4+n_prop_rows+i, j]) for j in range(M*W)]
                for i in range(n_prop_rows)])
            post_select_axes = post_select_axes.flatten()

            post_other_axes = []
            for i in range(2):
                for j in range(int(fig_half_width/2)):
                    start_y = B*H + 2*i
                    end_y = start_y + 2
                    start_x = 2*j
                    end_x = start_x + 2
                    ax = fig.add_subplot(gs[start_y:end_y, start_x:end_x])
                    post_other_axes.append(ax)

            post_other_axes = np.array(post_other_axes)

            post_axes = np.concatenate(
                [post_disc_axes.flatten(), post_prop_axes.flatten(),
                 post_select_axes.flatten(), post_other_axes.flatten()],
                axis=0)

            axes_sets = [
                ('post', post_disc_axes, post_prop_axes, post_select_axes, post_other_axes)
            ]

            if updater.network.learn_prior:
                prior_disc_axes = np.array([
                    [fig.add_subplot(gs[i, fig_half_width + 1 + j])for j in range(M*W)]
                    for i in range(B*H)])
                prior_prop_axes = np.array([
                    [fig.add_subplot(gs[B*H+4+i, fig_half_width + 1 + j]) for j in range(M*W)]
                    for i in range(n_prop_rows)])
                prior_prop_axes = prior_prop_axes.flatten()
                prior_select_axes = np.array([
                    [fig.add_subplot(gs[B*H+4+n_prop_rows+i, fig_half_width + 1 + j]) for j in range(M*W)]
                    for i in range(n_prop_rows)])
                prior_select_axes = prior_select_axes.flatten()

                prior_other_axes = []
                for i in range(2):
                    for j in range(int(fig_half_width/2)):
                        start_y = B*H + 2*i
                        end_y = start_y + 2
                        start_x = fig_half_width + 1 + 2*j
                        end_x = start_x + 2
                        ax = fig.add_subplot(gs[start_y:end_y, start_x:end_x])
                        prior_other_axes.append(ax)

                prior_other_axes = np.array(prior_other_axes)

                prior_axes = np.concatenate(
                    [prior_disc_axes.flatten(), prior_prop_axes.flatten(),
                     prior_select_axes.flatten(), prior_other_axes.flatten()],
                    axis=0)

                axes_sets.append(('prior', prior_disc_axes, prior_prop_axes, prior_select_axes, prior_other_axes))
            else:
                prior_axes = np.zeros_like(post_axes[:0])

            bottom_axes = np.array([fig.add_subplot(gs[-2:, 2*i:2*(i+1)]) for i in range(int(fig_width/2))])

            all_axes = np.concatenate([post_axes, prior_axes, bottom_axes], axis=0)

            for ax in all_axes.flatten():
                ax.set_axis_off()

            # --- plot data ---

            lw = self.linewidth

            print("Plotting patches for {}...".format(idx))

            def func(t):
                print("timestep {}".format(t))
                if updater.network.learn_prior:
                    time_text.set_text('post{}t = {}{}prior'.format(' '*40, t, ' '*40))
                else:
                    time_text.set_text('t={}'.format(t))

                ax_inp = bottom_axes[0]
                self.imshow(ax_inp, fetched.inp[idx, t])
                if t == 0:
                    ax_inp.set_title('input')

                ax = bottom_axes[1]
                self.imshow(ax, fetched.background[idx, t])
                if t == 0:
                    ax.set_title('background')

                if fetched.learned_bg:
                    ax = bottom_axes[2]

                    bg_y, bg_x, bg_h, bg_w = fetched.bg_y, fetched.bg_x, fetched.bg_h, fetched.bg_w

                    self.imshow(ax, fetched.bg_raw[idx])
                    if t == 0:
                        title = flt('bg_raw', y=bg_y[idx, t, 0], x=bg_x[idx, t, 0], h=bg_h[idx, t, 0], w=bg_w[idx, t, 0])
                        ax.set_title(title)

                    height = bg_h[idx, t, 0] * image_height
                    top = (bg_y[idx, t, 0] + 1) / 2 * image_height - height / 2

                    width = bg_w[idx, t, 0] * image_width
                    left = (bg_x[idx, t, 0] + 1) / 2 * image_width - width / 2

                    rect = patches.Rectangle(
                        (left, top), width, height, linewidth=lw, edgecolor="xkcd:green", facecolor='none')
                    ax.add_patch(rect)

                for i, (name, disc_axes, prop_axes, select_axes, other_axes) in enumerate(axes_sets):
                    _fetched = getattr(fetched, name)

                    final_weights = _fetched.select.final_weights[idx, t].sum(axis=0)
                    obj_idx = 0

                    # --- disc objects ---

                    for h, w, b in product(range(H), range(W), range(B)):
                        obj = _fetched.disc.obj[idx, t, obj_idx, 0]
                        render_obj = _fetched.disc.render_obj[idx, t, obj_idx, 0]
                        z = _fetched.disc.z[idx, t, obj_idx, 0]

                        ax = disc_axes[h * B + b, M * w]

                        color = obj * self.on_color + (1-obj) * self.off_color

                        self.imshow(ax, _fetched.disc.glimpse[idx, t, obj_idx, :, :, :])

                        obj_rect = patches.Rectangle(
                            (1., 0), 0.2, 1, clip_on=False, transform=ax.transAxes, facecolor=color)
                        ax.add_patch(obj_rect)

                        ax = disc_axes[h * B + b, M * w + 1]
                        self.imshow(ax, _fetched.disc.appearance[idx, t, obj_idx, :, :, :3])

                        fw = final_weights[n_prop_objects + obj_idx]

                        color = fw * self.selected_color + (1-fw) * self.unselected_color
                        obj_rect = patches.Rectangle(
                            (1., 0), 0.2, 1, clip_on=False, transform=ax.transAxes, facecolor=color)
                        ax.add_patch(obj_rect)

                        yt, xt, ys, xs = _fetched.disc.normalized_box[idx, t, obj_idx]

                        nbox = "bx={:.2f},{:.2f},{:.2f},{:.2f}".format(yt, xt, ys, xs)
                        ax.set_title(flt(nbox, obj=obj, robj=render_obj, z=z, final_weight=fw))

                        ax = disc_axes[h * B + b, M * w + 2]
                        self.imshow(ax, _fetched.disc.appearance[idx, t, obj_idx, :, :, 3], cmap="gray")

                        obj_idx += 1

                    # --- prop objects ---

                    for k in range(n_prop_objects):
                        obj = _fetched.prop.obj[idx, t, k, 0]
                        z = _fetched.prop.z[idx, t, k, 0]
                        d_obj = _fetched.prop.d_obj[idx, t, k, 0]
                        xs_logit = _fetched.prop.xs_logit[idx, t, k, 0]
                        ys_logit = _fetched.prop.ys_logit[idx, t, k, 0]
                        d_xt_logit = _fetched.prop.d_xt_logit[idx, t, k, 0]
                        d_yt_logit = _fetched.prop.d_yt_logit[idx, t, k, 0]
                        xs = _fetched.prop.xs[idx, t, k, 0]
                        ys = _fetched.prop.ys[idx, t, k, 0]
                        xt = _fetched.prop.xt[idx, t, k, 0]
                        yt = _fetched.prop.yt[idx, t, k, 0]

                        # --- object location superimposed on reconstruction ---

                        ax_idx = M*k
                        ax = prop_axes[ax_idx]
                        self.imshow(ax, _fetched.render.output[idx, t])

                        color = obj * self.on_color + (1-obj) * self.off_color
                        top, left, height, width = _fetched.prop.pixel_space_box[idx, t, k]
                        rect = patches.Rectangle(
                            (left, top), width, height, linewidth=lw, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)

                        top, left, height, width = _fetched.prop.glimpse_prime_pixel_space_box[idx, t, k]
                        rect = patches.Rectangle(
                            (left, top), width, height, linewidth=lw, edgecolor=self.glimpse_color, facecolor='none')
                        ax.add_patch(rect)

                        # --- glimpse ---

                        ax_idx += 1
                        ax = prop_axes[ax_idx]
                        self.imshow(ax, _fetched.prop.glimpse[idx, t, k, :, :, :])

                        color = obj * self.on_color + (1-obj) * self.off_color
                        obj_rect = patches.Rectangle(
                            (1., 0), 0.2, 1, clip_on=False, transform=ax.transAxes, facecolor=color)
                        ax.add_patch(obj_rect)

                        # --- glimpse mask ---

                        if updater.network.use_sqair_prop:
                            ax_idx += 1
                            ax = prop_axes[ax_idx]
                            self.imshow(ax, _fetched.prop.glimpse_mask[idx, t, k, :, :, 0], cmap="gray")

                        # --- glimpse_prime ---

                        ax_idx += 1
                        ax = prop_axes[ax_idx]
                        self.imshow(ax, _fetched.prop.glimpse_prime[idx, t, k, :, :, :])

                        fw = final_weights[k]
                        color = fw * self.selected_color + (1-fw) * self.unselected_color
                        obj_rect = patches.Rectangle(
                            (1., 0), 0.2, 1, clip_on=False, transform=ax.transAxes, facecolor=color)
                        ax.add_patch(obj_rect)

                        # --- glimpse_prime mask ---

                        if updater.network.use_sqair_prop:
                            ax_idx += 1
                            ax = prop_axes[ax_idx]
                            self.imshow(ax, _fetched.prop.glimpse_prime_mask[idx, t, k, :, :, 0], cmap="gray")

                        # --- appearance ---

                        ax_idx += 1
                        ax = prop_axes[ax_idx]
                        self.imshow(ax, _fetched.prop.appearance[idx, t, k, :, :, :3])
                        nbox = "bx={:.2f},{:.2f},{:.2f},{:.2f}".format(yt, xt, ys, xs)
                        d_nbox = "dbxl={:.2f},{:.2f},{:.2f},{:.2f}".format(d_yt_logit, d_xt_logit, ys_logit, xs_logit)
                        ax.set_title(flt(nbox + ", " + d_nbox, dobj=d_obj, obj=obj, z=z,))

                        # --- alpha ---

                        ax_idx += 1
                        ax = prop_axes[ax_idx]
                        self.imshow(ax, _fetched.prop.appearance[idx, t, k, :, :, 3], cmap="gray")

                    # --- select object ---

                    prop_weight_images = None

                    prop_weight_images = _fetched.select.final_weights[idx, t, :, :n_prop_objects]
                    _H = int(np.ceil(n_prop_objects / W))
                    padding = W * _H - n_prop_objects
                    prop_weight_images = np.pad(prop_weight_images, ((0, 0), (0, padding)), 'constant')
                    prop_weight_images = prop_weight_images.reshape(n_prop_objects, _H, W, 1)
                    prop_weight_images = (
                        prop_weight_images * self.selected_color + (1-prop_weight_images) * self.unselected_color)

                    final_weight_images = _fetched.select.final_weights[idx, t, :, n_prop_objects:]

                    final_weight_images = final_weight_images.reshape(n_prop_objects, H, W, 1)
                    final_weight_images = (
                        final_weight_images * self.selected_color + (1-final_weight_images) * self.unselected_color)

                    for k in range(n_prop_objects):
                        obj = _fetched.select.obj[idx, t, k, 0]
                        z = _fetched.select.z[idx, t, k, 0]
                        xs = _fetched.select.xs[idx, t, k, 0]
                        ys = _fetched.select.ys[idx, t, k, 0]
                        xt = _fetched.select.xt[idx, t, k, 0]
                        yt = _fetched.select.yt[idx, t, k, 0]

                        ax = select_axes[M*k]

                        self.imshow(ax, prop_weight_images[k])

                        ax = select_axes[M*k+1]

                        ax.set_title(flt(obj=obj, z=z, xs=xs, ys=ys, xt=xt, yt=yt))
                        self.imshow(ax, final_weight_images[k])

                        color = obj * self.on_color + (1-obj) * self.off_color
                        obj_rect = patches.Rectangle(
                            (-0.2, 0), 0.2, 1, clip_on=False, transform=ax.transAxes, facecolor=color)
                        ax.add_patch(obj_rect)

                    # --- other ---

                    ax = other_axes[6]
                    self.imshow(ax, fetched.inp[idx, t])
                    if t == 0:
                        ax.set_title('input')

                    ax = other_axes[7]
                    self.imshow(ax, _fetched.render.output[idx, t])
                    if t == 0:
                        ax.set_title('reconstruction')

                    ax = other_axes[8]
                    self.imshow(ax, _fetched.render.diff[idx, t])
                    if t == 0:
                        ax.set_title('abs error')

                    ax = other_axes[9]
                    self.imshow(ax, _fetched.render.xent[idx, t])
                    if t == 0:
                        ax.set_title('xent')

                    gt_axes = []
                    axis_idx = 0

                    names = ('select', 'disc', 'prop')

                    for name in names:
                        ax_all_bb = other_axes[axis_idx]
                        self.imshow(ax_all_bb, _fetched.render.output[idx, t])
                        if t == 0:
                            ax_all_bb.set_title('{} all bb'.format(name))

                        ax_on_bb = other_axes[axis_idx+1]
                        self.imshow(ax_on_bb, _fetched.render.output[idx, t])
                        if t == 0:
                            ax_on_bb.set_title('{} on bb'.format(name))

                        axis_idx += 2
                        gt_axes.extend([ax_all_bb, ax_on_bb])

                        flat_obj = getattr(_fetched, name).obj[idx, t].flatten()
                        flat_box = getattr(_fetched, name).pixel_space_box[idx, t].reshape(-1, 4)

                        # Plot proposed bounding boxes
                        for o, (top, left, height, width) in zip(flat_obj, flat_box):
                            color = o * self.on_color + (1-o) * self.off_color

                            rect = patches.Rectangle(
                                (left, top), width, height, linewidth=lw, edgecolor=color, facecolor='none')
                            ax_all_bb.add_patch(rect)

                            if o > self.cutoff:
                                rect = patches.Rectangle(
                                    (left, top), width, height, linewidth=lw, edgecolor=color, facecolor='none')
                                ax_on_bb.add_patch(rect)

                    # Plot true bounding boxes
                    for k in range(fetched.n_annotations[idx]):
                        valid, _, _, top, bottom, left, right = fetched.annotations[idx, t, k]

                        if not valid:
                            continue

                        height = bottom - top
                        width = right - left

                        for ax in gt_axes:
                            rect = patches.Rectangle(
                                (left, top), width, height, linewidth=lw, edgecolor=self.gt_color, facecolor='none')
                            ax.add_patch(rect)

            plt.subplots_adjust(left=0.02, right=.98, top=.95, bottom=0.02, wspace=0.1, hspace=0.12)

            anim = animation.FuncAnimation(fig, func, frames=T, interval=500)

            path = self.path_for('patches/{}'.format(idx), updater, ext="mp4")
            anim.save(path, writer='ffmpeg', codec='hevc', extra_args=['-preset', 'ultrafast'])

            plt.close(fig)

            shutil.copyfile(
                path,
                os.path.join(
                    os.path.dirname(path),
                    'latest_stage{:0>4}.mp4'.format(updater.stage_idx)))
