import numpy as np
import tensorflow as tf
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb

from dps.utils import Param
from dps.utils.tf import build_scheduled_value, RenderHook, tf_mean_sum, tf_shape

from auto_yolo.models.core import AP, xent_loss
from auto_yolo.models.object_layer import ObjectLayer

from spair_video.core import VideoNetwork


class SequentialSpair(VideoNetwork):
    build_backbone = Param()
    build_feature_fuser = Param()
    build_obj_feature_extractor = Param()

    n_backbone_features = Param()
    anchor_boxes = Param()

    train_reconstruction = Param()
    reconstruction_weight = Param()
    train_kl = Param()
    kl_weight = Param()

    backbone = None
    object_layer = None
    feature_fuser = None
    obj_feature_extractor = None

    @property
    def eval_funcs(self):
        if "annotations" in self._tensors:
            if self._eval_funcs is None:
                ap_iou_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                eval_funcs = {"AP_at_point_{}".format(int(10 * v)): AP(v) for v in ap_iou_values}
                eval_funcs["AP"] = AP(ap_iou_values)
                self._eval_funcs = eval_funcs
            return self._eval_funcs
        else:
            return {}

    def build_representation(self):
        # --- init modules ---
        self.B = len(self.anchor_boxes)

        if self.backbone is None:
            self.backbone = self.build_backbone(scope="backbone")
            if "backbone" in self.fixed_weights:
                self.backbone.fix_variables()

        if self.feature_fuser is None:
            self.feature_fuser = self.build_feature_fuser(scope="feature_fuser")
            if "feature_fuser" in self.fixed_weights:
                self.feature_fuser.fix_variables()

        if self.obj_feature_extractor is None and self.build_obj_feature_extractor is not None:
            self.obj_feature_extractor = self.build_obj_feature_extractor(scope="obj_feature_extractor")
            if "obj_feature_extractor" in self.fixed_weights:
                self.obj_feature_extractor.fix_variables()

        backbone_output, n_grid_cells, grid_cell_size = self.backbone(
            self.inp, self.B*self.n_backbone_features, self.is_training)

        self.H, self.W = [int(i) for i in n_grid_cells]
        self.HWB = self.H * self.W * self.B
        self.pixels_per_cell = tuple(int(i) for i in grid_cell_size)
        H, W, B = self.H, self.W, self.B

        if self.object_layer is None:
            self.object_layer = ObjectLayer(self.pixels_per_cell, scope="objects")

        self.object_rep_tensors = []
        object_rep_tensors = None
        _tensors = defaultdict(list)

        for f in range(self.n_frames):
            print("Bulding network for frame {}".format(f))
            early_frame_features = backbone_output[:, f]

            if f > 0 and self.obj_feature_extractor is not None:
                object_features = object_rep_tensors["all"]
                object_features = tf.reshape(
                    object_features, (self.batch_size, H, W, B*tf_shape(object_features)[-1]))
                early_frame_features += self.obj_feature_extractor(
                    object_features, B*self.n_backbone_features, self.is_training)

            frame_features = self.feature_fuser(
                early_frame_features, B*self.n_backbone_features, self.is_training)

            frame_features = tf.reshape(
                frame_features, (self.batch_size, H, W, B, self.n_backbone_features))

            object_rep_tensors = self.object_layer(
                self.inp[:, f], frame_features, self._tensors["background"][:, f], self.is_training)

            self.object_rep_tensors.append(object_rep_tensors)

            for k, v in object_rep_tensors.items():
                _tensors[k].append(v)

        self._tensors.update(**{k: tf.stack(v, axis=1) for k, v in _tensors.items()})

        # --- specify values to record ---

        obj = self._tensors["obj"]
        pred_n_objects = self._tensors["pred_n_objects"]

        self.record_tensors(
            batch_size=self.batch_size,
            float_is_training=self.float_is_training,

            cell_y=self._tensors["cell_y"],
            cell_x=self._tensors["cell_x"],
            h=self._tensors["h"],
            w=self._tensors["w"],
            z=self._tensors["z"],
            area=self._tensors["area"],

            cell_y_std=self._tensors["cell_y_std"],
            cell_x_std=self._tensors["cell_x_std"],
            h_std=self._tensors["h_std"],
            w_std=self._tensors["w_std"],
            z_std=self._tensors["z_std"],

            n_objects=pred_n_objects,
            obj=obj,

            latent_area=self._tensors["latent_area"],
            latent_hw=self._tensors["latent_hw"],

            attr=self._tensors["attr"],
        )

        # --- losses ---

        if self.train_reconstruction:
            output = self._tensors['output']
            inp = self._tensors['inp']
            self._tensors['per_pixel_reconstruction_loss'] = xent_loss(pred=output, label=inp)
            self.losses['reconstruction'] = (
                self.reconstruction_weight * tf_mean_sum(self._tensors['per_pixel_reconstruction_loss'])
            )

        if self.train_kl:
            self.losses.update(
                obj_kl=self.kl_weight * tf_mean_sum(self._tensors["obj_kl"]),
                cell_y_kl=self.kl_weight * tf_mean_sum(obj * self._tensors["cell_y_kl"]),
                cell_x_kl=self.kl_weight * tf_mean_sum(obj * self._tensors["cell_x_kl"]),
                h_kl=self.kl_weight * tf_mean_sum(obj * self._tensors["h_kl"]),
                w_kl=self.kl_weight * tf_mean_sum(obj * self._tensors["w_kl"]),
                z_kl=self.kl_weight * tf_mean_sum(obj * self._tensors["z_kl"]),
                attr_kl=self.kl_weight * tf_mean_sum(obj * self._tensors["attr_kl"]),
            )

        # --- other evaluation metrics ---

        if "n_annotations" in self._tensors:
            count_1norm = tf.to_float(
                tf.abs(tf.to_int32(self._tensors["pred_n_objects_hard"]) - self._tensors["n_valid_annotations"]))

            self.record_tensors(
                count_1norm=count_1norm,
                count_error=count_1norm > 0.5,
            )

