import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib.patches as patches
from scipy import optimize


from dps import cfg
from dps.utils.tf import tf_shape, RenderHook, build_scheduled_value
from dps.utils import Param, Config

from auto_yolo.models.baseline import tf_find_connected_components
from auto_yolo.models.core import AP

from spair_video.core import VideoNetwork, coords_to_pixel_space, MOTMetrics


class BaselineAP(AP):
    keys_accessed = "normalized_box obj annotations n_annotations"

    def _process_data(self, tensors, updater):
        obj = tensors['obj']
        y, x, height, width = np.split(tensors['normalized_box'], 4, axis=-1)

        image_shape = (updater.network.image_height, updater.network.image_width)
        anchor_box = updater.network.anchor_box
        top, left, height, width = coords_to_pixel_space(y, x, height, width, image_shape, anchor_box, top_left=True)

        batch_size = obj.shape[0]
        n_frames = getattr(updater.network, 'n_frames', 0)

        annotations = tensors["annotations"]
        n_annotations = tensors["n_annotations"]

        if n_frames > 0:
            n_objects = np.prod(obj.shape[2:-1])
            n_frames = obj.shape[1]
        else:
            n_objects = np.prod(obj.shape[1:-1])
            annotations = annotations.reshape(batch_size, 1, *annotations.shape[1:])
            n_frames = 1

        shape = (batch_size, n_frames, n_objects)

        obj = obj.reshape(*shape)
        n_digits = n_objects * np.ones((batch_size, n_frames), dtype=np.int32)
        top = top.reshape(*shape)
        left = left.reshape(*shape)
        height = height.reshape(*shape)
        width = width.reshape(*shape)

        return obj, n_digits, top, left, height, width, annotations, n_annotations


class BaselineMOTMetrics(MOTMetrics):
    keys_accessed = "color normalized_box obj annotations n_annotations"

    dist_threshold = 30

    def _process_data(self, tensors, updater):
        obj = tensors['obj']
        color = tensors['color']

        B, F, n_objects = shape = obj.shape[:3]
        obj = obj.reshape(shape)
        color = color.reshape(shape)

        y, x, height, width = np.split(tensors['normalized_box'], 4, axis=-1)
        image_shape = (updater.network.image_height, updater.network.image_width)
        anchor_box = updater.network.anchor_box
        top, left, height, width = coords_to_pixel_space(y, x, height, width, image_shape, anchor_box, top_left=True)

        top = top.reshape(shape)
        left = left.reshape(shape)
        height = height.reshape(shape)
        width = width.reshape(shape)

        cy = top + height / 2
        cx = left + width / 2

        ids = np.zeros((B, F), dtype=np.object)

        nan_value = 10000.

        for b in range(B):
            prev_ids = np.arange(int(obj[b, 0].sum()))
            next_id = np.amax(prev_ids, initial=0) + 1
            ids[b, 0] = prev_ids

            prev_centroids = [[cy[b, 0, i], cx[b, 0, i]] for i in range(n_objects) if obj[b, 0, i] > 0.5]
            prev_centroids = np.array(prev_centroids)

            prev_colors = [color[b, 0, i] for i in range(n_objects) if obj[b, 0, i] > 0.5]
            prev_colors = np.array(prev_colors).astype('i')

            for f in range(1, F):
                new_centroids = [[cy[b, f, i], cx[b, f, i]] for i in range(n_objects) if obj[b, f, i] > 0.5]
                new_centroids = np.array(new_centroids)

                new_colors = [color[b, f, i] for i in range(n_objects) if obj[b, f, i] > 0.5]
                new_colors = np.array(new_colors).astype('i')
                n_new_objects = new_centroids.shape[0]

                if not len(prev_ids):
                    # no objects on previous timestep
                    new_ids = np.arange(next_id, next_id+n_new_objects)
                    next_id += n_new_objects

                    ids[b, f] = new_ids

                    prev_centroids = new_centroids
                    prev_colors = new_colors
                    prev_ids = new_ids

                    continue

                if not len(new_centroids):
                    # no objects on current timestep
                    ids[b, f] = new_ids = []

                    prev_centroids = new_centroids
                    prev_colors = new_colors
                    prev_ids = new_ids

                    continue

                distances = np.linalg.norm(new_centroids[:, None, :] - prev_centroids[None, :, :], axis=2)
                distances[distances > self.dist_threshold] = nan_value

                matching_colors = new_colors[:, None] == prev_colors[None, :]
                distances += nan_value * (1 - matching_colors)

                new_ids = np.zeros(n_new_objects)

                row_indices, col_indices = optimize.linear_sum_assignment(distances)
                row_indices = list(row_indices)

                for j in range(n_new_objects):
                    match_found = False
                    if j in row_indices:
                        pos = row_indices.index(j)
                        col_idx = col_indices[pos]
                        distance = distances[j, col_idx]
                        if distance < nan_value:
                            # found a match
                            new_ids[j] = prev_ids[col_idx]
                            match_found = True

                    if not match_found:
                        new_ids[j] = next_id
                        next_id += 1

                ids[b, f] = new_ids

                prev_centroids = new_centroids
                prev_colors = new_colors
                prev_ids = new_ids

        pred_n_objects = n_objects * np.ones((B, F), dtype=np.int32)

        return obj, pred_n_objects, ids, top, left, height, width


class Baseline_RenderHook(RenderHook):
    fetches = "obj inp normalized_box annotations n_annotations"

    def __call__(self, updater):
        fetched = self._fetch(updater)
        fetched = Config(fetched)

        B, T, image_height, image_width, _ = fetched.inp.shape
        yt, xt, ys, xs = np.split(fetched.normalized_box, 4, axis=-1)
        pixel_space_box = coords_to_pixel_space(
            yt, xt, ys, xs, (image_height, image_width), updater.network.anchor_box, top_left=True)
        fetched.pixel_space_box = np.concatenate(pixel_space_box, axis=-1)

        lw = 0.1
        prop_color = np.array(to_rgb("xkcd:azure"))
        gt_color = np.array(to_rgb("xkcd:yellow"))

        fig_unit_size = 3

        fig_height = B * fig_unit_size
        fig_width = T * fig_unit_size

        fig, axes = plt.subplots(B, T, figsize=(fig_width, fig_height))

        for ax in axes.flatten():
            ax.set_axis_off()

        for b in range(B):
            for t in range(T):
                ax = axes[b, t]
                if b == 0:
                    ax.set_title('t={}'.format(t))
                self.imshow(ax, fetched.inp[b, t])

                for o, (top, left, height, width) in zip(fetched.obj[b, t], fetched.pixel_space_box[b, t]):
                    if o[0] > 0.5:
                        rect = patches.Rectangle(
                            (left, top), width, height, linewidth=lw, edgecolor=prop_color, facecolor='none')
                        ax.add_patch(rect)

                for k in range(fetched.n_annotations[b]):
                    valid, _, _, top, bottom, left, right = fetched.annotations[b, t, k]

                    if not valid:
                        continue

                    height = bottom - top
                    width = right - left

                    rect = patches.Rectangle(
                        (left, top), width, height, linewidth=lw, edgecolor=gt_color, facecolor='none')
                    ax.add_patch(rect)

        self.savefig('boxes', fig, updater, is_dir=False)


class BaselineTracker(VideoNetwork):
    attr_prior_mean = None
    attr_prior_std = None
    noisy = None

    anchor_box = Param()
    cc_threshold = Param()
    cosine_threshold = Param()
    stage_steps = Param()
    initial_n_frames = Param()
    n_frames_scale = Param()
    colours = Param()

    needs_background = True

    def __init__(self, env, updater, scope=None, **kwargs):
        ap_iou_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.eval_funcs = {"AP_at_point_{}".format(int(10 * v)): BaselineAP(v) for v in ap_iou_values}
        self.eval_funcs["AP"] = BaselineAP(ap_iou_values)
        self.eval_funcs["MOT"] = BaselineMOTMetrics()

        self.cc_threshold = build_scheduled_value(self.cc_threshold, 'cc_threshold')

        try:
            self.cosine_threshold = build_scheduled_value(self.cosine_threshold, 'cosine_threshold')
        except Exception:
            self.cosine_threshold = None

        super().__init__(env, updater, scope=scope, **kwargs)

    def build_representation(self):
        assert cfg.background_cfg.mode == 'colour'

        # dummy variable to satisfy dps
        tf.get_variable("dummy", shape=(1,), dtype=tf.float32)

        B, T, *rest = tf_shape(self._tensors["background"])

        inp = tf.reshape(self._tensors["inp"], (T*B, *rest))
        bg = tf.reshape(self._tensors["background"], (T*B, *rest))

        program_tensors = tf_find_connected_components(
            inp, bg, self.cc_threshold, self.colours, self.cosine_threshold)
        self._tensors.update(
            {k: tf.reshape(v, (B, T, *tf_shape(v)[1:]))
             for k, v in program_tensors.items()
             if k != 'max_objects'}
        )
        self.record_tensors(
            cc_threshold=self.cc_threshold,
            cosine_threshold=self.cosine_threshold if self.cosine_threshold is not None else tf.constant(0.0, tf.float32),
        )

        if "n_annotations" in self._tensors:
            count_1norm = tf.to_float(
                tf.abs(tf.to_int32(self._tensors["n_objects"]) - self._tensors["n_valid_annotations"]))
            self.record_tensors(
                count_1norm=count_1norm,
                count_error=count_1norm > 0.5
            )