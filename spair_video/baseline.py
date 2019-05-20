import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib.patches as patches


from dps import cfg
from dps.utils.tf import tf_shape, RenderHook
from dps.utils import Param, Config

from auto_yolo.models.baseline import tf_find_connected_components
from auto_yolo.models.core import AP

from spair_video.core import VideoNetwork, coords_to_pixel_space


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
    stage_steps = Param()
    initial_n_frames = Param()
    n_frames_scale = Param()

    needs_background = True

    def __init__(self, env, updater, scope=None, **kwargs):
        ap_iou_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.eval_funcs = {"AP_at_point_{}".format(int(10 * v)): BaselineAP(v) for v in ap_iou_values}
        self.eval_funcs["AP"] = BaselineAP(ap_iou_values)

        super().__init__(env, updater, scope=scope, **kwargs)

    def build_representation(self):
        assert cfg.background_cfg.mode == 'colour'

        # dummy variable to satisfy dps
        tf.get_variable("dummy", shape=(1,), dtype=tf.float32)

        B, T, *rest = tf_shape(self._tensors["background"])

        inp = tf.reshape(self._tensors["inp"], (T*B, *rest))
        bg = tf.reshape(self._tensors["background"], (T*B, *rest))

        program_tensors = tf_find_connected_components(inp, bg, self.cc_threshold)
        self._tensors.update(
            {k: tf.reshape(v, (B, T, *tf_shape(v)[1:]))
             for k, v in program_tensors.items()
             if k != 'max_objects'}
        )

        if "n_annotations" in self._tensors:
            count_1norm = tf.to_float(
                tf.abs(tf.to_int32(self._tensors["n_objects"]) - self._tensors["n_valid_annotations"]))
            self.record_tensors(
                count_1norm=count_1norm,
                count_error=count_1norm > 0.5
            )