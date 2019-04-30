import numpy as np
import tensorflow as tf
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb

from dps import cfg
from dps.utils import Param
from dps.utils.tf import RenderHook, tf_mean_sum, tf_shape

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
        if getattr(self, '_eval_funcs', None) is None:
            if "annotations" in self._tensors:
                ap_iou_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                eval_funcs = {"AP_at_point_{}".format(int(10 * v)): AP(v) for v in ap_iou_values}
                eval_funcs["AP"] = AP(ap_iou_values)
                self._eval_funcs = eval_funcs
            else:
                self._eval_funcs = {}

        return self._eval_funcs

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

            if cfg.background_cfg.mode == "learn_and_transform":
                self.losses.update(
                    bg_attr_kl=self.kl_weight * tf_mean_sum(self._tensors["bg_attr_kl"]),
                    bg_transform_kl=self.kl_weight * tf_mean_sum(self._tensors["bg_transform_kl"]),
                )

        # --- other evaluation metrics ---

        if "n_annotations" in self._tensors:
            count_1norm = tf.to_float(
                tf.abs(tf.to_int32(self._tensors["pred_n_objects_hard"]) - self._tensors["n_valid_annotations"]))

            self.record_tensors(
                count_1norm=count_1norm,
                count_error=count_1norm > 0.5,
            )


class SequentialSpair_RenderHook(RenderHook):
    N = 4
    linewidth = 2
    on_color = np.array(to_rgb("xkcd:azure"))
    off_color = np.array(to_rgb("xkcd:red"))
    gt_color = "xkcd:yellow"
    cutoff = 0.5

    fetches = "obj raw_obj z inp output appearance normalized_box background glimpse"

    def __call__(self, updater):
        network = updater.network
        if "n_annotations" in network._tensors:
            self.fetches += " annotations n_annotations"

        if 'prediction' in network._tensors:
            self.fetches += " prediction targets"

        if "actions" in network._tensors:
            self.fetches += " actions"

        if "bg_y" in network._tensors:
            self.fetches += " bg_y bg_x bg_h bg_w bg_raw"

        fetched = self._fetch(updater)

        self._prepare_fetched(fetched)

        # self._plot_reconstruction(updater, fetched)
        self._plot_patches(updater, fetched)

        # try:
        #     self._plot_reconstruction(updater, fetched)
        # except Exception:
        #     pass

    @staticmethod
    def normalize_images(images):
        mx = images.reshape(*images.shape[:-3], -1).max(axis=-1)
        return images / mx[..., None, None, None]

    def _prepare_fetched(self, fetched):
        inp = fetched['inp']
        output = fetched['output']
        prediction = fetched.get("prediction", None)
        targets = fetched.get("targets", None)

        N, T, image_height, image_width, _ = inp.shape

        flat_obj = fetched['obj'].reshape(N, T, -1)
        background = fetched['background']

        box = (
            fetched['normalized_box']
            * [image_height, image_width, image_height, image_width]
        )
        flat_box = box.reshape(N, T, -1, 4)

        n_annotations = fetched.get("n_annotations", np.zeros(N, dtype='i'))
        annotations = fetched.get("annotations", None)
        # actions = fetched.get("actions", None)

        diff = self.normalize_images(np.abs(inp - output).mean(axis=-1, keepdims=True))
        xent = self.normalize_images(
            xent_loss(pred=output, label=inp, tf=False).mean(axis=-1, keepdims=True))

        learned_bg = "bg_y" in fetched
        bg_y = fetched.get("bg_y", None)
        bg_x = fetched.get("bg_x", None)
        bg_h = fetched.get("bg_h", None)
        bg_w = fetched.get("bg_w", None)
        bg_raw = fetched.get("bg_raw", None)

        fetched.update(
            prediction=prediction,
            targets=targets,
            flat_obj=flat_obj,
            background=background,
            box=box,
            flat_box=flat_box,
            n_annotations=n_annotations,
            annotations=annotations,
            diff=diff,
            xent=xent,
            learned_bg=learned_bg,
            bg_y=bg_y,
            bg_x=bg_x,
            bg_h=bg_h,
            bg_w=bg_w,
            bg_raw=bg_raw,
        )

    def _plot_reconstruction(self, updater, fetched):
        N, T, image_height, image_width, _ = fetched['inp'].shape

        print("Plotting for {} data points...".format(N))
        n_images = 8 if fetched['learned_bg'] else 7

        fig_unit_size = 4
        fig_height = T * fig_unit_size
        fig_width = n_images * fig_unit_size

        for n in range(N):
            fig, axes = plt.subplots(T, n_images, figsize=(fig_width, fig_height))

            if fetched['prediction'] is not None:
                fig_title = "target={}, prediction={}".format(
                    np.argmax(fetched['targets'][n]),
                    np.argmax(fetched['prediction'][n]))
                fig.suptitle(fig_title, fontsize=16)

            for ax in axes.flatten():
                ax.set_axis_off()

            for t in range(T):
                self._plot_helper(n, t, axes[t], **fetched)

            plt.subplots_adjust(left=0.02, right=.98, top=.98, bottom=0.02, wspace=0.1, hspace=0.1)
            self.savefig("reconstruction/" + str(n), fig, updater)

    def _plot_helper(
            self, n, t, axes, *, inp, output, diff, xent, background, flat_obj, flat_box,
            n_annotations, annotations, learned_bg, bg_y, bg_x, bg_h, bg_w, bg_raw, **kwargs):

        N, T, image_height, image_width, _ = inp.shape
        lw = self.linewidth

        def safe_remove(obj):
            try:
                obj.remove()
            except NotImplementedError:
                pass

        ax_inp = axes[0]
        self.imshow(ax_inp, inp[n, t])
        for obj in ax_inp.findobj(match=plt.Rectangle):
            safe_remove(obj)
        if t == 0:
            ax_inp.set_title('input')

        ax = axes[1]
        self.imshow(ax, output[n, t])
        if t == 0:
            ax.set_title('reconstruction')

        ax = axes[2]
        self.imshow(ax, diff[n, t])
        if t == 0:
            ax.set_title('abs error')

        ax = axes[3]
        self.imshow(ax, xent[n, t])
        if t == 0:
            ax.set_title('xent')

        ax_all_bb = axes[4]
        self.imshow(ax_all_bb, output[n, t])
        for obj in ax_all_bb.findobj(match=plt.Rectangle):
            safe_remove(obj)
        if t == 0:
            ax_all_bb.set_title('all bb')

        ax_proposed_bb = axes[5]
        self.imshow(ax_proposed_bb, output[n, t])
        for obj in ax_proposed_bb.findobj(match=plt.Rectangle):
            safe_remove(obj)
        if t == 0:
            ax_proposed_bb.set_title('proposed bb')

        ax = axes[6]
        self.imshow(ax, background[n, t])
        if t == 0:
            ax.set_title('background')

        # Plot proposed bounding boxes
        for o, (top, left, height, width) in zip(flat_obj[n, t], flat_box[n, t]):
            color = o * self.on_color + (1-o) * self.off_color

            rect = patches.Rectangle(
                (left, top), width, height, linewidth=lw, edgecolor=color, facecolor='none')
            ax_all_bb.add_patch(rect)

            if o > self.cutoff:
                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=lw, edgecolor=color, facecolor='none')
                ax_proposed_bb.add_patch(rect)

        # Plot true bounding boxes
        for k in range(n_annotations[n]):
            valid, _, top, bottom, left, right = annotations[n, t, k]

            if not valid:
                continue

            height = bottom - top
            width = right - left

            rect = patches.Rectangle(
                (left, top), width, height, linewidth=lw, edgecolor=self.gt_color, facecolor='none')
            ax_inp.add_patch(rect)

            rect = patches.Rectangle(
                (left, top), width, height, linewidth=lw, edgecolor=self.gt_color, facecolor='none')
            ax_all_bb.add_patch(rect)

            rect = patches.Rectangle(
                (left, top), width, height, linewidth=lw, edgecolor=self.gt_color, facecolor='none')
            ax_proposed_bb.add_patch(rect)

        if learned_bg:
            ax = axes[7]
            self.imshow(ax, bg_raw[n])
            for obj in ax.findobj(match=plt.Rectangle):
                safe_remove(obj)
            if t == 0:
                ax.set_title('raw_bg, y={:.2f}, x={:.2f}, h={:.2f}, w={:.2f}'.format(
                    bg_y[n, t, 0], bg_x[n, t, 0], bg_h[n, t, 0], bg_w[n, t, 0]))

            height = bg_h[n, t, 0] * image_height
            top = (bg_y[n, t, 0] + 1) / 2 * image_height - height / 2

            width = bg_w[n, t, 0] * image_width
            left = (bg_x[n, t, 0] + 1) / 2 * image_width - width / 2

            rect = patches.Rectangle(
                (left, top), width, height, linewidth=lw, edgecolor="xkcd:green", facecolor='none')
            ax.add_patch(rect)

    def _plot_patches(self, updater, fetched):
        # Create a plot showing what each object is generating

        import matplotlib.pyplot as plt
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        from matplotlib import animation
        import matplotlib.gridspec as gridspec
        from itertools import product

        N, T, image_height, image_width, _ = fetched['inp'].shape
        H, W, B = updater.network.H, updater.network.W, updater.network.B

        glimpse = fetched['glimpse']
        appearance = fetched['appearance']
        obj = fetched['obj']
        raw_obj = fetched['raw_obj']
        z = fetched['z']

        fig_unit_size = 3
        fig_height = 2 * B * H * fig_unit_size
        fig_width = 3 * W * fig_unit_size

        for idx in range(N):
            fig = plt.figure(figsize=(fig_width, fig_height))
            time_text = fig.suptitle('', fontsize=20, fontweight='bold')

            gs = gridspec.GridSpec(2*B*H, 3*W, figure=fig)

            axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(3*W)] for i in range(B*H)])

            for ax in axes.flatten():
                ax.set_axis_off()

            other_axes = []
            for i in range(2):
                for j in range(4):
                    start_y = B*H + 2*i
                    end_y = start_y + 2
                    start_x = 2*j
                    end_x = start_x + 2
                    ax = fig.add_subplot(gs[start_y:end_y, start_x:end_x])
                    other_axes.append(ax)

            other_axes = np.array(other_axes)

            for ax in other_axes.flatten():
                ax.set_axis_off()

            print("Plotting patches for {}...".format(idx))

            def func(t, axes=axes, other_axes=other_axes):
                print("timestep {}".format(t))
                time_text.set_text('t = {}'.format(t))

                for h, w, b in product(range(H), range(W), range(B)):
                    _obj = obj[idx, t, h, w, b, 0]
                    _raw_obj = raw_obj[idx, t, h, w, b, 0]
                    _z = z[idx, t, h, w, b, 0]

                    ax = axes[h * B + b, 3 * w]

                    color = _obj * self.on_color + (1-_obj) * self.off_color
                    obj_rect = patches.Rectangle(
                        (-0.2, 0), 0.2, 1, clip_on=False, transform=ax.transAxes, facecolor=color)
                    ax.add_patch(obj_rect)

                    if t == 0 and h == 0 and b == 0:
                        ax.set_title("w={}".format(w))
                    if t == 0 and w == 0 and b == 0:
                        ax.set_ylabel("h={}".format(h))

                    self.imshow(ax, glimpse[idx, t, h, w, b, :, :, :])

                    ax = axes[h * B + b, 3 * w + 1]
                    self.imshow(ax, appearance[idx, t, h, w, b, :, :, :3])

                    ax.set_title("obj={:.2f}, raw_obj={:.2f}, z={:.2f}".format(_obj, _raw_obj, _z, b))

                    ax = axes[h * B + b, 3 * w + 2]
                    self.imshow(ax, appearance[idx, t, h, w, b, :, :, 3], cmap="gray")

                self._plot_helper(idx, t, other_axes, **fetched)

            plt.subplots_adjust(left=0.02, right=.98, top=.9, bottom=0.02, wspace=0.1, hspace=0.1)

            anim = animation.FuncAnimation(fig, func, frames=T, interval=500)

            path = self.path_for('patches/{}'.format(idx), updater, ext="mp4")
            anim.save(path, writer='ffmpeg', codec='hevc')

            plt.close(fig)
