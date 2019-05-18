import tensorflow as tf

from dps.utils import Param

from auto_yolo.models.baseline import Baseline_Network
from auto_yolo.models.core import AP

from spair_video.core import VideoNetwork


class BaselineTracker(VideoNetwork, Baseline_Network):
    cc_threshold = Param()
    object_shape = Param()

    object_encoder = None
    object_decoder = None

    def __init__(self, env, updater, scope=None, **kwargs):
        """
        Have to run ConnectedComponents on each cell.
        Can I adapt the previous code to work?
        """
        ap_iou_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.eval_funcs = {"AP_at_point_{}".format(int(10 * v)): AP(v) for v in ap_iou_values}
        self.eval_funcs["AP"] = AP(ap_iou_values)

        super().__init__(env, updater, scope=scope, **kwargs)

    def build_representation(self):

        self.maybe_build_subnet("object_encoder")
        self.maybe_build_subnet("object_decoder")

        # TODO: reshape videos into images, run algorithm, reshape back into videos, then do tracking
        # by data association.

        program_tensors = self._build_program_generator(self._tensors)
        self._tensors.update(program_tensors)

        interpreter_tensors = self._build_program_interpreter(self._tensors)
        self._tensors.update(interpreter_tensors)

        # --- specify values to record ---

        self.record_tensors(
            n_objects=self._tensors["n_objects"],
            attr=self._tensors["attr"]
        )

        # --- losses ---

        if self.train_reconstruction:
            output = self._tensors['output']
            inp = self._tensors['inp']
            self._tensors['per_pixel_reconstruction_loss'] = xent_loss(pred=output, label=inp)
            self.losses['reconstruction'] = (
                self.reconstruction_weight
                * tf_mean_sum(self._tensors['per_pixel_reconstruction_loss'])
            )

        if self.train_kl:
            obj = self._tensors["obj"]
            self.losses['attr_kl'] = self.kl_weight * tf_mean_sum(obj * self._tensors["attr_kl"])

        # --- other evaluation metrics

        if "n_annotations" in self._tensors:
            count_1norm = tf.to_float(
                tf.abs(tf.to_int32(self._tensors["n_objects"]) - self._tensors["n_annotations"]))
            self.record_tensors(
                count_1norm=count_1norm,
                count_error=count_1norm > 0.5
            )