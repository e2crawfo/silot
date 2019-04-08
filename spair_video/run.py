import numpy as np
import tensorflow as tf

from dps import cfg
from dps.hyper import run_experiment
from dps.utils import Config
from dps.datasets.base import VisualArithmeticDataset
from dps.datasets.shapes import RandomShapesDataset
from dps.utils.tf import MLP, CompositeCell

from auto_yolo.models.core import Updater

from spair_video.core import SimpleVideoVAE, SimpleVAE_RenderHook
from spair_video.tracking_by_animation import TrackingByAnimation, TbaBackbone, TBA_RenderHook


class MovingMNIST(object):
    def __init__(self):
        train_seed, val_seed, test_seed = 0, 1, 2

        train = VisualArithmeticDataset(
            n_examples=int(cfg.n_train), shuffle=True,
            episode_range=cfg.train_episode_range, seed=train_seed)

        val = VisualArithmeticDataset(
            n_examples=int(cfg.n_val), shuffle=True,
            episode_range=cfg.val_episode_range, seed=val_seed)

        test = VisualArithmeticDataset(
            n_examples=int(cfg.n_val), shuffle=True,
            episode_range=cfg.test_episode_range, seed=test_seed)

        self.datasets = dict(train=train, val=val, test=test)

    def close(self):
        pass


class MovingShapes(object):
    def __init__(self):
        train_seed, val_seed, test_seed = 0, 1, 2

        train = RandomShapesDataset(
            n_examples=int(cfg.n_train), shuffle=True, seed=train_seed)

        val = RandomShapesDataset(
            n_examples=int(cfg.n_val), shuffle=True, seed=val_seed)

        test = RandomShapesDataset(
            n_examples=int(cfg.n_val), shuffle=True, seed=test_seed)

        self.datasets = dict(train=train, val=val, test=test)

    def close(self):
        pass


basic_config = Config(
    use_gpu=True,
    gpu_allow_growth=True,
    stopping_criteria="loss,min",
    max_experiments=None,
    preserve_env=False,
    threshold=-np.inf,
    load_path=-1,

    curriculum=[dict()],

    batch_size=32,
    shuffle_buffer_size=32,
    patience=10000,

    n_train=60000,
    n_val=1e3,

    lr_schedule=1e-4,
    optimizer_spec="adam",
    max_grad_norm=10.0,
    eval_step=100,
    display_step=100,
    render_step=100,
    max_steps=int(3e5),

    get_updater=Updater,
    fixed_weights="",
    fixed_values={},
    no_gradient="",
    overwrite_plots=False,
    render_first=True
)


env_configs = dict(
    moving_mnist=Config(
        build_env=MovingMNIST,

        n_patch_examples=0,
        image_shape=(48, 48),
        patch_shape=(14, 14),
        object_shape=(14, 14),
        min_digits=1,
        max_digits=9,
        max_overlap=14**2/2,
        n_classes=82,
        largest_digit=81,
        one_hot=True,
        reductions="sum",
        characters=list(range(10)),
        patch_size_std=0.0,
        colours="white",
        n_distractors_per_image=0,

        train_episode_range=(0.0, 0.8),
        val_episode_range=(0.8, 0.9),
        test_episode_range=(0.9, 1.0),
        n_frames=5,

        backgrounds="",
        backgrounds_sample_every=False,
        background_colours="",
        background_cfg=dict(mode="colour", colour="black"),
        postprocessing="",
    ),
    easy_shapes=Config(
        build_env=MovingShapes,

        image_shape=(96, 96),
        patch_shape=(21, 21),
        object_shape=(21, 21),
        min_shapes=1,
        max_shapes=3,
        max_overlap=14**2/2,
        one_hot=True,
        colours="red green blue cyan magenta yellow",
        shapes="circle diamond x",
        n_distractors_per_image=0,

        n_frames=8,

        backgrounds="",
        backgrounds_sample_every=False,
        background_colours="black",
        background_cfg=dict(mode="colour", colour="black"),
        postprocessing="",
    )
)

alg_configs = dict(
    simple_vae=Config(
        build_network=SimpleVideoVAE,

        attr_prior_mean=0.,
        attr_prior_std=1.0,

        A=128,

        train_reconstruction=True,
        reconstruction_weight=1.0,

        train_kl=True,
        kl_weight=1.0,

        noisy=True,
        build_encoder=lambda scope: MLP([128, 128, 128], scope=scope),
        build_decoder=lambda scope: MLP([128, 128, 128], scope=scope),
        build_cell=lambda scope: CompositeCell(
            tf.contrib.rnn.GRUBlockCellV2(128),
            MLP([128], scope="GRU"), 2*128),
        render_hook=SimpleVAE_RenderHook(),
    ),
    tracking_by_animation=Config(
        # This is the config used for Sprites dataset in the paper

        build_network=TrackingByAnimation,

        build_backbone=TbaBackbone,
        build_cell=lambda scope, n_hidden: tf.contrib.rnn.GRUBlockCellV2(n_hidden),
        build_key_network=lambda scope: MLP([], scope=scope),
        build_write_network=lambda scope: MLP([], scope=scope),
        build_output_network=lambda scope: MLP([377], scope=scope),
        # another possibility, not clear from the paper:
        # build_output_network=lambda scope: MLP([80, 377], scope=scope),

        lr=5e-4,

        lmbda=0.0,
        # lmbda=1.0,
        n_trackers=4,
        n_layers=3,
        n_hidden=80,
        S=20,
        eta=(0.2, 0.2),
        prioritize=True,
        anchor_box=(21, 21),
        render_hook=TBA_RenderHook(),
        # fixed_values=dict(conf=1.),
        discrete_eval=False,
    )
)


for k, v in env_configs.items():
    v['env_name'] = k
for k, v in alg_configs.items():
    v['alg_name'] = k


if __name__ == "__main__":
    config = basic_config.copy()
    run_experiment(
        "test_spair_video", config, "First test of spair_video.",
        alg_configs=alg_configs, env_configs=env_configs)
