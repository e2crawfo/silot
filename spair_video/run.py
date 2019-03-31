import numpy as np

from dps import cfg
from dps.hyper import run_experiment
from dps.utils import Config
from dps.datasets.base import VisualArithmeticDataset

from auto_yolo.models.core import Updater

from spair_video.core import VideoVAE


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

    n_train=25000,
    n_val=1e3,

    lr_schedule=1e-4,
    optimizer_spec="adam",
    max_grad_norm=10.0,
    eval_step=100,
    display_step=100,
    render_step=100,
    max_steps=int(3e5),
)


env_config = Config(
    env_name="moving_mnist",
    build_env=MovingMNIST,

    n_patch_examples=0,
    image_shape=(48, 48),
    patch_shape=(14, 14),
    min_digits=1,
    max_digits=9,
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
    n_frames=3,

    backgrounds="",
    backgrounds_sample_every=False,
    background_colours="",
    background_cfg=dict(mode="colour", colour="black"),
    object_shape=(14, 14),
    postprocessing="",
)


alg_config = Config(
    alg_name="VideoVAE",
    get_updater=Updater,
    build_network=VideoVAE,

    fixed_weights="",
    fixed_values={},
    no_gradient="",

    attr_prior_mean=0.,
    attr_prior_std=1.0,

    A=128,

    train_reconstruction=True,
    reconstruction_weight=1.0,

    train_kl=True,
    kl_weight=1.0,

    noisy=True,
    max_possible_objects=0,
)


if __name__ == "__main__":
    config = basic_config.copy()
    config.update(**env_config, **alg_config)

    run_experiment("test_spair_video", config, "First test of spair_video.")
