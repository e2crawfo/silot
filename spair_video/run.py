import numpy as np
import tensorflow as tf

from dps import cfg
from dps.hyper import run_experiment
from dps.utils import Config
from dps.datasets.base import VisualArithmeticDataset
from dps.datasets.shapes import RandomShapesDataset
from dps.utils.tf import MLP, CompositeCell, RecurrentGridConvNet, ConvNet
from dps.config import DEFAULT_CONFIG

from auto_yolo.models.core import Updater

from spair_video.core import SimpleVideoVAE, SimpleVAE_RenderHook, BackgroundExtractor
from spair_video.tracking_by_animation import TrackingByAnimation, TbaBackbone, TBA_RenderHook
from spair_video.sspair import SequentialSpair, SequentialSpair_RenderHook


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


basic_config = DEFAULT_CONFIG.copy(
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

    noisy=True,
    train_reconstruction=True,
    train_kl=True,
    reconstruction_weight=1.0,
    kl_weight=1.0,

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
        tile_shape=(48, 48),
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
        tile_shape=(96, 96),
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

env_configs["small_shapes"] = env_configs["easy_shapes"].copy(
    image_shape=(48, 48),
    tile_shape=(48, 48),
    patch_shape=(14, 14),
    object_shape=(14, 14),
)

env_configs["moving_background"] = env_configs["small_shapes"].copy(
    min_shapes=2,
    max_shapes=2,
    background_cfg=dict(
        mode="learn_and_transform", A=8,
        bg_shape=(60, 60),
        build_encoder=lambda scope: BackgroundExtractor(
            scope=scope,
            build_cell=lambda n_hidden, scope: tf.contrib.rnn.GRUBlockCellV2(n_hidden, name=scope),
            layers=[
                dict(filters=8, kernel_size=4, strides=3),
                dict(filters=8, kernel_size=4, strides=2),
                dict(filters=8, kernel_size=4, strides=2),
            ],
        ),
        build_decoder=lambda scope: ConvNet(
            scope=scope,
            layers=[
                dict(filters=8, kernel_size=4, strides=2, transpose=True,),
                dict(filters=8, kernel_size=4, strides=2, transpose=True,),
                dict(filters=8, kernel_size=4, strides=3, transpose=True,),
            ],
        ),
    )
)


def spair_prepare_func():
    from dps import cfg
    cfg.anchor_boxes = [cfg.tile_shape]
    cfg.count_prior_log_odds = (
        "Exp(start=1000000.0, end={}, decay_rate=0.1, "
        "decay_steps={}, log=True)".format(
            cfg.final_count_prior_log_odds, cfg.count_prior_decay_steps)
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
    ),
    sspair=Config(
        build_network=SequentialSpair,
        render_hook=SequentialSpair_RenderHook(),
        prepare_func=spair_prepare_func,

        stopping_criteria="AP,max",
        threshold=1.0,

        build_backbone=lambda scope: RecurrentGridConvNet(
            bidirectional=True,
            layers=[
                dict(filters=128, kernel_size=4, strides=3),
                dict(filters=128, kernel_size=4, strides=2),
                dict(filters=128, kernel_size=4, strides=2),
                dict(filters=128, kernel_size=1, strides=1),
                dict(filters=128, kernel_size=1, strides=1),
                dict(filters=128, kernel_size=1, strides=1),
            ],
            build_cell=lambda n_hidden, scope: CompositeCell(
                tf.contrib.rnn.GRUBlockCellV2(n_hidden),
                MLP([n_hidden], scope="GRU"), n_hidden),
            scope=scope,
        ),
        build_feature_fuser=lambda scope: ConvNet(
            scope=scope, layers=[
                dict(filters=None, kernel_size=3, strides=1, padding="SAME"),
                dict(filters=None, kernel_size=3, strides=1, padding="SAME"),
            ],
        ),
        build_obj_feature_extractor=lambda scope: ConvNet(
            scope=scope, layers=[
                dict(filters=None, kernel_size=3, strides=1, padding="SAME"),
                dict(filters=None, kernel_size=3, strides=1, padding="SAME"),
            ],
        ),

        build_lateral=lambda scope: MLP([100, 100], scope=scope),
        build_object_encoder=lambda scope: MLP([256, 128], scope=scope),
        build_object_decoder=lambda scope: MLP([128, 256], scope=scope),

        n_backbone_features=100,
        n_passthrough_features=100,

        n_lookback=1,

        use_concrete_kl=False,
        obj_concrete_temp=1.0,
        obj_temp=1.0,
        object_shape=(14, 14),
        A=50,

        # TODO: see if this helps / is necessary
        batch_size=4,

        min_hw=0.0,
        max_hw=1.0,

        min_yx=-0.5,
        max_yx=1.5,

        yx_prior_mean=0.0,
        yx_prior_std=1.0,

        attr_prior_mean=0.0,
        attr_prior_std=1.0,
        z_prior_mean=0.0,
        z_prior_std=1.0,

        obj_logit_scale=2.0,
        alpha_logit_scale=0.1,
        alpha_logit_bias=5.0,

        training_wheels="Exp(1.0, 0.0, decay_rate=0.0, decay_steps=1000, staircase=True)",
        count_prior_dist=None,
        noise_schedule="Exp(0.001, 0.0, 1000, 0.1)",

        # Found through hyper parameter search
        hw_prior_mean=np.log(0.1/0.9),
        hw_prior_std=0.5,
        count_prior_decay_steps=1000,
        final_count_prior_log_odds=0.0125,
    ),
)

alg_configs["indep_sspair"] = alg_configs["sspair"].copy(
    build_obj_feature_extractor=None,
)

alg_configs["sspair_test"] = alg_configs["sspair"].copy(
    render_step=5,
    eval_step=5,
    display_step=5,
    batch_size=4,
    shuffle_buffer_size=32,
    n_train=1000,
    n_val=16,
    A=16,
    n_frames=2,
    n_backbone_features=16,
    n_passthrough_features=16,
    build_backbone=lambda scope: RecurrentGridConvNet(
        bidirectional=True,
        layers=[
            dict(filters=16, kernel_size=4, strides=3),
            dict(filters=16, kernel_size=4, strides=2),
            dict(filters=16, kernel_size=4, strides=2),
            dict(filters=16, kernel_size=1, strides=1),
            dict(filters=16, kernel_size=1, strides=1),
            dict(filters=16, kernel_size=1, strides=1),
        ],
        build_cell=lambda n_hidden, scope: CompositeCell(
            tf.contrib.rnn.GRUBlockCellV2(n_hidden),
            MLP([n_hidden], scope="GRU"), n_hidden),
        scope=scope,
    ),
    build_feature_fuser=lambda scope: ConvNet(
        scope=scope, layers=[
            dict(filters=None, kernel_size=3, strides=1, padding="SAME"),
            dict(filters=None, kernel_size=3, strides=1, padding="SAME"),
        ],
    ),
    build_obj_feature_extractor=lambda scope: ConvNet(
        scope=scope, layers=[
            dict(filters=None, kernel_size=3, strides=1, padding="SAME"),
            dict(filters=None, kernel_size=3, strides=1, padding="SAME"),
        ],
    ),

    build_lateral=lambda scope: MLP([16, 16], scope=scope),
    build_object_encoder=lambda scope: MLP([64, 64], scope=scope),
    build_object_decoder=lambda scope: MLP([64, 64], scope=scope),
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
