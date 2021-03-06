import numpy as np
import tensorflow as tf
import sonnet as snt
import itertools

from dps import cfg
from dps.hyper import run_experiment
from dps.utils import Config, copy_update, NumpySeed, get_default_config
from dps.datasets.base import VisualArithmeticDataset, LongVideoVisualArithmetic, Environment
from dps.datasets.shapes import RandomShapesDataset, LongVideoRandomShapes
from dps.datasets.atari import AtariVideoDataset, AtariLongVideoVideoDataset, count_episodes
from dps.utils.tf import (
    MLP, CompositeCell, GridConvNet, TransposeGridConvNet, ConvNet,
    SpatialBroadcastDecoder, tf_shape, LookupSchedule
)

from auto_yolo.models.core import Updater
from auto_yolo.models.obj_kl import SimpleObjKL, SequentialObjKL

from silot.core import SimpleVideoVAE, SimpleVAE_RenderHook, BackgroundExtractor
from silot.tba_model import TrackingByAnimation, TBA_Backbone, TBA_RenderHook
from silot.sqair_model import SQAIR, SQAIRUpdater, SQAIR_RenderHook
from silot.sspair_model import SequentialSpair, SequentialSpair_RenderHook
from silot.silot_model import (
    SILOT, SILOT_RenderHook, SimpleSILOT_RenderHook, PaperSILOT_RenderHook, LongVideoSILOT_RenderHook)
from silot.baseline_model import BaselineTracker, Baseline_RenderHook, BaselineUpdater
from silot.background_only import BackgroundOnly, BackgroundOnly_RenderHook
from silot.background import ScalorBackground


class MovingMNIST(Environment):
    def __init__(self):
        train_seed, val_seed, test_seed = 0, 1, 2

        train = VisualArithmeticDataset(
            n_examples=int(cfg.n_train), shuffle=True,
            example_range=cfg.train_example_range, seed=train_seed)

        val = VisualArithmeticDataset(
            n_examples=int(cfg.n_val), shuffle=True,
            example_range=cfg.val_example_range, seed=val_seed)

        test = VisualArithmeticDataset(
            n_examples=int(cfg.n_val), shuffle=True,
            example_range=cfg.test_example_range, seed=test_seed)

        self.datasets = dict(train=train, val=val, test=test)


class MovingMNISTLongVideoEnv(Environment):
    def __init__(self):
        train_seed, val_seed, test_seed = 0, 1, 2

        train = LongVideoVisualArithmetic(
            shuffle=False, example_range=cfg.train_example_range, seed=train_seed)

        val = LongVideoVisualArithmetic(
            shuffle=False, example_range=cfg.val_example_range, seed=val_seed)

        test = LongVideoVisualArithmetic(
            shuffle=False, example_range=cfg.test_example_range, seed=test_seed)

        self.datasets = dict(train=train, val=val, test=test)


class MovingShapes(Environment):
    def __init__(self):
        train_seed, val_seed, test_seed = 0, 1, 2

        train = RandomShapesDataset(
            n_examples=int(cfg.n_train), shuffle=True, seed=train_seed)

        val = RandomShapesDataset(
            n_examples=int(cfg.n_val), shuffle=True, seed=val_seed)

        test = RandomShapesDataset(
            n_examples=int(cfg.n_val), shuffle=True, seed=test_seed)

        self.datasets = dict(train=train, val=val, test=test)


class MovingShapesLongVideoEnv(Environment):
    def __init__(self):
        train_seed, val_seed, test_seed = 0, 1, 2

        train = LongVideoRandomShapes(shuffle=False, seed=train_seed)
        val = LongVideoRandomShapes(shuffle=False, seed=val_seed)
        test = LongVideoRandomShapes(shuffle=False, seed=test_seed)

        self.datasets = dict(train=train, val=val, test=test)


class AtariEnv(Environment):
    def __init__(self):
        train_seed, val_seed, test_seed = 0, 1, 2
        perm_seed = 4

        n_episodes = count_episodes(cfg.atari_game, cfg.after_warp)

        with NumpySeed(perm_seed):
            perm = np.random.permutation(n_episodes)

        n_val_episodes = max(1, int(cfg.val_fraction*n_episodes))
        train_end = n_episodes - 2*n_val_episodes
        val_end = n_episodes - n_val_episodes

        if cfg.do_train:
            train_episode_range = (None, train_end)
        else:
            train_episode_range = (None, 1)
        val_episode_range = (train_end, val_end)
        test_episode_range = (val_end, None)

        train_episode_indices = perm[slice(*train_episode_range)]
        val_episode_indices = perm[slice(*val_episode_range)]
        test_episode_indices = perm[slice(*test_episode_range)]

        get_annotations = not cfg.do_train

        train = AtariVideoDataset(
            max_examples=int(cfg.n_train), seed=train_seed,
            episode_indices=train_episode_indices, get_annotations=False,
            sample_density=cfg.train_sample_density)

        val = AtariVideoDataset(
            max_examples=int(cfg.n_val), seed=val_seed,
            episode_indices=val_episode_indices, get_annotations=get_annotations,
            sample_density=cfg.val_sample_density)

        test = AtariVideoDataset(
            max_examples=int(cfg.n_val), seed=test_seed,
            episode_indices=test_episode_indices, get_annotations=get_annotations,
            sample_density=cfg.val_sample_density)

        self.datasets = dict(train=train, val=val, test=test)


class AtariLongVideoEnv(Environment):
    def __init__(self):
        train_seed, val_seed, test_seed = 0, 1, 2
        perm_seed = 4

        n_episodes = count_episodes(cfg.atari_game, cfg.after_warp)

        with NumpySeed(perm_seed):
            perm = np.random.permutation(n_episodes)

        n_val_episodes = max(1, int(cfg.val_fraction*n_episodes))
        train_end = n_val_episodes
        val_end = n_episodes - n_val_episodes

        if cfg.do_train:
            train_episode_range = (None, train_end)
        else:
            train_episode_range = (None, 5)

        val_episode_range = (train_end, val_end)
        test_episode_range = (val_end, None)

        train_episode_indices = perm[slice(*train_episode_range)]
        val_episode_indices = perm[slice(*val_episode_range)]
        test_episode_indices = perm[slice(*test_episode_range)]

        self.dataset = dict(
            train=AtariLongVideoVideoDataset(seed=train_seed, episode_indices=train_episode_indices,),
            val=AtariLongVideoVideoDataset(seed=val_seed, episode_indices=val_episode_indices,),
            test=AtariLongVideoVideoDataset(seed=test_seed, episode_indices=test_episode_indices,),
        )


DEFAULT_CONFIG = get_default_config()


basic_config = DEFAULT_CONFIG.copy(
    scratch_dir='./dps_data',

    use_gpu=True,
    gpu_allow_growth=True,
    stopping_criteria="loss,min",
    threshold=-np.inf,
    max_experiments=None,
    preserve_env=False,
    load_path="-1",
    start_tensorboard=10,
    render_final=False,
    render_first=False,
    overwrite_plots=False,

    curriculum=[dict()],

    batch_size=16,
    shuffle_buffer_size=1000,
    prefetch_buffer_size_in_batches=100,
    prefetch_to_device=True,
    patience=0,

    n_train=60000,
    n_val=128,  # has to be a multiple of the batch size for sqair

    lr_schedule=1e-4,
    optimizer_spec="adam",
    max_grad_norm=10.0,
    grad_n_record_groups=0,
    eval_step=5000,
    display_step=0,
    render_step=0,
    max_steps=np.inf,

    stage_steps=50000,
    initial_n_frames=2,
    n_frames_scale=2,

    noisy=True,
    eval_noisy=True,
    train_reconstruction=True,
    train_kl=True,
    reconstruction_weight=1.0,
    kl_weight=1.0,

    get_updater=Updater,
    fixed_weights="",
    fixed_values={},
    no_gradient="",

    annotation_scheme="correct",
    warning_mode="ignore",
    simple_obj_kl=False,

    obj_threshold=0.5,
)


env_configs = dict()


env_configs['moving_mnist'] = Config(
    build_env=MovingMNIST,

    n_patch_examples=0,
    image_shape=(48, 48),
    tile_shape=(48, 48),
    patch_shape=(14, 14),
    object_shape=(14, 14),
    n_objects=12,
    min_digits=1,
    max_digits=12,
    max_overlap=14**2/2,
    n_classes=82,
    largest_digit=81,
    one_hot=True,
    reductions="sum",
    characters=list(range(10)),
    patch_size_std=0.0,
    colours="white",
    n_distractors_per_image=0,

    train_example_range=(0.0, 0.8),
    val_example_range=(0.8, 0.9),
    test_example_range=(0.9, 1.0),
    digits=list(range(10)),
    n_frames=8,

    backgrounds="",
    background_colours="",
    background_cfg=dict(mode="colour", colour="black"),
    postprocessing="",
    patch_speed=2,
    bounce_patches=True,

    appearance_prob=1.0,
    disappearance_prob=0.0,
)

env_configs['moving_mnist_sub'] = env_configs['moving_mnist'].copy(
    postprocessing="random",
    tile_shape=(48, 48),
    image_shape=(96, 96),
    n_objects=8*4,
    min_digits=8*4,
    max_digits=8*4,
)

env_configs['moving_mnist_gen'] = env_configs['moving_mnist_sub'].copy(
    postprocessing="",
    n_train=4,
    initial_n_frames=8,
    render_first=True,
)

env_configs['moving_mnist_big'] = env_configs['moving_mnist'].copy(
    image_shape=(96, 96),
    tile_shape=(48, 48),
    n_objects=4*8,
    min_digits=4*8,
    max_digits=4*8,
)

env_configs["mnist_learned_background"] = env_configs["moving_mnist"].copy(
    build_background_encoder=lambda scope: BackgroundExtractor(
        scope=scope,
        build_cell=lambda n_hidden, scope: tf.contrib.rnn.GRUBlockCellV2(n_hidden, name=scope),
        layers=[
            dict(filters=8, kernel_size=4, strides=3),
            dict(filters=8, kernel_size=4, strides=2),
            dict(filters=8, kernel_size=4, strides=2),
        ],
    ),
    build_background_decoder=lambda scope: ConvNet(
        scope=scope,
        layers=[
            dict(filters=8, kernel_size=4, strides=2, transpose=True,),
            dict(filters=8, kernel_size=4, strides=2, transpose=True,),
            dict(filters=8, kernel_size=4, strides=3, transpose=True,),
        ],
    ),
    background_cfg=dict(mode="learn_and_transform", A=8, bg_shape=(60, 60))
)

env_configs["sqair_mnist"] = env_configs["moving_mnist"].copy(
    patch_shape=(21, 21),
    image_shape=(50, 50),
    tile_shape=(50, 50),
    colours="",
)

# --- SHAPES ---

env_configs["shapes"] = Config(
    build_env=MovingShapes,

    image_shape=(48, 48),
    tile_shape=(48, 48),
    patch_shape=(14, 14),
    object_shape=(14, 14),
    min_shapes=1,
    max_shapes=8,
    n_objects=8,

    max_overlap=14**2/2,
    one_hot=True,
    colours="red green blue cyan magenta yellow",
    shapes="circle diamond star x plus",
    n_distractors_per_image=0,

    n_frames=8,
    backgrounds="",
    background_colours="",
    background_cfg=dict(mode="colour", colour="black"),
    postprocessing="",
    patch_size_std=0.1,
    patch_speed=5,
)

env_configs["big_shapes"] = env_configs["shapes"].copy(
    image_shape=(96, 96),
    tile_shape=(48, 48),
    anchor_box=(48, 48),
    min_shapes=10,
    max_shapes=20,
    n_objects=30,
)

env_configs["big_shapes_small"] = env_configs["big_shapes"].copy(
    postprocessing="random",
    n_samples_per_image=4,
    tile_shape=(60, 60),
    anchor_box=(48, 48),
)

env_configs['big_shapes_gen'] = env_configs['big_shapes'].copy(
    n_train=4,
    initial_n_frames=8,
    render_first=True,
    n_val=1000,
)

# --- ATARI ---

# For eval config, just set postprocessing=""
env_configs['atari'] = atari_train_config = Config(
    build_env=AtariEnv,
    atari_game='',
    atari_data_dir='./atari_rollouts/atari_rollouts',

    background_cfg=dict(mode="colour", colour="black"),

    image_shape=None,
    anchor_box=(30, 30),
    tile_shape=(72, 72),
    postprocessing="random",
    object_shape=(21, 21),

    n_frames=8,
    after_warp=False,
    max_episodes=None,
    train_sample_density=0.5,
    val_sample_density=0.05,
    n_samples_per_image=8,
    frame_skip=1,
    n_dilate=1,
    n_erode=0,
    filter_size=2,
    allowed_colors_for_annotations=None,
    distance_threshold=10000,
    distance_ord=2,
    average_frames=False,
    crop=None,
    val_fraction=0.05,
)

env_configs["space_invaders"] = atari_train_config.copy(
    atari_game="SpaceInvaders",
    n_episodes=37,
    crop=(0, 195, 0, 160),

    anchor_box=(24, 24),
)

env_configs["asteroids"] = atari_train_config.copy(
    atari_game="Asteroids",
    n_episodes=80,
    crop=None,
)

env_configs["carnival"] = atari_train_config.copy(
    atari_game="Carnival",
    n_episodes=50,
    crop=(30, 215, 0, 160),
)

env_configs["wizard_of_wor"] = atari_train_config.copy(
    n_episodes=24,
    atari_game="WizardOfWor",
)

# --- ALGS ---


def spair_prepare_func():
    from dps import cfg

    if not hasattr(cfg, 'anchor_box'):
        cfg.anchor_box = cfg.tile_shape

    cfg.count_prior_log_odds = (
        "Exp(start={}, end={}, decay_rate=0.1, decay_steps={}, log=True)".format(
            cfg.initial_count_prior_log_odds,
            cfg.final_count_prior_log_odds, cfg.count_prior_decay_steps)
    )
    cfg.training_wheels = "Exp(1.0, 0.0, decay_rate=0.0, decay_steps={}, staircase=True)".format(cfg.end_training_wheels)


alg_configs = dict()

alg_configs['simple_vae'] = Config(
    build_network=SimpleVideoVAE,

    attr_prior_mean=0.,
    attr_prior_std=1.0,

    A=128,

    train_reconstruction=True,
    reconstruction_weight=1.0,

    train_kl=True,
    kl_weight=1.0,

    build_encoder=lambda scope: MLP(n_units=[128, 128, 128], scope=scope),
    build_decoder=lambda scope: MLP(n_units=[128, 128, 128], scope=scope),
    build_cell=lambda scope: CompositeCell(
        tf.contrib.rnn.GRUBlockCellV2(128),
        MLP(n_units=[128], scope="GRU"), 2*128),
    render_hook=SimpleVAE_RenderHook(),

    flat_latent=True,
)

alg_configs['simple_vae_conv'] = alg_configs['simple_vae'].copy(
    build_encoder=lambda scope: GridConvNet(
        layers=[
            dict(filters=64, kernel_size=4, strides=1),
            dict(filters=64, kernel_size=4, strides=2),
            dict(filters=64, kernel_size=4, strides=1),
            dict(filters=64, kernel_size=4, strides=2),
            dict(filters=64, kernel_size=4, strides=1),
            dict(filters=64, kernel_size=4, strides=2),
            dict(filters=64, kernel_size=4, strides=1),
        ],
        scope=scope,
    ),
    build_decoder=lambda scope: TransposeGridConvNet(
        layers=[
            dict(filters=64, kernel_size=3, strides=1),
            dict(filters=64, kernel_size=3, strides=2),
            dict(filters=64, kernel_size=3, strides=1),
            dict(filters=64, kernel_size=3, strides=2),
            dict(filters=64, kernel_size=3, strides=1),
            dict(filters=64, kernel_size=3, strides=2),
            dict(filters=64, kernel_size=3, strides=1),
        ],
        scope=scope,
    ),
    A=64,
    initial_n_frames=1,
    n_frames=1,
    max_digits=3,
    build_cell=None,
)

alg_configs['simple_vae_sbd'] = alg_configs['simple_vae'].copy(
    build_encoder=lambda scope: ConvNet(
        layers=[
            dict(filters=64, kernel_size=4, strides=1),
            dict(filters=64, kernel_size=4, strides=2),
            dict(filters=64, kernel_size=4, strides=1),
            dict(filters=64, kernel_size=4, strides=2),
            dict(kind='fc', n_units=128),
            dict(kind='fc', n_units=128),
            dict(kind='fc', n_units=128, nl='linear'),
        ],
        scope=scope,
    ),
    build_decoder=lambda scope: SpatialBroadcastDecoder(
        layers=[
            dict(filters=128, kernel_size=1, strides=1, padding='SAME'),
            dict(filters=128, kernel_size=1, strides=1, padding='SAME'),
            dict(filters=128, kernel_size=1, strides=1, padding='SAME'),
            dict(filters=3, kernel_size=1, strides=1, padding='SAME'),
        ],
        scope=scope,
    ),
    initial_n_frames=1,
    n_frames=1,
    max_digits=3,
)

# --- SSPAIR --- (just applying SPAIR separately to each frame)

alg_configs['sspair'] = Config(
    build_network=SequentialSpair,
    render_hook=SequentialSpair_RenderHook(),
    prepare_func=spair_prepare_func,
    n_objects_per_cell=1,

    # stopping_criteria="MOT:mota,max",
    # stopping_criteria="AP,max",
    # threshold=np.inf,

    build_obj_kl=SequentialObjKL,
    SequentialObjKL=dict(
        use_concrete_kl=False,
        count_prior_dist=None,
    ),
    SimpleObjKL=dict(
        exp_rate=0.0125,
    ),

    build_backbone=lambda scope: GridConvNet(
        layers=[
            dict(filters=128, kernel_size=4, strides=3),
            dict(filters=128, kernel_size=4, strides=2),
            dict(filters=128, kernel_size=4, strides=2),
            dict(filters=128, kernel_size=1, strides=1),
            dict(filters=128, kernel_size=1, strides=1),
            dict(filters=128, kernel_size=1, strides=1),
        ],
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

    build_lateral=lambda scope: MLP(n_units=[100, 100], scope=scope),
    build_object_encoder=lambda scope: MLP(n_units=[256, 128], scope=scope),
    build_object_decoder=lambda scope: MLP(n_units=[128, 256], scope=scope),

    n_backbone_features=64,
    n_passthrough_features=64,

    n_lookback=1,

    use_concrete_kl=False,
    obj_concrete_temp=1.0,
    obj_temp=1.0,
    importance_temp=.25,
    object_shape=(14, 14),
    A=64,

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

    color_logit_scale=2.0,

    # values we were using previously, resulted in large bounding boxes
    # alpha_logit_scale=1.0,
    # alpha_logit_bias=0.0,

    # values used in SPAIR
    alpha_logit_scale=0.1,
    alpha_logit_bias=5.0,

    end_training_wheels=1000,
    count_prior_dist=None,
    noise_schedule="Exp(0.001, 0.0, 1000, 0.1)",

    # Found through hyper parameter search
    hw_prior_mean=float(np.log(0.1/0.9)),
    hw_prior_std=0.5,
    count_prior_decay_steps=1000,
    initial_count_prior_log_odds=1e6,  # log_odds: 13.815, sigmoid: .999
    final_count_prior_log_odds=0.0125,  # log_odds: -4.38, sigmoid: 0.012
)

alg_configs["test_sspair"] = alg_configs["sspair"].copy(
    n_train=1000,
    n_val=16,
    A=32,
    n_frames=2,
    n_backbone_features=32,
    n_passthrough_features=32,
    build_backbone=lambda scope: GridConvNet(
        bidirectional=True,
        layers=[
            dict(filters=32, kernel_size=4, strides=3),
            dict(filters=32, kernel_size=4, strides=2),
            dict(filters=32, kernel_size=4, strides=2),
            dict(filters=32, kernel_size=1, strides=1),
            dict(filters=32, kernel_size=1, strides=1),
            dict(filters=32, kernel_size=1, strides=1),
        ],
        build_cell=lambda n_hidden, scope: CompositeCell(
            tf.contrib.rnn.GRUBlockCellV2(n_hidden),
            MLP(n_units=[n_hidden], scope="GRU"), n_hidden),
        scope=scope,
    ),

    build_lateral=lambda scope: MLP(n_units=[32, 32], scope=scope),
    build_object_encoder=lambda scope: MLP(n_units=[64, 64], scope=scope),
    build_object_decoder=lambda scope: MLP(n_units=[64, 64], scope=scope),
)

# --- BG ONLY ---

alg_configs["background_only"] = dict(
    build_network=BackgroundOnly,

    train_reconstruction=True,
    reconstruction_weight=1.0,

    train_kl=True,
    kl_weight=1.0,

    render_hook=BackgroundOnly_RenderHook(N=16),

    attr_prior_mean=1.0,
    attr_prior_std=1.0,
    noisy=True,
    stage_steps=None,
    initial_n_frames=8,
    n_frames_scale=1,
)


# --- SILOT ---

alg_configs["silot"] = alg_configs["sspair"].copy(
    render_hook=SILOT_RenderHook(),
    plot_prior=True,
    mot_metrics=True,

    stopping_criteria="mota_post_prior_sum,max",
    threshold=np.inf,

    prior_start_step=-1,
    eval_prior_start_step=3,
    learn_prior=True,

    build_discovery_feature_fuser=lambda scope: ConvNet(
        scope=scope, layers=[
            dict(filters=None, kernel_size=3, strides=1, padding="SAME"),
            dict(filters=None, kernel_size=3, strides=1, padding="SAME"),
        ],
    ),
    build_prop_cell=snt.GRU,
    build_network=SILOT,
    build_mlp=lambda scope: MLP(n_units=[64, 64], scope=scope),
    build_background_model=None,

    n_prop_objects=16,
    n_hidden=64,
    kernel_std=0.1,

    d_yx_prior_mean=0.0,
    d_yx_prior_std=0.3,
    d_attr_prior_mean=0.0,
    d_attr_prior_std=0.4,
    d_z_prior_mean=0.0,
    d_z_prior_std=1.0,

    disc_dropout_prob=0.5,

    learn_glimpse_prime=False,
    glimpse_prime_scale=2.0,

    initial_n_frames=2,
    n_frames_scale=2,

    where_t_scale=1.0,

    do_lateral=False,

    conv_discovery=True,
    gate_d_attr=False,
    independent_prop=True,
    use_sqair_prop=True,
    use_abs_posn=True,
    edge_resampler=False,

    patience=30000,
    patience_start=160000,
    curriculum=[
        dict(),
        dict(
            patience_start=1,
            lr_schedule=1. / 3 * 1e-4,
            initial_n_frames=8,
            count_prior_log_odds=0.0125,
            training_wheels=0.0,
            noise_schedule=0.0,
        ),
        dict(
            patience_start=1,
            lr_schedule=1. / 9 * 1e-4,
            initial_n_frames=8,
            count_prior_log_odds=0.0125,
            training_wheels=0.0,
            noise_schedule=0.0,
        ),
    ],

    build_conv_lateral=lambda scope: ConvNet(
        scope=scope, layers=[
            dict(filters=None, kernel_size=1, strides=1, padding="SAME"),
            dict(filters=None, kernel_size=1, strides=1, padding="SAME"),
        ],
    ),
)

alg_configs["silot_bg"] = alg_configs["silot"].copy(
    build_background_model=ScalorBackground,

    ScalorBackground=dict(

        build_background_encoder=lambda scope: ConvNet(
            layers=[
                dict(filters=32, kernel_size=4, strides=3),
                dict(filters=64, kernel_size=4, strides=2),
                dict(filters=128, kernel_size=4, strides=2),
                dict(kind='fc', n_units=128),
                dict(kind='fc', n_units=128),
                dict(kind='fc', n_units=128, nl='linear'),
            ],
            scope=scope,
        ),

        build_background_decoder=lambda scope: SpatialBroadcastDecoder(
            layers=[
                dict(filters=64, kernel_size=1, strides=1, padding='SAME'),
                dict(filters=64, kernel_size=1, strides=1, padding='SAME'),
                dict(filters=64, kernel_size=1, strides=1, padding='SAME'),
            ],
            scope=scope,
        ),
    ),

)

alg_configs["silot_plot"] = alg_configs["silot"].copy(
    render_hook=PaperSILOT_RenderHook(N=16),
    n_train=32,
    do_train=False,
    n_frames=8,
    initial_n_frames=8,
    eval_noisy=False,
    curriculum=[dict()],
    render_first=True,
)

alg_configs["mnist_long_video_silot"] = alg_configs["silot_plot"].copy(
    render_hook=LongVideoSILOT_RenderHook(),
    build_env=MovingMNISTLongVideoEnv,
    n_examples=10,
    batch_size=10,
    n_frames=100,
    n_batches=1,
    # There is something not quite working when I'm using n_batches > 1...
    # but luckily I am able to run 100 frames without breaking it up at all.
    # n_frames=100,
    # n_batches=20,
    shuffle_val=False,
    obj_threshold=0.5,
)

alg_configs["shapes_silot"] = alg_configs["silot"].copy(
    color_logit_scale=1.0,
    alpha_logit_scale=1.0,
    alpha_logit_bias=3.0,
    final_count_prior_log_odds=0.0125,
    independent_prop=False,
    kernel_std=0.15,
    eval_noisy=False,
)

alg_configs["shapes_eval_silot"] = alg_configs["shapes_silot"].copy(
    postprocessing="",
    n_train=32,
    do_train=False,
    eval_noisy=False,
    obj_threshold=0.05,
    curriculum=[dict()],
    n_prop_objects=36,
    render_first=True,
    n_frames=8,
    initial_n_frames=8,
)

alg_configs["shapes_plot_silot"] = alg_configs["shapes_eval_silot"].copy(
    render_hook=PaperSILOT_RenderHook(N=16),
    render_first=True,
)

alg_configs["shapes_long_video_silot"] = alg_configs["shapes_eval_silot"].copy(
    render_hook=LongVideoSILOT_RenderHook(),
    build_env=MovingShapesLongVideoEnv,
    n_examples=10,
    batch_size=10,
    n_frames=100,
    n_batches=1,
    shuffle_val=False,
    obj_threshold=0.05,
)

alg_configs["atari_train_silot"] = alg_configs["silot"].copy(
    stopping_criteria="loss_reconstruction,min",
    threshold=-np.inf,
    stage_steps=20000,
    patience_start=100000,
    patience=20000,
    render_first=False,
    plot_prior=False,
    final_count_prior_log_odds=0.0125,
    n_prop_objects=36,
    eval_noisy=False,
    obj_threshold=0.05,
    build_object_encoder=lambda scope: MLP(n_units=[512, 256], scope=scope),
    build_object_decoder=lambda scope: MLP(n_units=[256, 512], scope=scope),
)

alg_configs["atari_eval_silot"] = alg_configs["atari_train_silot"].copy(
    render_hook=None,
    postprocessing="",
    do_train=False,
    curriculum=[dict()],
    n_prop_objects=128,
    n_frames=8,
    batch_size=8,
    plot_prior=True,
    render_first=False,
    render_final=False,
)

alg_configs["atari_plot_silot"] = alg_configs["atari_eval_silot"].copy(
    render_hook=SimpleSILOT_RenderHook(),
    render_first=True,
    n_frames=16,
    batch_size=16,
)

alg_configs["atari_long_video_silot"] = alg_configs["atari_plot_silot"].copy(
    render_hook=LongVideoSILOT_RenderHook(),
    build_env=AtariLongVideoEnv,
    batch_size=5,
    n_frames=4,
    n_batches=20,
    n_samples_per_episode=1,
    shuffle_val=False,
)


def silot_mnist_eval_prepare_func():
    from dps import cfg
    spair_prepare_func()
    if cfg.max_digits == 6:
        experiment_path = "/scratch/e2crawfo/dps_data/parallel_experiments_run/aaai_2020_silot/mnist/run/run_env=moving-mnist_max-digits=6_alg=conv-silot_duration=long_2019_07_08_04_05_50_seed=0/experiments"
    else:
        experiment_path = "/scratch/e2crawfo/dps_data/parallel_experiments_run/aaai_2020_silot/mnist/run/run_env=moving-mnist_max-digits=12_alg=conv-silot_duration=long_2019_07_08_04_06_03_seed=0/experiments"

    import os
    dirs = os.listdir(experiment_path)
    my_dir = sorted(dirs)[cfg.repeat]

    cfg.load_path = os.path.join(experiment_path, my_dir, 'weights/best_of_stage_2')


def silot_shapes_restart_prepare_func():
    from dps import cfg
    spair_prepare_func()

    # a string representation of a dict mapping from repeat (in original experiment) to restart step
    restart_steps = dict([
        (int(i) for i in s.split(':'))
        for s in cfg.restart_steps.split()
    ])

    my_repeat = sorted(restart_steps.keys())[cfg.repeat]
    my_restart_step = restart_steps[my_repeat]
    cfg.start_from = "0,{}".format(my_restart_step)

    experiment_path = cfg.experiment_restart_path
    import os
    dirs = os.listdir(experiment_path)
    my_dir = sorted(dirs)[my_repeat]
    assert my_dir.startswith('exp_idx=0_repeat={}_'.format(my_repeat))
    cfg.load_path = os.path.join(experiment_path, my_dir, 'weights/checkpoint_stage_0')


# --- SQAIR ---


def sqair_prepare_func():
    from dps import cfg
    cfg.patience_start = 4 * cfg.stage_steps


def sqair_mnist_eval_prepare_func():
    from dps import cfg
    sqair_prepare_func()

    if cfg.max_digits == 6:
        if cfg.alg_name == 'conv_sqair':
            experiment_path = "/scratch/e2crawfo/dps_data/parallel_experiments_run/aaai_2020_silot/mnist/run/run_env=moving-mnist_max-digits=6_alg=conv-sqair_duration=long_2019_07_17_11_35_58_seed=0/experiments"
        else:
            experiment_path = "/scratch/e2crawfo/dps_data/parallel_experiments_run/aaai_2020_silot/mnist/run/run_env=moving-mnist_max-digits=6_alg=sqair_duration=long_2019_07_17_11_35_26_seed=0/experiments"
    else:
        if cfg.alg_name == 'conv_sqair':
            experiment_path = "/scratch/e2crawfo/dps_data/parallel_experiments_run/aaai_2020_silot/mnist/run/run_env=moving-mnist_max-digits=12_alg=conv-sqair_duration=long_2019_07_17_11_36_13_seed=0/experiments"
        else:
            experiment_path = "/scratch/e2crawfo/dps_data/parallel_experiments_run/aaai_2020_silot/mnist/run/run_env=moving-mnist_max-digits=12_alg=sqair_duration=long_2019_07_17_11_35_42_seed=0/experiments"

    import os
    dirs = os.listdir(experiment_path)
    my_dir = sorted(dirs)[cfg.repeat]

    stage = 2
    while True:
        cfg.load_path = os.path.join(experiment_path, my_dir, 'weights/best_of_stage_{}'.format(stage))

        if os.path.exists(cfg.load_path + '.meta'):
            break

        stage -= 1

        if stage < 0:
            raise Exception("No valid weights found.")


alg_configs['sqair'] = Config(
    stopping_criteria="mota_post_prior_sum,max",
    threshold=np.inf,
    prepare_func=sqair_prepare_func,

    get_updater=SQAIRUpdater,
    build_network=SQAIR,
    render_hook=SQAIR_RenderHook(),
    debug=False,
    batch_size=16,
    disc_prior_type='geom',
    # disc_prior_type='cat',
    step_success_prob=0.75,  # Not used when disc_prior_type==cat

    # disc_step_bias=1.,
    disc_step_bias=5.,
    prop_step_bias=5.,
    prop_prior_step_bias=10.,
    prop_prior_type='rnn',

    masked_glimpse=True,
    k_particles=2,
    n_objects=12,
    sample_from_prior=False,
    rec_where_prior=True,
    scale_prior=(-2., -2.),
    scale_bounds=(0.0, 1.0),
    transform_var_bias=-3.,
    # scale_prior=(0., 0.),
    # scale_bounds=(0.125, 0.42),
    # transform_var_bias=0.,
    rnn_class=snt.VanillaRNN,
    time_rnn_class=snt.GRU,
    prior_rnn_class=snt.GRU,
    optimizer_spec="rmsprop,momentum=0.9",
    build_input_encoder=None,
    lr_schedule=1e-5,
    max_grad_norm=None,
    l2_schedule=0.0,
    n_layers=2,
    n_hidden=8*32,
    n_what=50,
    output_scale=0.25,
    output_std=0.3,
    variable_scope_depth=10,
    training_wheels=0.0,
    fixed_presence=False,

    fast_discovery=False,
    fast_propagation=False,

    patience=30000,
    stage_steps=20000,
    curriculum=[
        dict(),
        dict(
            patience_start=1,
            lr_schedule=1. / 3 * 1e-5,
            initial_n_frames=8,
        ),
        dict(
            patience_start=1,
            lr_schedule=1. / 9 * 1e-5,
            initial_n_frames=8,
        ),
    ],

    prior_start_step=-1,
    eval_prior_start_step=3,
)


alg_configs['fixed_sqair'] = alg_configs['sqair'].copy(
    fixed_presence=True,
    disc_prior_type='fixed',
)


class SQAIRWrapper:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __call__(self, inp):
        output = self.wrapped(inp, None, True)[0]
        batch_size = tf_shape(output)[0]
        n_trailing = np.prod(tf_shape(output)[1:])
        return tf.reshape(output, (batch_size, n_trailing))


alg_configs['conv_sqair'] = alg_configs['sqair'].copy(
    build_input_encoder=lambda: SQAIRWrapper(
        GridConvNet(
            layers=[
                dict(filters=128, kernel_size=4, strides=3),
                dict(filters=128, kernel_size=4, strides=2),
                dict(filters=128, kernel_size=4, strides=2),
                dict(filters=128, kernel_size=1, strides=1),
                dict(filters=128, kernel_size=1, strides=1),
                dict(filters=128, kernel_size=1, strides=1),
            ],
            scope='input_encoder',
        )
    ),
)

# --- TBA ---

alg_configs['tba_shapes'] = Config(
    stopping_criteria="mota_post_prior_sum,max",
    threshold=np.inf,

    build_network=TrackingByAnimation,

    build_backbone=TBA_Backbone,
    build_cell=snt.GRU,

    build_key_network=lambda scope: MLP(n_units=[], scope=scope),
    build_beta_network=lambda scope: MLP(n_units=[], scope=scope),
    build_write_network=lambda scope: MLP(n_units=[], scope=scope),
    build_erase_network=lambda scope: MLP(n_units=[], scope=scope),
    build_output_network=lambda scope: MLP(n_units=[80, 377], scope=scope),

    optimizer_spec="weight_decay_adam,decay=1e-6",
    lr=5e-4,
    max_grad_norm=5,

    # lmbda=13.0,
    lmbda=1.0,  # paper says 1, but 13 is used in the code for mnist and sprites
    # n_trackers=1,
    n_trackers=4,
    n_layers=3,

    # number of units in hidden states for each object is set to be 4 * (# of conv features at output)
    n_hidden=80,
    S=20,

    # in the code they use 0.1 (look at scripts/gen_sprite.py), but the paper says 0.2
    # might be equivalent in different spaces
    eta=(0.1, 0.1),

    prioritize=True,
    learn_initial_state=False,
    anchor_box=(21, 21),
    render_hook=TBA_RenderHook(),
    discrete_eval=False,
    fixed_mask=False,
    clamp_appearance=False,

    initial_n_frames=8,
)

alg_configs['tba_mnist'] = alg_configs['tba_shapes'].copy(
    n_layers=1,
    n_hidden=200,
    S=50,
    build_output_network=lambda scope: MLP(n_units=[200, 397], scope=scope),

    anchor_box=(14, 14),  # for original paper it's (28, 28), because the digits themselves are (28, 28)
    # anchor_box=(28, 28),
    eta=(0.5, 0.5),  # this is not a typo: they tell the network exactly what the size should be
    # TODO restore this
    # eta=(0.0, 0.0),  # this is not a typo: they tell the network exactly what the size should be
    fixed_mask=True,
    clamp_appearance=True,
)

# --- BASELINE ---


def baseline_prepare_func():
    from dps import cfg
    cfg.anchor_box = cfg.tile_shape


alg_configs['baseline'] = Config(
    get_updater=BaselineUpdater,
    build_network=BaselineTracker,
    render_hook=Baseline_RenderHook(N=16),
    prepare_func=baseline_prepare_func,
    stage_steps=None,
    initial_n_frames=8,
    stopping_criteria="MOT:mota,max",
    threshold=np.inf,

    no_gradient=True,

    render_step=0,
    eval_step=1,
    n_train=32,
    n_val=128
)

n_values = 50
cc_values = np.linspace(0.0001, 3.0001, n_values+2)

alg_configs['mnist_baseline'] = alg_configs['baseline'].copy(
    curriculum=[dict(min_digits=i, max_digits=i) for i in range(1, 13)],
    cc_threshold=LookupSchedule(cc_values),
    cosine_threshold=None,
    max_steps=len(cc_values),
)

test_cc_threshold_AP = [
    9.999999747378753e-05, 9.999999747378753e-05, 9.999999747378753e-05, 9.999999747378753e-05, 0.4706882238388062,
    0.5883352756500244, 0.5295117497444153, 0.6471588015556335, 0.7059823274612427, 0.8236294388771057,
    0.7059823274612427, 0.7648058533668518
]

test_cc_threshold_count_1norm = [
    9.999999747378753e-05, 0.411864697933197, 0.6471588015556335, 0.7648058533668518, 0.7059823274612427,
    1.0001000165939329, 1.0589234828948977, 1.0001000165939329, 1.1765705347061155, 1.2353941202163696,
    1.2942177057266235, 1.353041172027588,
]

test_cc_threshold_mota = [
    9.999999747378753e-05, 0.411864697933197, 0.4706882238388062, 0.5295117497444153, 0.5883352756500244,
    0.5883352756500244, 0.5295117497444153, 0.6471588015556335, 0.5883352756500244, 0.5883352756500244,
    0.6471588015556335, 0.6471588015556335,
]

alg_configs['mnist_baseline_test'] = alg_configs['mnist_baseline'].copy(
    do_train=False,
    cc_threshold=LookupSchedule(cc_values),
    cosine_threshold=None,
)

alg_configs['mnist_baseline_AP'] = alg_configs['mnist_baseline_test'].copy(
    curriculum=[dict(min_digits=i+1, max_digits=i+1, cc_threshold=cct) for i, cct in enumerate(test_cc_threshold_AP)]
)

alg_configs['mnist_baseline_count_1norm'] = alg_configs['mnist_baseline_test'].copy(
    curriculum=[dict(min_digits=i+1, max_digits=i+1, cc_threshold=cct) for i, cct in enumerate(test_cc_threshold_count_1norm)]
)

alg_configs['mnist_baseline_mota'] = alg_configs['mnist_baseline_test'].copy(
    curriculum=[dict(min_digits=i+1, max_digits=i+1, cc_threshold=cct) for i, cct in enumerate(test_cc_threshold_mota)]
)

n_values = 25
cosine_values = np.linspace(0.0001, 1.0001, n_values+2)
shapes_cc_values, cosine_values = zip(*itertools.product(cc_values, cosine_values))

shapes_baseline_n_shapes = [1, 5, 10, 15, 20, 25, 30, 35]

alg_configs['shapes_baseline'] = alg_configs['baseline'].copy(
    curriculum=[dict(min_shapes=i, max_shapes=i) for i in shapes_baseline_n_shapes],
    cc_threshold=LookupSchedule(shapes_cc_values),
    cosine_threshold=LookupSchedule(cosine_values),
    max_steps=len(shapes_cc_values),
)

shapes_baseline_values = {
    'AP': [{'cc_threshold': 9.999999747378753e-05,
            'cosine_threshold': 0.7308692336082458},
           {'cc_threshold': 0.058923527598381036,
            'cosine_threshold': 0.8847153782844543},
           {'cc_threshold': 9.999999747378753e-05,
            'cosine_threshold': 0.9616384506225586},
           {'cc_threshold': 9.999999747378753e-05,
            'cosine_threshold': 0.9616384506225586},
           {'cc_threshold': 9.999999747378753e-05,
            'cosine_threshold': 0.9616384506225586},
           {'cc_threshold': 0.058923527598381036,
            'cosine_threshold': 0.9616384506225586},
           {'cc_threshold': 0.11774706095457076,
            'cosine_threshold': 0.9616384506225586},
           {'cc_threshold': 9.999999747378753e-05,
            'cosine_threshold': 0.9616384506225586}
    ],
    'MOT:mota': [{'cc_threshold': 9.999999747378753e-05,
                  'cosine_threshold': 0.7308692336082458},
                 {'cc_threshold': 0.058923527598381036,
                  'cosine_threshold': 0.8847153782844543},
                 {'cc_threshold': 9.999999747378753e-05,
                  'cosine_threshold': 0.9616384506225586},
                 {'cc_threshold': 9.999999747378753e-05,
                  'cosine_threshold': 0.9616384506225586},
                 {'cc_threshold': 9.999999747378753e-05,
                  'cosine_threshold': 0.9616384506225586},
                 {'cc_threshold': 0.8236294388771057,
                  'cosine_threshold': 0.9616384506225586},
                 {'cc_threshold': 1.353041172027588,
                  'cosine_threshold': 0.9616384506225586},
                 {'cc_threshold': 9.999999747378753e-05,
                  'cosine_threshold': 1.0001000165939329}
    ],
    'count_1norm': [{'cc_threshold': 9.999999747378753e-05,
                     'cosine_threshold': 0.7308692336082458},
                    {'cc_threshold': 9.999999747378753e-05,
                     'cosine_threshold': 0.9616384506225586},
                    {'cc_threshold': 0.8236294388771057,
                     'cosine_threshold': 0.9616384506225586},
                    {'cc_threshold': 0.8236294388771057,
                     'cosine_threshold': 0.9616384506225586},
                    {'cc_threshold': 0.8236294388771057,
                     'cosine_threshold': 0.9616384506225586},
                    {'cc_threshold': 0.8236294388771057,
                     'cosine_threshold': 0.9616384506225586},
                    {'cc_threshold': 0.8236294388771057,
                     'cosine_threshold': 0.9616384506225586},
                    {'cc_threshold': 0.8236294388771057,
                     'cosine_threshold': 0.9616384506225586}
    ],
}

alg_configs['shapes_baseline_test'] = alg_configs['shapes_baseline'].copy(
    do_train=False,
    cc_threshold=None,
    cosine_threshold=None,
)

alg_configs['shapes_baseline_AP'] = alg_configs['shapes_baseline_test'].copy(
    curriculum=[
        copy_update(v, min_shapes=i, max_shapes=i)
        for i, v in zip(shapes_baseline_n_shapes, shapes_baseline_values['AP'])]
)

alg_configs['shapes_baseline_count_1norm'] = alg_configs['shapes_baseline_test'].copy(
    curriculum=[
        copy_update(v, min_shapes=i, max_shapes=i)
        for i, v in zip(shapes_baseline_n_shapes, shapes_baseline_values['count_1norm'])]
)

alg_configs['shapes_baseline_mota'] = alg_configs['shapes_baseline_test'].copy(
    curriculum=[
        copy_update(v, min_shapes=i, max_shapes=i)
        for i, v in zip(shapes_baseline_n_shapes, shapes_baseline_values['MOT:mota'])]
)


for k, v in env_configs.items():
    v['env_name'] = k
for k, v in alg_configs.items():
    v['alg_name'] = k


if __name__ == "__main__":
    config = basic_config.copy()
    run_experiment(
        "test_silot", config, "First test of silot.",
        alg_configs=alg_configs, env_configs=env_configs,
        cl_mode='strict')
