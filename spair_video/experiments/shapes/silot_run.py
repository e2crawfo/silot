from dps.hyper import run_experiment
from dps.utils import copy_update
from dps.updater import DummyUpdater
from spair_video.run import basic_config, alg_configs, env_configs

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--max-shapes', type=int, choices=[10, 20, 30], required=True)
parser.add_argument('--small', action='store_true')
args, _ = parser.parse_known_args()


readme = "Running SILOT experiment on shapes."

run_kwargs = dict(
    max_hosts=1, ppn=4, cpp=2, gpu_set="0,1", pmem=15000, project="rpp-bengioy",
    wall_time="96hours", cleanup_time="5mins", slack_time="5mins", n_repeats=4,
    copy_locally=True,
)

durations = dict(
    long=copy_update(run_kwargs),
    short=dict(
        wall_time="60mins", gpu_set="0", ppn=2, n_repeats=2, distributions=None,
        config=dict(
            max_steps=3000, render_step=500, eval_step=500, display_step=100,
            stage_steps=600, curriculum=[dict()]),
    ),
    build=dict(
        ppn=1, cpp=3, gpu_set="0", wall_time="300mins", n_repeats=1, distributions=None,
        config=dict(
            do_train=False, get_updater=DummyUpdater, render_hook=None,
            curriculum=[dict()],
        )
    ),
)

config = basic_config.copy()
if args.small:
    config.update(env_configs['big_shapes_small'])
    config.n_prop_objects = 10
else:
    config.update(env_configs['big_shapes'])
    config.n_prop_objects = 30

config.update(alg_configs['shapes_silot'])

if args.small:
    config.n_prop_objects = 10
    config.batch_size = 16
else:
    config.n_prop_objects = 30
    config.batch_size = 8

config.update(
    min_shapes=args.max_shapes-9, max_shapes=args.max_shapes,
    stage_steps=40000, render_step=1000000,
    final_count_prior_log_odds=0.0125,
    # final_count_prior_log_odds=2.5,
    curriculum=[
        dict(),
        dict(
            patience_start=1,
            lr_schedule=1. / 3 * 1e-4,
            initial_n_frames=8,
            initial_count_prior_log_odds=0.0125,
            # initial_count_prior_log_odds=2.5,
            end_training_wheels=1,
            noise_schedule=0.0,
        ),
        dict(
            patience_start=1,
            lr_schedule=1. / 9 * 1e-4,
            initial_n_frames=8,
            initial_count_prior_log_odds=0.0125,
            # initial_count_prior_log_odds=2.5,
            end_training_wheels=1,
            noise_schedule=0.0,
        ),
    ],
)

run_experiment(
    "shapes_silot", config, "silot on shapes.", name_variables="max_shapes", durations=durations
)
