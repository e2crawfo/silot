from dps.hyper import run_experiment
from dps.utils import copy_update
from dps.updater import DummyUpdater
from spair_video.run import basic_config, alg_configs, env_configs, silot_shapes_restart_prepare_func

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--max-shapes', type=int, choices=[10, 20, 30], required=True)
parser.add_argument('--small', action='store_true')
args, _ = parser.parse_known_args()


readme = "Running SILOT experiment on shapes."

run_kwargs = dict(
    max_hosts=1, ppn=4, cpp=4, gpu_set="0,1", pmem=5000, project="rpp-bengioy",
    wall_time="96hours", cleanup_time="5mins", slack_time="5mins", n_repeats=4,
)

durations = dict(
    long=copy_update(run_kwargs),
    restart10=copy_update(
        run_kwargs, wall_time="75hours",
        cpp=4, pmem=5000, ppn=1, n_repeats=1, gpu_set="0",
        config=dict(
            seed=100,
            restart_steps="1:120000",
            experiment_restart_path="/scratch/e2crawfo/dps_data/parallel_experiments_run/aaai_2020_silot/shapes/run/run_env=big-shapes_max-shapes=10_alg=shapes-silot_duration=long_2019_07_30_16_55_21_seed=0/experiments",
            prepare_func=silot_shapes_restart_prepare_func,
        ),
    ),
    restart20=copy_update(
        run_kwargs, wall_time="75hours",
        cpp=4, pmem=5000, ppn=1, n_repeats=1, gpu_set="0",
        config=dict(
            seed=200,
            restart_steps="2:120000",
            experiment_restart_path="/scratch/e2crawfo/dps_data/parallel_experiments_run/aaai_2020_silot/shapes/run/run_env=big-shapes_max-shapes=20_alg=shapes-silot_duration=long_2019_08_01_07_44_22_seed=0/experiments",
            prepare_func=silot_shapes_restart_prepare_func,
        ),
    ),
    short=dict(
        wall_time="60mins", gpu_set="0", ppn=2, cpp=4, n_repeats=2, distributions=None, pmem=15000,
        config=dict(
            max_steps=3000, render_step=500, eval_step=500, display_step=100,
            stage_steps=500, curriculum=[dict()], backup_step=300,
        ),
    ),
    build=dict(
        ppn=1, cpp=3, gpu_set="0", wall_time="5hours", n_repeats=1, distributions=None,
        config=dict(
            do_train=False, get_updater=DummyUpdater, render_hook=None,
            curriculum=[dict()],
        )
    ),
)

config = basic_config.copy()
if args.small:
    config.update(env_configs['big_shapes_small'])
else:
    config.update(env_configs['big_shapes'])

config.update(alg_configs['shapes_silot'])

if args.small:
    config.batch_size = 16
    config.n_prop_objects = 25
else:
    config.batch_size = 8
    config.n_prop_objects = 36


config.update(
    min_shapes=args.max_shapes-9, max_shapes=args.max_shapes,
    stage_steps=40000, render_step=1000000,
)

run_experiment(
    "shapes_silot", config, "silot on shapes.", name_variables="max_shapes", durations=durations
)
