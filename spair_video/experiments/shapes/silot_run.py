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
    max_hosts=1, ppn=4, cpp=2, gpu_set="0,1", pmem=15000, project="rpp-bengioy",
    wall_time="96hours", cleanup_time="5mins", slack_time="5mins", n_repeats=4,
    copy_locally=True,
)

durations = dict(
    long=copy_update(run_kwargs),
    restart=copy_update(
        run_kwargs, wall_time="75hours", ppn=3, n_repeats=4,
        config=dict(
            restart_steps="0:135000 1:120000 2:135000 3:120000",
            experiment_restart_path="/scratch/e2crawfo/dps_data/parallel_experiments_run/shapes-silot/run_env=big-shapes_max-shapes=30_alg=shapes-silot_duration=long_2019_08_01_07_44_41_seed=0/experiments",
            prepare_func=silot_shapes_restart_prepare_func,
        ),
    ),
    short=dict(
        wall_time="180mins", gpu_set="0", ppn=2, n_repeats=2, distributions=None, pmem=15000,
        config=dict(
            max_steps=3000, render_step=500, eval_step=500, display_step=100,
            stage_steps=500, curriculum=[dict()]
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
durations["restart_short"] = copy_update(durations["restart"], wall_time="120mins")

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
