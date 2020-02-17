from dps.hyper import run_experiment
from dps.tf.updater import DummyUpdater
from dps.utils import copy_update
from silot.run import basic_config, alg_configs, env_configs

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--max-shapes', type=int, choices=[10, 20, 30], required=True)
parser.add_argument('--is-small', action='store_true')
args, _ = parser.parse_known_args()


readme = "Eval SILOT experiment on shapes."

max_shapes = args.max_shapes
small = args.is_small

if max_shapes == 10:
    if small:
        dir_name = 'run_env=big-shapes-small_max-shapes=10_alg=shapes-silot_duration=long_2019_08_14_22_21_50_seed=0'
    else:
        dir_name = 'run_env=big-shapes_max-shapes=10_alg=shapes-silot_duration=restart10_2019_08_19_17_39_20_seed=0'
        # dir_name = 'run_env=big-shapes_max-shapes=10_alg=shapes-silot_duration=long_2019_07_30_16_55_21_seed=0'
elif max_shapes == 20:
    if small:
        dir_name = 'run_env=big-shapes-small_max-shapes=20_alg=shapes-silot_duration=long_2019_08_14_22_22_05_seed=0'
    else:
        dir_name = 'run_env=big-shapes_max-shapes=20_alg=shapes-silot_duration=restart20_2019_08_19_17_39_34_seed=0'
        # dir_name = 'run_env=big-shapes_max-shapes=20_alg=shapes-silot_duration=long_2019_08_01_07_44_22_seed=0'
elif max_shapes == 30:
    if small:
        dir_name = 'run_env=big-shapes-small_max-shapes=30_alg=shapes-silot_duration=long_2019_08_05_10_25_53_seed=0'
    else:
        dir_name = 'run_env=big-shapes_max-shapes=30_alg=shapes-silot_duration=restart_2019_08_07_12_39_32_seed=0'
else:
    raise Exception()

import os
experiment_path = os.path.join(
    "/scratch/e2crawfo/dps_data/parallel_experiments_run/aaai_2020_silot/shapes/run",
    dir_name, 'experiments')

dirs = sorted(os.listdir(experiment_path))
load_paths = []
for d in dirs:
    stage = 2
    while True:
        load_path = os.path.join(experiment_path, d, 'weights/best_of_stage_{}'.format(stage))

        if os.path.exists(load_path + '.meta'):
            break

        stage -= 1

        if stage < 0:
            raise Exception("No valid weights found.")

    print("Using stage {} for load_path {}.".format(stage, load_path))
    load_paths.append(load_path)

durations = dict(
    long=dict(
        max_hosts=1, ppn=4, cpp=2, gpu_set="0,1", pmem=10000, project="rpp-bengioy",
        wall_time="16hours", cleanup_time="5mins", slack_time="5mins", n_repeats=1,
        copy_locally=True, distributions=dict(load_path=load_paths),
        config=dict(
            render_step=1000000,
            n_train=32,
            n_val=1008,
            do_train=False,
            curriculum=[dict(min_shapes=i, max_shapes=i) for i in range(1, 36)],
        ),
    ),
    build=dict(
        ppn=1, cpp=2, gpu_set="0", wall_time="6hours", n_repeats=1, distributions=None,
        config=dict(
            do_train=False, n_train=32, n_val=1008, get_updater=DummyUpdater, render_hook=None,
            curriculum=[dict(min_shapes=i, max_shapes=i) for i in range(1, 36)]
        )
    ),
)

durations['long_restart'] = copy_update(durations['long'], ppn=1, gpu_set="0")

config = basic_config.copy()
config.update(env_configs['big_shapes'])
config.update(alg_configs['shapes_silot'])

config.batch_size = 8
config.n_prop_objects = 36

config.update(max_shapes=args.max_shapes, small=args.is_small)

run_experiment(
    "shapes_silot", config, "silot on shapes.", name_variables="max_shapes,small", durations=durations
)
