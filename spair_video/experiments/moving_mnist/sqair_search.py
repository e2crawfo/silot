from dps.hyper import run_experiment
from dps.utils import copy_update
from spair_video.run import basic_config, alg_configs, env_configs

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--conv', action='store_true')
parser.add_argument('--max-digits', default=0, type=int, choices=[6, 12])
args, _ = parser.parse_known_args()

late_config = dict()

distributions = None
n = args.max_digits
late_config.update(max_digits=n, n_objects=n,)

wall_time_lookup = {
    6: "30hours",
    12: "60hours",
}

conv_wall_time_lookup = {
    6: "30hours",
    12: "60hours",
}

if args.conv:
    long_wall_time = conv_wall_time_lookup[n]
else:
    long_wall_time = wall_time_lookup[n]

distributions = dict(
    disc_step_bias=[0, 2.5, 5., 7.5],
    prop_step_bias=[0, 2.5, 5., 7.5],
)

readme = "Running SQAIR experiment on moving_mnist."

run_kwargs = dict(
    max_hosts=1, ppn=32, cpp=2, gpu_set="0,1,2,3", pmem=6000, project="rpp-bengioy",
    wall_time=long_wall_time, cleanup_time="5mins", slack_time="5mins", n_repeats=1,
    copy_locally=True, config=dict(max_steps=250000)
)

durations = dict(
    long=copy_update(run_kwargs),
    medium=copy_update(
        run_kwargs,
        wall_time="6hours", config=dict(stage_steps=3000, max_steps=12000),
    ),
    short=dict(
        wall_time="180mins", gpu_set="0", ppn=8, n_repeats=4, distributions=None,
        config=dict(max_steps=3000, render_step=500, eval_step=100, display_step=100, stage_steps=600),
    ),
    build=dict(
        ppn=1, cpp=1, gpu_set="0", wall_time="60mins", n_repeats=1, distributions=None,
        config=dict(do_train=False, render_first=False, render_final=False),
    ),
)

config = basic_config.copy()
config.update(env_configs['moving_mnist'])
alg_name = 'conv_sqair' if args.conv else 'sqair'
config.update(alg_configs[alg_name])

run_experiment(
    "moving_mnist_sqair",
    config, "sqair on moving_mnist.",
    name_variables="max_digits",
    distributions=distributions,
    durations=durations
)
