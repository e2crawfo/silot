from dps.hyper import run_experiment
from dps.utils import copy_update
from silot.run import basic_config, alg_configs, env_configs
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--conv', action='store_true')
parser.add_argument('--max-digits', type=int, choices=[6, 12], required=True)
args, _ = parser.parse_known_args()


distributions = dict(
    disc_step_bias=list(np.linspace(0., 5., 3)),
    step_success_prob=list(np.linspace(0.4, 0.75, 4)),
)

readme = "Running SQAIR experiment on moving_mnist."

run_kwargs = dict(
    max_hosts=2, ppn=12, cpp=2, gpu_set="0,1,2,3", pmem=5000, project="rpp-bengioy",
    wall_time="71hours", cleanup_time="5mins", slack_time="5mins", n_repeats=2,
    copy_locally=True, config=dict(max_steps=120000, patience=0, curriculum=[dict()])
)

durations = dict(
    long=copy_update(run_kwargs),
    short=dict(
        wall_time="180mins", gpu_set="0", ppn=4, n_repeats=4, distributions=None,
        config=dict(
            max_steps=3000, render_step=500, eval_step=100,
            display_step=100, stage_steps=600, curriculum=[dict()]),
    ),
    test_load=dict(
        wall_time="180mins", gpu_set="0", ppn=4, n_repeats=4, distributions=None,
        config=dict(max_steps=500, curriculum=[dict(), dict()]),
    ),
    build=dict(
        ppn=1, gpu_set="0", wall_time="60mins", n_repeats=1, distributions=None,
        config=dict(do_train=False, render_first=False, render_final=False),
    ),
)

config = basic_config.copy()
config.update(env_configs['moving_mnist'])
alg_name = 'conv_sqair' if args.conv else 'sqair'
config.update(
    alg_configs[alg_name], max_digits=args.max_digits, n_objects=args.max_digits, render_step=10000000,
    curriculum=[dict(), dict(max_steps=100)],
)

run_experiment(
    "moving_mnist_sqair",
    config, "sqair on moving_mnist.",
    name_variables="max_digits",
    distributions=distributions,
    durations=durations
)
