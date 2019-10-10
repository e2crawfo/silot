from dps.hyper import run_experiment
from dps.utils import copy_update
from silot.run import basic_config, alg_configs, env_configs

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--conv', action='store_true')
parser.add_argument('--max-digits', type=int, choices=[6, 12], required=True)
args, _ = parser.parse_known_args()

readme = "Running SQAIR experiment on moving_mnist."

run_kwargs = dict(
    max_hosts=1, ppn=6, cpp=2, gpu_set="0,1", pmem=10000, project="rpp-bengioy",
    wall_time="96hours", cleanup_time="5mins", slack_time="5mins", n_repeats=6,
    copy_locally=True
)

durations = dict(
    long=copy_update(run_kwargs),
    short=dict(
        wall_time="180mins", ppn=3, gpu_set="0", n_repeats=3,
        config=dict(max_steps=3000, render_step=500, eval_step=100, display_step=100, stage_steps=600, curriculum=[dict(), dict()]),
    ),
    build=dict(
        ppn=1, cpp=1, gpu_set="0", wall_time="60mins", n_repeats=1,
        config=dict(do_train=False, render_first=False, render_final=False),
    ),
)

config = basic_config.copy()
config.update(env_configs['moving_mnist'])

alg_name = 'conv_sqair' if args.conv else 'sqair'
config.update(
    alg_configs[alg_name], max_digits=args.max_digits, n_objects=args.max_digits,
)

search_params = {
    (True, 12): dict(disc_step_bias=5, step_success_prob=0.4),
    (True, 6): dict(disc_step_bias=5., step_success_prob=0.516),
    (False, 12): dict(disc_step_bias=5., step_success_prob=0.4),
    (False, 6): dict(disc_step_bias=5., step_success_prob=0.516),
}[(args.conv, args.max_digits)]

config.update(**search_params, render_step=1000000)

run_experiment(
    "moving_mnist_sqair",
    config, "sqair on moving_mnist.",
    name_variables="max_digits",
    durations=durations
)
