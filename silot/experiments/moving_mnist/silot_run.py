from dps.hyper import run_experiment
from dps.utils import copy_update
from dps.updater import DummyUpdater
from silot.run import basic_config, alg_configs, env_configs

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--max-digits', type=int, choices=[6, 12], required=True)
args, _ = parser.parse_known_args()


readme = "Running SILOT experiment on moving_mnist."

run_kwargs = dict(
    max_hosts=1, ppn=6, cpp=2, gpu_set="0,1", pmem=10000, project="rpp-bengioy",
    wall_time="96hours", cleanup_time="5mins", slack_time="5mins", n_repeats=6,
    copy_locally=True, config=dict(render_step=1000000)
)

durations = dict(
    long=copy_update(run_kwargs),
    short=dict(
        wall_time="180mins", gpu_set="0", ppn=4, n_repeats=4, distributions=None,
        config=dict(max_steps=3000, render_step=500, eval_step=100, display_step=100, stage_steps=600, curriculum=[dict()]),
    ),
    build=dict(
        ppn=1, cpp=1, gpu_set="0", wall_time="180mins", n_repeats=1, distributions=None,
        config=dict(
            do_train=False, get_updater=DummyUpdater, render_hook=None,
            curriculum=[dict()] + [dict(max_digits=i, n_train=100, n_val=1000) for i in range(1, 13)]
        )
    ),
)

config = basic_config.copy()
config.update(env_configs['moving_mnist'])
config.update(alg_configs['conv_silot'], max_digits=args.max_digits)
config.update(final_count_prior_log_odds=0.0125, stage_steps=40000)

run_experiment(
    "moving_mnist_silot",
    config, "silot on moving_mnist.",
    name_variables="max_digits",
    durations=durations
)
