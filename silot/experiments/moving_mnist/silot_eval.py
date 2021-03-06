from dps.hyper import run_experiment
from dps.tf.updater import DummyUpdater
from silot.run import basic_config, alg_configs, env_configs, silot_mnist_eval_prepare_func

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--max-digits', type=int, choices=[6, 12], required=True)
args, _ = parser.parse_known_args()

readme = "Evaluate SILOT experiment on moving_mnist."


durations = dict(
    long=dict(
        max_hosts=1, ppn=6, cpp=2, gpu_set="0,1", pmem=10000, project="rpp-bengioy",
        wall_time="12hours", cleanup_time="5mins", slack_time="5mins", n_repeats=6,
        copy_locally=True, distributions=None,
        config=dict(
            render_step=1000000,
            n_train=96,
            n_val=1008,
            do_train=False,
            curriculum=[dict(min_digits=i, max_digits=i) for i in range(1, 13)],
            prepare_func=silot_mnist_eval_prepare_func,
        ),
    ),
    build=dict(
        ppn=1, cpp=1, gpu_set="0", wall_time="180mins", n_repeats=1, distributions=None,
        config=dict(
            do_train=False, n_train=96, n_val=1008, get_updater=DummyUpdater, render_hook=None,
            curriculum=[dict(min_digits=i, max_digits=i) for i in range(1, 13)]
        )
    ),
)

config = basic_config.copy()
config.update(env_configs['moving_mnist'])
config.update(alg_configs['silot'], max_digits=args.max_digits)

run_experiment(
    "eval_moving_mnist_silot",
    config, "silot on moving_mnist.",
    name_variables="max_digits",
    durations=durations
)
