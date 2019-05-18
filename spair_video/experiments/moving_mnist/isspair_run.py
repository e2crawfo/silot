import numpy as np
from dps.hyper import run_experiment
from spair_video.run import basic_config, alg_configs, env_configs

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n-digits', default=0, type=int, choices=[2, 4, 6, 8, 10])
args, _ = parser.parse_known_args()

late_config = dict(
    train_example_range=(0.0, 0.7),
    val_example_range=(0.7, 0.8),
    test_example_range=(0.8, 0.9),
    n_frames=2,
)

distributions = []
n = args.n_digits
late_config.update(n_digits=n, min_digits=n, max_digits=n, n_steps_per_image=n,)


readme = "Running ISSPAIR experiment on moving_mnist."

pmem = 16000
project = "rpp-bengioy"
ppn = max(len(distributions), 1)
n_gpus = int(np.ceil(ppn / 4))
gpu_set = ",".join(str(i) for i in range(n_gpus))

durations = dict(
    long=dict(
        max_hosts=1, ppn=ppn, cpp=1, gpu_set=gpu_set, pmem=pmem, project=project,
        wall_time="24hours", cleanup_time="5mins", slack_time="5mins", n_repeats=1,
    ),
    medium=dict(
        max_hosts=1, ppn=ppn, cpp=1, gpu_set=gpu_set, pmem=pmem, project=project,
        wall_time="12hours", cleanup_time="5mins", slack_time="5mins", n_repeats=1,
    ),
    short=dict(
        max_hosts=1, ppn=ppn, cpp=1, gpu_set=gpu_set, pmem=pmem, project=project,
        wall_time="120mins", cleanup_time="2mins", slack_time="2mins", n_repeats=1,
        config=dict(max_steps=100, render_step=25, eval_step=25, display_step=25),
        distributions=None, copy_locally=True,
    ),
    build=dict(
        max_hosts=1, ppn=1, cpp=1, gpu_set=gpu_set, pmem=pmem, project=project,
        wall_time="120mins", cleanup_time="2mins", slack_time="2mins", n_repeats=1,
        config=dict(do_train=False, render_first=False, render_final=False),
        distributions=None,
    ),
)

config = basic_config.copy()
config.update(env_configs['moving_mnist'])
config.update(alg_configs['exp_isspair'])
config.update(late_config)

run_experiment(
    "moving_mnist_isspair",
    config, "isspair on moving_mnist.",
    name_variables="n_digits",
    distributions=distributions, durations=durations
)
