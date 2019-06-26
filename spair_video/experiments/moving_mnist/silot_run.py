from dps.hyper import run_experiment
from spair_video.run import basic_config, alg_configs, env_configs

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n-digits', default=0, type=int, choices=[2, 4, 6, 8, 10])
args, _ = parser.parse_known_args()

late_config = dict(max_steps=250000)

distributions = None
n = args.n_digits
late_config.update(n_digits=n, min_digits=n, max_digits=n)

readme = "Running SILOT experiment on moving_mnist."

pmem = 15000
project = "rpp-bengioy"
n_gpus = 2
gpu_set = ",".join(str(i) for i in range(n_gpus))

durations = dict(
    long=dict(
        max_hosts=1, ppn=6, cpp=1, gpu_set=gpu_set, pmem=pmem, project=project,
        wall_time="96hours", cleanup_time="5mins", slack_time="5mins", n_repeats=6,
        copy_locally=True, config=dict(max_steps=250000)
    ),
    test_finish=dict(
        max_hosts=1, ppn=6, cpp=1, gpu_set=gpu_set, pmem=pmem, project=project,
        wall_time="1hour", cleanup_time="5mins", slack_time="5mins", n_repeats=6,
        copy_locally=True, config=dict(max_steps=250000)
    ),
    short=dict(
        max_hosts=1, ppn=6, cpp=1, gpu_set=gpu_set, pmem=pmem, project=project,
        wall_time="180mins", cleanup_time="2mins", slack_time="2mins", n_repeats=6,
        config=dict(max_steps=3000, render_step=500, eval_step=500, display_step=100, stage_steps=600),
        copy_locally=True,
    ),
    build=dict(
        max_hosts=1, ppn=1, cpp=1, gpu_set=gpu_set, pmem=pmem, project=project,
        wall_time="120mins", cleanup_time="2mins", slack_time="2mins", n_repeats=1,
        config=dict(do_train=False, render_first=False, render_final=False),
    ),
)

config = basic_config.copy()
config.update(env_configs['moving_mnist'])
config.update(alg_configs['exp_silot'])
config.update(late_config)

run_experiment(
    "moving_mnist_silot",
    config, "silot on moving_mnist.",
    name_variables="n_digits",
    distributions=distributions, durations=durations
)
