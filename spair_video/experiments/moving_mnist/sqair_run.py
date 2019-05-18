from dps.hyper import run_experiment
from spair_video.run import basic_config, alg_configs, env_configs

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n-digits', default=0, type=int, choices=[2, 4, 6, 8, 10])
args, _ = parser.parse_known_args()

late_config = dict()

distributions = []
n = args.n_digits
late_config.update(
    n_digits=n, min_digits=n, max_digits=n, n_steps_per_image=n,
    fixed_presence=True, disc_prior_type='special'
)

wall_time_lookup = {
    2: "24hours",
    4: "36hours",
    6: "50hours",
    8: "72hours",
    10: "96hours",
}

long_wall_time = wall_time_lookup[n]

readme = "Running SQAIR experiment on moving_mnist."

pmem = 6000
project = "rpp-bengioy"
n_gpus = 2
gpu_set = ",".join(str(i) for i in range(n_gpus))

durations = dict(
    long=dict(
        max_hosts=1, ppn=6, cpp=1, gpu_set=gpu_set, pmem=pmem, project=project,
        wall_time=long_wall_time, cleanup_time="5mins", slack_time="5mins", n_repeats=6,
        copy_locally=True, config=dict(max_steps=250000)
    ),
    medium=dict(
        max_hosts=1, ppn=6, cpp=1, gpu_set=gpu_set, pmem=pmem, project=project,
        wall_time="6hours", cleanup_time="5mins", slack_time="5mins", n_repeats=6,
        config=dict(stage_steps=3000, max_steps=12000), copy_locally=True,
    ),
    short=dict(
        max_hosts=1, ppn=6, cpp=1, gpu_set=gpu_set, pmem=pmem, project=project,
        wall_time="120mins", cleanup_time="2mins", slack_time="2mins", n_repeats=6,
        config=dict(max_steps=1000, render_step=500, eval_step=100, display_step=100, stage_steps=200),
        distributions=None, copy_locally=True,
    ),
    build=dict(
        max_hosts=1, ppn=1, cpp=1, gpu_set="0", pmem=pmem, project=project,
        wall_time="60mins", cleanup_time="2mins", slack_time="2mins", n_repeats=1,
        config=dict(do_train=False, render_first=False, render_final=False),
        distributions=None,
    ),
)

config = basic_config.copy()
config.update(env_configs['moving_mnist'])
config.update(alg_configs['sqair'])
config.update(late_config)

run_experiment(
    "moving_mnist_sqair",
    config, "sqair on moving_mnist.",
    name_variables="n_digits",
    distributions=distributions, durations=durations
)
