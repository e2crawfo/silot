import numpy as np
from dps.hyper import run_experiment
from silot.run import basic_config, alg_configs, env_configs

readme = "Running SILOT experiment."

distributions = [
    dict(d_attr_prior_std=s) for s in [0.1, 0.2, 0.4, 0.8]
]

pmem = 16000
project = "rpp-bengioy"
ppn = len(distributions)
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
config.update(alg_configs['exp_silot'])

run_experiment(
    "test_silot", config, "First test of silot.",
    distributions=distributions, durations=durations
)
