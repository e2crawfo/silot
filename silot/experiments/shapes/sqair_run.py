from dps.hyper import run_experiment
from silot.run import basic_config, alg_configs, env_configs

distributions = None
late_config = dict(max_steps=250000)

long_wall_time = "72hours"

readme = "Running SQAIR experiment on hard_shapes."

pmem = 6000
project = "rpp-bengioy"
n_gpus = 2
gpu_set = ",".join(str(i) for i in range(n_gpus))

durations = dict(
    long=dict(
        max_hosts=1, ppn=6, cpp=1, gpu_set=gpu_set, pmem=pmem, project=project,
        wall_time=long_wall_time, cleanup_time="5mins", slack_time="5mins", n_repeats=6,
        copy_locally=True,
    ),
    medium=dict(
        max_hosts=1, ppn=6, cpp=1, gpu_set=gpu_set, pmem=pmem, project=project,
        wall_time="6hours", cleanup_time="5mins", slack_time="5mins", n_repeats=6,
        config=dict(stage_steps=3000, max_steps=12000), copy_locally=True,
    ),
    short=dict(
        max_hosts=1, ppn=6, cpp=1, gpu_set=gpu_set, pmem=pmem, project=project,
        wall_time="180mins", cleanup_time="2mins", slack_time="2mins", n_repeats=6,
        config=dict(max_steps=3000, render_step=500, eval_step=500, display_step=100, stage_steps=600),
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
config.update(env_configs['hard_shapes'])
config.update(alg_configs['fixed_sqair'])
config.update(late_config)

run_experiment(
    "hard_shapes_sqair",
    config, "sqair on hard_shapes.",
    distributions=distributions, durations=durations
)
