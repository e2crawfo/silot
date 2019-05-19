from dps.hyper import run_experiment
from spair_video.run import basic_config, alg_configs, env_configs

late_config = dict(max_steps=250000)

distributions = None

readme = "Running ISSPAIR experiment on hard_shapes."

pmem = 15000
project = "rpp-bengioy"
n_gpus = 2
gpu_set = ",".join(str(i) for i in range(n_gpus))

durations = dict(
    long=dict(
        max_hosts=1, ppn=6, cpp=1, gpu_set=gpu_set, pmem=pmem, project=project,
        wall_time="72hours", cleanup_time="5mins", slack_time="5mins", n_repeats=6,
        copy_locally=True,
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
config.update(env_configs['hard_shapes'])
config.update(alg_configs['exp_isspair'])
config.update(late_config)

run_experiment(
    "hard_shapes_isspair",
    config, "isspair on hard_shapes.",
    distributions=distributions, durations=durations
)
