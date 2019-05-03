from dps.hyper import run_experiment
from spair_video.run import basic_config, alg_configs, env_configs

readme = "Running ISSPAIR experiment."

distributions = [
    dict(
        n_digits=n_digits,
        min_digits=n_digits,
        max_digits=n_digits,
        max_time_steps=n_digits)
    for n_digits in [1, 3, 5, 7, 9]]


durations = dict(
    # long=dict(
    #     max_hosts=1, ppn=16, cpp=1, gpu_set="0,1,2,3", wall_time="36hours",
    #     project="rpp-bengioy", cleanup_time="10mins",
    #     slack_time="10mins", n_repeats=6, step_time_limit="36hours"),

    # build=dict(
    #     max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="20mins",
    #     project="rpp-bengioy", cleanup_time="2mins",
    #     slack_time="2mins", n_repeats=1, step_time_limit="20mins",
    #     config=dict(do_train=False)),

    short=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="30mins",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, config=dict(max_steps=100))
)

config = basic_config.copy()
config.update(env_configs['moving_mnist'])
config.update(alg_configs['isspair'])

config = basic_config.copy()
run_experiment(
    "test_isspair", config, "First test of spair_video.",
    alg_configs=alg_configs, env_configs=env_configs
)
