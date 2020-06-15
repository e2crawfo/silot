from dps.hyper import run_experiment
from dps.utils import copy_update
from dps.tf.updater import DummyUpdater
from silot.run import basic_config, alg_configs, env_configs

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--game', choices='space_invaders asteroids carnival wizard_of_wor'.split(), required=True)
args, _ = parser.parse_known_args()


readme = "Running SILOT experiment on atari."

run_kwargs = dict(
    n_repeats=6, tasks_per_gpu=3, project="rrg-bengioy-ad_gpu",
    installation_script_path="/home/e2crawfo/spair_video/silot/slurm_build_local_env.sh",
    wall_time="96hours", cleanup_time="5mins", slack_time="5mins", config=dict(render_step=1000000)
)

durations = dict(
    long=copy_update(run_kwargs),
    short=dict(
        wall_time="180mins", tasks_per_gpu=4, n_repeats=4, distributions=None,
        config=dict(
            max_steps=3000, render_step=500, eval_step=100, display_step=100,
            stage_steps=600, curriculum=[dict()]
        ),
    ),
    build=dict(
        wall_time="180mins", tasks_per_gpu=1, n_repeats=1, distributions=None,
        config=dict(
            do_train=False, get_updater=DummyUpdater, render_hook=None,
        )
    ),
)

config = basic_config.copy()
config.update(env_configs[args.game])
config.update(alg_configs['atari_train_silot'])

run_experiment(
    "atari_silot",
    config, "silot on atari.",
    name_variables="game",
    durations=durations
)
