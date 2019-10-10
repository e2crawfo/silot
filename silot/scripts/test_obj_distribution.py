import matplotlib.pyplot as plt
import numpy as np

from dps.utils import Config
from dps import cfg

config = Config(
    prior_log_odds=-0.25,
    n_objects=32,
)


with config:
    cfg.update_from_command_line()

    failure_prob = 1. / (1. + np.exp(-cfg.prior_log_odds))
    print(failure_prob)

    support = np.arange(cfg.n_objects + 1)

    raw_probs = (1 - failure_prob) * failure_prob ** support
    probs = raw_probs / raw_probs.sum()
    print(probs)

    fig_height = 2
    fig_width = 1
    fig_unit_size = 3
    fig, axes = plt.subplots(fig_height, fig_width, figsize=(fig_unit_size*fig_width, fig_unit_size*fig_height))

    ax = axes[0]
    ax.bar(support, raw_probs, align='center', alpha=1.0, color='b')
    ax.set_ylim((0, 1))

    ax = axes[1]
    ax.bar(support, probs, align='center', alpha=1.0, color='r')
    ax.set_ylim((0, 1))
    plt.show()
