from cycler import cycler
import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt
from dps.hyper import HyperSearch
from dps.utils import (
    process_path, Config, sha_cache, set_clear_cache,
    confidence_interval, standard_error, grid_subplots
)

data_dir = "/media/data/Dropbox/experiment_data/active/aaai_2020/"
cache_dir = process_path('/home/eric/.cache/dps_plots')
plot_dir = '/media/data/Dropbox/writeups/spatially_invariant_air/aaai_2020/figures'

plot_paths = Config()
plot_paths[''] = ''

verbose_cache = True


def std_dev(ys):
    y_upper = y_lower = [_y.std() for _y in ys]
    return y_upper, y_lower


def ci95(ys):
    conf_int = [confidence_interval(_y, 0.95) for _y in ys]
    y = ys.mean(axis=1)
    y_lower = y - np.array([ci[0] for ci in conf_int])
    y_upper = np.array([ci[1] for ci in conf_int]) - y
    return y_upper, y_lower


def std_err(ys):
    y_upper = y_lower = [standard_error(_y) for _y in ys]
    return y_upper, y_lower


spread_measures = {func.__name__: func for func in [std_dev, ci95, std_err]}


def _get_stage_data_helper(path, stage_idx):
    """
    Return a dataframe, each row of which corresponds to the `stage_idx`-th stage
    of a different run.
    """
    job = HyperSearch(path)
    stage_data = job.extract_stage_data()

    dist_keys = job.dist_keys()

    records = []

    for i, (key, value) in enumerate(sorted(stage_data.items())):
        for (repeat, seed), (df, sc, md) in value.items():
            record = dict(df.iloc[stage_idx])

            for dk in dist_keys:
                record[dk] = md[dk]

            record['idx'] = key.idx
            record['repeat'] = repeat
            record['seed'] = seed

            records.append(record)

    return pd.DataFrame.from_records(records)


def get_stage_data(path, stage_idx, x_key, y_key, spread_measure, y_func=None):
    # Group by idx, get average and spread measure, return x values, mean and spread-measures for y values
    y_func = y_func or (lambda x: x)
    df = _get_stage_data_helper(path, stage_idx)
    groups = sorted(df.groupby(x_key))

    x = [v for v, _df in groups]
    ys = [y_func(_df[y_key]) for v, _df in groups]

    y = [_y.mean() for _y in ys]
    y_upper, y_lower = spread_measures[spread_measure](ys)

    return np.stack([x, y, y_upper, y_lower])


@sha_cache(cache_dir, verbose=verbose_cache)
def get_transfer_baseline_data(path, x_key, y_key, spread_measure, y_func=None):
    y_func = y_func or (lambda y: y)

    job = HyperSearch(path)
    stage_data = job.extract_stage_data()

    x = range(1, 21)
    y = []

    for i, (key, value) in enumerate(sorted(stage_data.items())):
        data = []
        for (repeat, seed), (df, sc, md) in value.items():
            data.append(df[y_key][0])
        data = y_func(np.array(data))
        y.append(data.mean())

    return x, np.array(y)


@sha_cache(cache_dir, verbose=verbose_cache)
def get_mnist_data(path, y_keys, spread_measure):
    if isinstance(y_keys, str):
        y_keys = y_keys.split()

    if not isinstance(y_keys, dict):
        y_keys = {yk: lambda y: y for yk in y_keys}

    job = HyperSearch(path)
    stage_data = job.extract_stage_data()
    assert len(stage_data) == 1  # Should only be one parameter setting
    stage_data = next(iter(stage_data.values()))

    data = defaultdict(list)

    for (repeat, seed), (df, sc, md) in stage_data.items():
        for yk, _ in y_keys.items():
            data[yk].append(df['_test_' + yk])

    x = list(range(1, 13))

    data_stats = {}

    for yk, func in y_keys.items():
        data_yk = np.array(data[yk]).T
        data_yk = func(data_yk)

        y = data_yk.mean(axis=1)
        yu, yl = spread_measures[spread_measure](data_yk)

        data_stats[yk] = (x, y, yu, yl)

    return data_stats


def tex(s):
    return s.replace('_', '\_')


def plot_mnist_n_digits(extension):
    eval_dir = os.path.join(data_dir, 'mnist/eval')

    data_paths = {
        ('SILOT', 'silot', 12): os.path.join(eval_dir, 'run_env=moving-mnist_max-digits=12_alg=conv-silot_duration=long_2019_07_25_22_45_23_seed=0'),
        ('SILOT', 'silot', 6): os.path.join(eval_dir, 'run_env=moving-mnist_max-digits=6_alg=conv-silot_duration=long_2019_07_25_22_45_08_seed=0'),

        # ('SQAIR (conv)', 'conv_sqair', 12): os.path.join(eval_dir, 'run_env=moving-mnist_max-digits=12_alg=conv-sqair_duration=long_2019_07_25_22_46_21_seed=0'),
        ('SQAIR (conv)', 'conv_sqair', 6): os.path.join(eval_dir, 'run_env=moving-mnist_max-digits=6_alg=conv-sqair_duration=long_2019_07_25_22_46_07_seed=0'),

        ('SQAIR (mlp)', 'sqair', 12): os.path.join(eval_dir, 'run_env=moving-mnist_max-digits=12_alg=sqair_duration=long_2019_07_25_22_45_53_seed=0'),
        ('SQAIR (mlp)', 'sqair', 6): os.path.join(eval_dir, 'run_env=moving-mnist_max-digits=6_alg=sqair_duration=long_2019_07_25_22_45_38_seed=0'),
    }

    xlabel = '\# Digits in Test Image'
    xticks = [0, 2, 4, 6, 8, 10, 12]

    ax_params = OrderedDict({
        "MOT:mota": dict(ylabel='MOTA', ylim=(-1.05, 1.05)),
        "AP": dict(ylabel='AP', ylim=(-0.05, 1.05)),
        "count_1norm": dict(ylabel='Count Abs. Error', ylim=(-0.05, 4.0),),
        "prior_MOT:mota": dict(ylabel='Prior MOTA', ylim=(-1.05, 1.05)),
        "prior_AP": dict(ylabel='Prior AP', ylim=(-0.05, 1.05)),
    })

    measures = list(ax_params.keys())
    n_measures = len(measures)

    fig_unit_size = 3

    fig, axes = grid_subplots(1, n_measures, fig_unit_size)
    axes = axes.reshape(-1)

    data_sets = {key: get_mnist_data(path, measures, "ci95") for key, path in data_paths.items()}

    for (measure, axp), ax in zip(ax_params.items(), axes):
        for (title, kind, max_train_digits), dset in data_sets.items():
            label = tex("{} trained on 1--{}".format(title, max_train_digits))
            x, y, *yerr = dset[measure]
            ax.errorbar(x, y, yerr=yerr, label=label)

        fontsize = None
        labelsize = None

        ax.set_ylabel(axp['ylabel'], fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=labelsize)
        ax.set_ylim(axp['ylim'])
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_xticks(xticks)

    axes[0].legend(loc="lower left", fontsize=8)


if __name__ == "__main__":
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['times']})
    rc('text', usetex=True)
    rc('text.latex', preamble=r'\usepackage{amsmath}')
    rc('errorbar', capsize=3)

    funcs = {k[5:]: v for k, v in vars().items() if k.startswith('plot_') and callable(v)}

    parser = argparse.ArgumentParser()
    parser.add_argument("plots", nargs='+', help=",".join(sorted(funcs)))

    style_list = ['default', 'classic'] + sorted(style for style in plt.style.available if style != 'classic')
    parser.add_argument("--style", default="bmh", choices=style_list)

    parser.add_argument("--no-block", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--ext", default="pdf")
    args = parser.parse_args()
    plt.rc('lines', linewidth=1)

    color_cycle = plt.get_cmap("Dark2").colors
    # color_cycle = plt.get_cmap("Paired").colors
    # color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown']

    os.makedirs(plot_dir, exist_ok=True)

    if args.clear_cache:
        set_clear_cache(True)

    with plt.style.context(args.style):
        plt.rc('axes', prop_cycle=(cycler('color', color_cycle)))

        print(funcs)

        for name, do_plot in funcs.items():
            if name in args.plots:
                fig = do_plot(args.ext)

                if args.show:
                    plt.show(block=not args.no_block)