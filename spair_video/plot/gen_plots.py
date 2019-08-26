from cycler import cycler
import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt
from dps import cfg
from dps.hyper import HyperSearch
from dps.train import FrozenTrainingLoopData
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


@sha_cache(cache_dir, verbose=verbose_cache)
def get_mnist_baseline_data(path, y):
    data = FrozenTrainingLoopData(path)
    data = pd.DataFrame.from_records(data.history)
    return list(range(1, 13)), data['_test_' + y]


def tex(s):
    return s.replace('_', '\_')


def plot_mnist_post():
    return _plot_mnist(False)


def plot_mnist_prior():
    return _plot_mnist(True)


def _plot_mnist(prior):
    eval_dir = os.path.join(data_dir, 'mnist/eval')

    data_paths = {
        ('SILOT', 'silot', 12): os.path.join(eval_dir, 'run_env=moving-mnist_max-digits=12_alg=conv-silot_duration=long_2019_07_25_22_45_23_seed=0'),
        ('SILOT', 'silot', 6): os.path.join(eval_dir, 'run_env=moving-mnist_max-digits=6_alg=conv-silot_duration=long_2019_07_25_22_45_08_seed=0'),

        # ('SQAIR (conv)', 'conv_sqair', 12): os.path.join(eval_dir, 'run_env=moving-mnist_max-digits=12_alg=conv-sqair_duration=long_2019_07_30_11_35_31_seed=0'),
        ('SQAIR (conv)', 'conv_sqair', 6): os.path.join(eval_dir, 'run_env=moving-mnist_max-digits=6_alg=conv-sqair_duration=long_2019_07_25_22_46_07_seed=0'),

        # ('SQAIR (mlp)', 'sqair', 12): os.path.join(eval_dir, 'run_env=moving-mnist_max-digits=12_alg=sqair_duration=long_2019_07_25_22_45_53_seed=0'),
        ('SQAIR (mlp)', 'sqair', 6): os.path.join(eval_dir, 'run_env=moving-mnist_max-digits=6_alg=sqair_duration=long_2019_07_25_22_45_38_seed=0'),
    }

    xlabel = '\# Digits in Test Image'
    xticks = [0, 2, 4, 6, 8, 10, 12]

    if prior:
        ax_params = OrderedDict({
            "prior_MOT:mota": dict(ylabel='Prior MOTA', ylim=(-1.05, 1.05)),
            "prior_AP": dict(ylabel='Prior AP', ylim=(-0.05, 1.05)),
        })
        baseline_data = {
            'prior_AP': get_mnist_baseline_data(os.path.join(eval_dir, 'exp_alg=mnist-baseline-AP_2019_08_07_14_34_43_seed=1113481622'), 'AP'),
            'prior_MOT:mota': get_mnist_baseline_data(os.path.join(eval_dir, 'exp_alg=mnist-baseline-mota_2019_08_07_14_44_38_seed=1470331190'), 'MOT:mota'),
        }
    else:
        ax_params = OrderedDict({
            "MOT:mota": dict(ylabel='MOTA', ylim=(-1.05, 1.05)),
            "AP": dict(ylabel='AP', ylim=(-0.05, 1.05)),
            "count_1norm": dict(ylabel='Count Abs. Error', ylim=(-0.05, 4.0),),
        })

        baseline_data = {
            'AP': get_mnist_baseline_data(os.path.join(eval_dir, 'exp_alg=mnist-baseline-AP_2019_08_07_14_34_43_seed=1113481622'), 'AP'),
            'MOT:mota': get_mnist_baseline_data(os.path.join(eval_dir, 'exp_alg=mnist-baseline-mota_2019_08_07_14_44_38_seed=1470331190'), 'MOT:mota'),
            'count_1norm': get_mnist_baseline_data(os.path.join(eval_dir, 'exp_alg=mnist-baseline-count-1norm_2019_08_07_17_24_28_seed=1920714344'), 'count_1norm'),
        }

    baseline_name = 'ConnComp'

    measures = list(ax_params.keys())
    n_measures = len(measures)

    fig_unit_size = 3

    fig, axes = grid_subplots(1, n_measures, fig_unit_size)
    axes = axes.reshape(-1)

    data_sets = {key: get_mnist_data(path, measures, "ci95") for key, path in data_paths.items()}

    for (measure, axp), ax in zip(ax_params.items(), axes):
        for (title, kind, max_train_digits), dset in data_sets.items():
            label = tex("{} - trained on 1--{} digits".format(title, max_train_digits))
            x, y, *yerr = dset[measure]
            ax.errorbar(x, y, yerr=yerr, label=label)

        ax.plot(*baseline_data[measure], label=baseline_name, ls='--')

        fontsize = 12
        labelsize = None

        ax.set_ylabel(axp['ylabel'], fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=labelsize)
        ax.set_ylim(axp['ylim'])
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_xticks(xticks)

    axes[0].legend(loc="lower left", fontsize=8)
    plt.subplots_adjust(left=0.07, bottom=0.15, right=0.98, top=0.94, wspace=.24)

    name = "prior" if prior else "post"

    plot_path = os.path.join(plot_dir, 'mnist', name + cfg.ext)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)


@sha_cache(cache_dir, verbose=verbose_cache)
def get_shapes_data(paths, y_keys, spread_measure):
    data = defaultdict(list)
    x = np.arange(1, 36)
    for path in paths:
        ignore = []

        # For two of the experiments, the training stalled just before starting the 4th stage, so ignore those samples.
        if 'max-shapes=10_small=False_alg=shapes-silot_duration=long_2019_08_13_00_00_51_seed=0' in path:
            ignore = [1]
        if 'max-shapes=20_small=False_alg=shapes-silot_duration=long_2019_08_19_17_30_18_seed=0' in path:
            ignore = [2]

        if isinstance(y_keys, str):
            y_keys = y_keys.split()

        if not isinstance(y_keys, dict):
            y_keys = {yk: lambda y: y for yk in y_keys}

        job = HyperSearch(path)
        stage_data = job.extract_stage_data()

        for (idx, _), d in stage_data.items():
            if idx in ignore:
                continue

            assert len(d) == 1
            df = next(iter(d.values()))[0]
            for yk, _ in y_keys.items():
                data[yk].append(df['_test_' + yk][x-1])

    x = list(x)
    data_stats = {}
    for yk, func in y_keys.items():
        data_yk = np.array(data[yk]).T
        data_yk = func(data_yk)

        y = data_yk.mean(axis=1)
        yu, yl = spread_measures[spread_measure](data_yk)

        data_stats[yk] = (x, y, yu, yl)

    return data_stats


@sha_cache(cache_dir, verbose=verbose_cache)
def get_shapes_baseline_data(path, y):
    data = FrozenTrainingLoopData(path)
    data = pd.DataFrame.from_records(data.history)
    return [1, 5, 10, 15, 20, 25, 30, 35], data['_test_' + y]


def plot_shapes_post():
    return _plot_shapes(False)


def plot_shapes_prior():
    return _plot_shapes(True)


def _plot_shapes(prior):
    eval_dir = os.path.join(data_dir, 'shapes/eval')

    if 0:
        data_paths = {
            ('SILOT', 'silot', 10, True): ['run_env=big-shapes_max-shapes=10_small=True_alg=shapes-silot_duration=long_2019_08_19_17_30_03_seed=0'],
            ('SILOT', 'silot', 20, True): ['run_env=big-shapes_max-shapes=20_small=True_alg=shapes-silot_duration=long_2019_08_24_14_26_42_seed=0'],
            ('SILOT', 'silot', 30, True): ['run_env=big-shapes_max-shapes=30_small=True_alg=shapes-silot_duration=long_2019_08_13_00_01_35_seed=0'],
        }
    else:
        data_paths = {
            ('SILOT', 'silot', 10, False): [
                # 'run_env=big-shapes_max-shapes=10_small=False_alg=shapes-silot_duration=long_2019_08_13_00_00_51_seed=0',
                'run_env=big-shapes_max-shapes=10_small=False_alg=shapes-silot_duration=long_restart_2019_08_24_14_26_55_seed=0',
            ],
            ('SILOT', 'silot', 20, False): [
                # 'run_env=big-shapes_max-shapes=20_small=False_alg=shapes-silot_duration=long_2019_08_19_17_30_18_seed=0',
                'run_env=big-shapes_max-shapes=20_small=False_alg=shapes-silot_duration=long_restart_2019_08_24_14_27_08_seed=0',
            ],
            ('SILOT', 'silot', 30, False): ['run_env=big-shapes_max-shapes=30_small=False_alg=shapes-silot_duration=long_2019_08_13_00_01_20_seed=0'],
        }

    data_paths = {
        k: [os.path.join(eval_dir, d) for d in v]
        for k, v in data_paths.items()}

    xlabel = '\# Shapes in Test Image'
    xticks = list(np.arange(0, 36, 5))

    if prior:
        ax_params = OrderedDict({
            "prior_MOT:mota": dict(ylabel='Prior MOTA', ylim=(-1.05, 1.05)),
            "prior_AP": dict(ylabel='Prior AP', ylim=(-0.05, 1.05)),
        })

        baseline_data = {
            'prior_AP': get_shapes_baseline_data(os.path.join(eval_dir, 'exp_alg=shapes-baseline-AP_2019_08_24_13_38_53_seed=1871884615'), 'AP'),
            'prior_MOT:mota': get_shapes_baseline_data(os.path.join(eval_dir, 'exp_alg=shapes-baseline-mota_2019_08_24_13_53_45_seed=1780273483'), 'MOT:mota'),
        }

    else:
        ax_params = OrderedDict({
            "MOT:mota": dict(ylabel='MOTA', ylim=(-1.05, 1.05)),
            "AP": dict(ylabel='AP', ylim=(-0.05, 1.05)),
            "count_1norm": dict(ylabel='Count Abs. Error', ylim=(-0.05, 4.0),),
        })

        baseline_data = {
            'AP': get_shapes_baseline_data(os.path.join(eval_dir, 'exp_alg=shapes-baseline-AP_2019_08_24_13_38_53_seed=1871884615'), 'AP'),
            'MOT:mota': get_shapes_baseline_data(os.path.join(eval_dir, 'exp_alg=shapes-baseline-mota_2019_08_24_13_53_45_seed=1780273483'), 'MOT:mota'),
            'count_1norm': get_shapes_baseline_data(os.path.join(eval_dir, 'exp_alg=shapes-baseline-count-1norm_2019_08_24_13_45_14_seed=2069764008'), 'count_1norm'),
        }

    baseline_name = 'ConnComp'

    measures = list(ax_params.keys())
    n_measures = len(measures)

    fig_unit_size = 3

    fig, axes = grid_subplots(1, n_measures, fig_unit_size)
    axes = axes.reshape(-1)

    data_sets = {key: get_shapes_data(paths, measures, "ci95") for key, paths in data_paths.items()}

    for (measure, axp), ax in zip(ax_params.items(), axes):
        for (title, kind, max_train_shapes, small), dset in data_sets.items():
            size = "Small" if small else "Big"
            # ls = '--' if small else '-'
            ls = '-'
            label = tex("{} - trained on {}--{} shapes".format(size, max_train_shapes-9, max_train_shapes,))
            x, y, *yerr = dset[measure]

            if cfg.fill:
                line = ax.plot(x, y, label=label, ls=ls)
                c = line[0].get_c()
                yu = y + yerr[0]
                yl = y - yerr[1]
                ax.fill_between(x, yl, yu, color=c, alpha=0.25)
            else:
                ax.errorbar(x, y, yerr=yerr, label=label, ls=ls)

        if baseline_data is not None:
            ax.plot(*baseline_data[measure], label=baseline_name, ls='--')

        fontsize = 12
        labelsize = None

        ax.set_ylabel(axp['ylabel'], fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=labelsize)
        ax.set_ylim(axp['ylim'])
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_xticks(xticks)

    axes[0].legend(loc="lower left", fontsize=8)
    plt.subplots_adjust(left=0.07, bottom=0.15, right=0.98, top=0.94, wspace=.24)

    name = "prior" if prior else "post"

    plot_path = os.path.join(plot_dir, 'shapes', name + cfg.ext)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)


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
    parser.add_argument("--ext", default=".pdf")
    parser.add_argument("--fill", action='store_true')
    args = parser.parse_args()
    plt.rc('lines', linewidth=1)
    ext = args.ext
    ext = ext if ext.startswith('.') else '.' + ext

    color_cycle = plt.get_cmap("Dark2").colors
    # color_cycle = plt.get_cmap("Paired").colors
    # color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown']

    os.makedirs(plot_dir, exist_ok=True)

    if args.clear_cache:
        set_clear_cache(True)

    config = Config(
        fill=args.fill,
        ext=ext,
    )

    with plt.style.context(args.style):
        with config:
            plt.rc('axes', prop_cycle=(cycler('color', color_cycle)))

            print(funcs)

            for name, do_plot in funcs.items():
                if name in args.plots:
                    fig = do_plot()

                    if args.show:
                        plt.show(block=not args.no_block)