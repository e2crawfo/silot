""" Find the value that gives the best performance for each metric on the validation set.
    Later we will evaluate these values on a test set.

"""
import argparse
import pprint

from dps.train import FrozenTrainingLoopData

parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()

data = FrozenTrainingLoopData(args.path)

val_data = data.step_data('val')

metrics = {'AP': True, 'count_1norm': False, 'MOT:mota': True}
best = {m: [] for m in metrics}

for stage_idx, stage_data in val_data.groupby('stage_idx'):
    for metric, maximize in metrics.items():
        metric_data = stage_data[metric]
        if maximize:
            best_idx = metric_data.idxmax()
        else:
            best_idx = metric_data.idxmin()

        best_value = metric_data[best_idx]
        best_threshold = stage_data.cc_threshold[best_idx]
        best_cc_threshold = stage_data.cc_threshold[best_idx]
        best_cosine_threshold = stage_data.cosine_threshold[best_idx]

        best[metric].append(dict(cc_threshold=best_cc_threshold, cosine_threshold=best_cosine_threshold))
        # best[metric].append(dict(idx=best_idx, value=best_value, threshold=best_threshold))

pprint.pprint(best)
