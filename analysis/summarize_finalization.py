"""Conditional paired variant-cluster bootstrap of saved finalization predictions.

No retraining, threshold selection or new candidate selection. Shared variant IDs
receive identical multiplicities across panels. Intervals are conditional on the
fitted CV models and do not include training/ES selection uncertainty.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'pipelines')]
from schema import normalize_sentinels, drop_all_nan_rows
from utils import PANELS, TEST_PRIORS


def score(y, decisions, weights, prior):
    positive = weights * (y == 1)
    negative = weights * (y == 0)
    if positive.sum() == 0 or negative.sum() == 0:
        return np.full(decisions.shape[1], np.nan)
    tpr = positive @ decisions / positive.sum()
    fpr = negative @ decisions / negative.sum()
    tp, fn = prior*tpr, prior*(1-tpr)
    fp, tn = (1-prior)*fpr, (1-prior)*(1-fpr)
    f1 = 2*tp / (2*tp+fp+fn)
    den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = np.divide(tp*tn-fp*fn, den, out=np.zeros_like(den), where=den>0)
    return (f1+mcc)/2


def main(directory, draws):
    directory = Path(directory)
    metrics = pd.read_csv(directory / 'fold_metrics.csv')
    arrays = np.load(directory / 'probabilities.npz')
    choices = [(rule, policy) for rule in ('train_composite', 'deployed_rule') for policy in ('P0', 'P1', 'P2')]
    prepared = {}
    labels = {}
    for panel, path in PANELS.items():
        frame = drop_all_nan_rows(normalize_sentinels(pd.read_csv(path)), verbose=False)
        y = frame.Label.to_numpy(int)
        ids = frame.Variant_ID.to_numpy()
        for ident, label in zip(ids, y):
            assert ident not in labels or labels[ident] == label
            labels[ident] = label
        decisions = np.zeros((len(y), len(choices)))
        for j, (rule, policy) in enumerate(choices):
            counts = np.zeros(len(y), int)
            subset = metrics[(metrics.panel == panel) & (metrics.rule == rule) & (metrics.policy == policy)]
            for row in subset.itertuples():
                prefix = f'{panel}:f{row.fold}:{policy}'
                idx = arrays[prefix+':te_idx']
                np.testing.assert_array_equal(y[idx], arrays[prefix+':te_y'])
                decisions[idx, j] = arrays[prefix+':te_p'] >= row.threshold
                counts[idx] += 1
            assert np.all(counts == 1)
        prepared[panel] = (ids, y, decisions)
    unique = list(labels)
    positions = {ident: i for i, ident in enumerate(unique)}
    cls = np.array([labels[ident] for ident in unique])
    class_indices = [np.flatnonzero(cls == value) for value in (0, 1)]
    prepared = {panel: (np.array([positions[ident] for ident in ids]), y, pred) for panel, (ids, y, pred) in prepared.items()}
    rng = np.random.default_rng(842)
    panel_order = list(PANELS)
    results = np.empty((draws, len(panel_order), len(choices)))
    for draw in range(draws):
        sampled = np.concatenate([rng.choice(idx, len(idx), replace=True) for idx in class_indices])
        multiplicities = np.bincount(sampled, minlength=len(unique))
        for i, panel in enumerate(panel_order):
            idx, y, pred = prepared[panel]
            results[draw, i] = score(y, pred, multiplicities[idx], TEST_PRIORS[panel])
    results = np.concatenate([results, results.mean(axis=1, keepdims=True)], axis=1)
    rows = []
    for i, panel in enumerate(panel_order + ['equal_four_panel_mean']):
        for base in (0, 3):
            for offset in (1, 2):
                delta = results[:, i, base+offset] - results[:, i, base]
                bounds = np.nanquantile(delta, [.025, .5, .975])
                rows.append({'panel': panel, 'rule': choices[base][0], 'policy': choices[base+offset][1],
                             'draws': draws, 'finite_draws': int(np.isfinite(delta).sum()),
                             'conditional_delta_p025': bounds[0], 'conditional_delta_median': bounds[1],
                             'conditional_delta_p975': bounds[2]})
    pd.DataFrame(rows).to_csv(directory / 'conditional_bootstrap.csv', index=False)
    (directory / 'conditional_bootstrap_method.json').write_text(json.dumps({
        'draws': draws, 'seed': 842, 'unit': 'shared Variant_ID cluster, resampled within class',
        'thresholds': 'frozen separately per original CV fold',
        'limitation': 'conditional on fitted CV models; no retraining, source-seed or policy-selection uncertainty; not a deployment significance test'
    }, indent=2)+'\n')
    print(pd.DataFrame(rows).query("panel == 'equal_four_panel_mean'").to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default=str(ROOT / 'analysis/output/finalization/comparison/full'))
    parser.add_argument('--draws', type=int, default=2000)
    args = parser.parse_args()
    main(args.directory, args.draws)
