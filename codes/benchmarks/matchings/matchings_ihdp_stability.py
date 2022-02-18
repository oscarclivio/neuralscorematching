#!/usr/bin/env python
# coding: utf-8


import sys

sys.path.append("../..")
from matching import Matcher

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pickle

SETTING = 4

PROGRAMME_BENCHMARKS_DEFAULT = [
    ('compute_distances', dict(on=['propensity_x', 'propensity_x_pca'])),
    ('compute_distances', dict(metric='euclidean', on=['x_eucl', 'x_pca_eucl'])),
    ('nearest_neighbor_matching_replacement',
     dict(n_neighbors=1, on=['x_eucl', 'propensity_x', 'x_pca_eucl', 'propensity_x_pca'])),  # 'x_maha','x_pca_maha',
    ('random_matching_replacement', dict(n_neighbors=1, on=['random'])),  # 'x_maha','x_pca_maha',
]

METRICS_DEFAULT = ['att', 'error_att', 'pehe_treated', 'linear_mmd_att', 'rbf_mmd_att', 'fraction_of_predicted_treated',
                   'accuracy', 'cross-entropy']

aggreg_funcs = {
    'median': np.median,
    'min': np.min,
    'max': np.max
}

from sklearn.linear_model import LogisticRegression

# Function to compute matching baselines
# If desired models - eg PCA, logistic regression - are not already pre-computed then we compute and return them
def matching_benchmarks(models, data, data_to_add=None, t_to_add=0, programme=None, metrics=None, seed=None):
    x, y, t, mu0, mu1, cate_true = data[:]
    ps_gt = None
    if data_to_add is not None:
        x_add, y_add, t_add, mu0_add, mu1_add, cate_true_add = data_to_add[:]
        mask = (t_add == t_to_add).ravel()
        x_add = x_add[mask]
        y_add = y_add[mask]
        mu0_add = mu0_add[mask]
        mu1_add = mu1_add[mask]
        cate_true_add = cate_true_add[mask]
        t_add = t_add[mask]
        x = np.vstack(
            (x, x_add))  # equivalent to torch.vstack, see https://pytorch.org/docs/stable/generated/torch.vstack.html
        y = np.vstack((y, y_add))
        t = np.vstack((t, t_add))
        mu0 = np.vstack((mu0, mu0_add))
        mu1 = np.vstack((mu1, mu1_add))
        cate_true = np.vstack((cate_true, cate_true_add))

    if metrics is None:
        metrics = METRICS_DEFAULT

    scores = {}
    scores["x_eucl"] = x
    scores["random"] = x
    # scores["x_maha"] = x
    if "scaler" not in models:
        models["scaler"] = preprocessing.StandardScaler().fit(x)
    if "ps" not in models:
        models["ps"] = LogisticRegression(C=1e6, max_iter=1000).fit(models["scaler"].transform(x), t.ravel())
    scores["propensity_x"] = models["ps"].predict_proba(models["scaler"].transform(x))[:, 1].reshape((-1, 1))
    if "pca" not in models:
        models["pca"] = PCA(n_components=5, random_state=seed).fit(x)
    scores["x_pca_eucl"] = models["pca"].transform(x)
    if "scaler_pca" not in models:
        models["scaler_pca"] = preprocessing.StandardScaler().fit(scores["x_pca_eucl"])
    if "ps_pca" not in models:
        models["ps_pca"] = LogisticRegression(C=1e6, max_iter=1000).fit(
            models["scaler_pca"].transform(scores["x_pca_eucl"]), t.ravel())
    scores["propensity_x_pca"] = models["ps_pca"].predict_proba(models["scaler_pca"].transform(scores["x_pca_eucl"]))[:, 1].reshape((-1, 1))

    if programme is None:
        programme = [el for el in PROGRAMME_BENCHMARKS_DEFAULT]
    programme += [
        ('get_treatment_effects', dict(y=y, evaluate=True, ites=cate_true)),
        ('get_balance_metrics', dict(x=x, add_nothing=True)),
    ]

    m = Matcher(scores, t, att=True, atc=False, propensity_key='propensity_x', seed=seed)
    results = m.apply_programme(programme)

    balance_df = results[-1]
    te_df = results[-2]

    matching_results = {}
    for df in [te_df, balance_df]:
        matching_results.update(
            {
                method + '_' + metric: df.loc[method, metric] \
                for metric in df.columns if metric in metrics \
                for method in df.index
            }
        )

    return matching_results, models, scores


from datasets import IHDP

res = {}
from datetime import datetime

# Compute and save matching results for both train and test data on different seeds and all experiment configurations
for seed in [k*1000 for k in range(2,11+1)]:
    for exp_num in range(1, 50 + 1):
        now = datetime.now()

        train_data = IHDP(exp_num, dataset="train_val", tensor=False, path_data="../../../data", normalise_y=False)
        test_data = IHDP(exp_num, dataset="test", tensor=False, path_data="../../../data", normalise_y=False)

        matching_results, models, scores = matching_benchmarks({}, train_data, seed=seed)
        for key, value in dict(**matching_results).items():
            res.update({key + '_in': res.get(key + '_in', []) + [value]})

        matching_results, _, scores = matching_benchmarks(models, test_data, seed=seed)
        for key, value in matching_results.items():
            res.update({key + '_out': res.get(key + '_out', []) + [value]})
        later = datetime.now()
        diff = later - now
        print("Matching on IHDP dataset {}, seed {}, took {} seconds".format( exp_num, seed, diff.seconds))

df_res = pd.DataFrame(res)

df_res.to_csv('../../../outputs/matching_benchmark_outputs/ihdp_stability.csv')


