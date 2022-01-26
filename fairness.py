# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
from aif360.datasets import BinaryLabelDataset
import warnings
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.metrics import accuracy_score, roc_auc_score
warnings.filterwarnings("ignore")

def counterfactual(df, model, pa):
    df_sel = df[df[pa] == 1]
    pred = model.predict(df_sel)
    prob_y1 = pred.sum() / len(pred)
    df_inv = df_sel
    df_inv[pa] = 0
    pred_inv = model.predict(df_inv)
    prob_y1_inv =  pred_inv.sum() / len(pred_inv)
    return prob_y1_inv - prob_y1

def compute_metrics(model, X_test, y_test, df_test, 
                    unprivileged_groups, privileged_groups, 
                    protect_attribute, label):

    result = {}
    
    y_pred_test = np.round(model.predict(X_test), 0)
    result['acc_test'] = accuracy_score(y_true=y_test, y_pred=y_pred_test)
    result['roc_auc'] = roc_auc_score(y_test, y_pred_test)
    
    df_test_aif = BinaryLabelDataset(df=df_test, label_names=[label], 
                                     protected_attribute_names = [protect_attribute]) 
    dataset_pred = df_test_aif.copy()
    dataset_pred.labels = y_pred_test

    classif_metric = ClassificationMetric(df_test_aif, dataset_pred, 
                                          unprivileged_groups=unprivileged_groups,
                                          privileged_groups=privileged_groups)
    result['avg_odds'] = classif_metric.average_odds_difference()
    result['equal_opport'] = classif_metric.equal_opportunity_difference()
    result['false_discovery_rate'] = classif_metric.false_discovery_rate_difference()
    result['entropy_index'] = classif_metric.generalized_entropy_index()
    result['acc_test_clf'] = classif_metric.accuracy(privileged=None)
    result['acc_test_priv'] = classif_metric.accuracy(privileged=True)
    result['acc_test_unpriv'] = classif_metric.accuracy(privileged=False)
    result['counterfactual'] = counterfactual(X_test, model, protect_attribute)
 
    bin_metric = BinaryLabelDatasetMetric(dataset_pred, 
                                          unprivileged_groups=unprivileged_groups,
                                          privileged_groups=privileged_groups)
    result['disp_impact'] = bin_metric.disparate_impact()
    result['stat_parity'] = bin_metric.mean_difference()

    return result


def fairCorrectUnder(df, pa, label, fav, d):#d=1
    """Correct the proportion of positive cases for favoured and unfavoured subgroups through
    subsampling the favoured positive and unfavoured negative classes. Parameter d should be
    a number between -1 and 1 for this to work properly."""
    # subset favoured positive, favoured negative, unfavoured positive, unfavoured negative
    fav_pos = df[(df[pa] == fav) & (df[label] == 1)]
    fav_neg = df[(df[pa] == fav) & (df[label] == 0)]
    unfav_pos = df[(df[pa] != fav) & (df[label] == 1)]
    unfav_neg = df[(df[pa] != fav) & (df[label] == 0)]

    # get favoured and unfavoured number of rows
    fav_size = fav_pos.shape[0] + fav_neg.shape[0]
    unfav_size = unfav_pos.shape[0] + unfav_neg.shape[0]

    # get positive ratios for favoured and unfavoured
    fav_pr = fav_pos.shape[0] / fav_size
    unfav_pr = unfav_pos.shape[0] / unfav_size
    pr = df[df[label] == 1].shape[0] / df.shape[0]

    # coefficients for fitting quad function
    a = ((fav_pr + unfav_pr) / 2.) - pr
    b = (fav_pr - unfav_pr) / 2.
    c = pr

    # corrected ratios
    corr_fpr = (a * (d ** 2)) + (b * d) + c
    corr_upr = (a * (d ** 2)) - (b * d) + c
    
    # correcting constants
    fav_k = corr_fpr / (1 - corr_fpr)
    unfav_k = (1 - corr_upr) / corr_upr
    
    # sample sizes for fav_pos and unfav_neg
    fav_pos_size = math.floor(fav_neg.shape[0] * fav_k)
    unfav_neg_size = math.floor(unfav_pos.shape[0] * unfav_k)
    
    # samples from fav_pos and unfav_neg to correct proportions
    corr_fav_pos = fav_pos.sample(fav_pos_size)
    corr_unfav_neg = unfav_neg.sample(unfav_neg_size)
    
    # concatenate df's
    corr_dfs = [corr_fav_pos, fav_neg, unfav_pos, corr_unfav_neg]
    corr_df = pd.concat(corr_dfs)
    
    corr_df = corr_df.sample(frac=1)
    
    return corr_df
