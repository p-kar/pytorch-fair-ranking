import pdb
import time
import numpy as np
import pulp
from pulp import *
import timeit
import torch
import pdb

def calc_fairness_func(order, gt_scores, genres, metric):

    N = gt_scores.shape[0]
    v = torch.log(torch.tensor(2.0)) / torch.log(torch.arange(N).float() + 2.0)
    v = v[order].to(order.device)
    gt_scores = gt_scores.float().to(order.device)

    genres = torch.max(genres, dim=1)[1]
    pos_indices = torch.eq(genres, 0).nonzero().squeeze(1)
    neg_indices = torch.eq(genres, 1).nonzero().squeeze(1)
    G0_exp = v[pos_indices].mean()
    G1_exp = v[neg_indices].mean()

    if metric == 'DP':
        fairness_score = G0_exp / G1_exp
    elif metric == 'DTR':
        G0_util = gt_scores[pos_indices].mean()
        G1_util = gt_scores[neg_indices].mean()
        fairness_score = (G0_exp * G1_util) / (G1_exp * G0_util)
    elif metric == 'DIR':
        G0_ctr = torch.mul(v, gt_scores)[pos_indices].mean()
        G1_ctr = torch.mul(v, gt_scores)[neg_indices].mean()
        fairness_score = (G0_exp * G1_ctr) / (G1_exp * G0_ctr)
    else:
        raise NotImplementedError('unknown metric')

    return fairness_score


