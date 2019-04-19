import os
import pdb
import time
import random
import shutil
import warnings
import argparse
import tensorboardX
from tensorboardX import SummaryWriter

import torch
import torch.optim
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import *
from models import RankNet
from logger import TensorboardXLogger
from dataset import RankSampler, RottenTomatoesRankingDataset, rank_collate_func

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def calculate_dcg(scores, gt_scores):
    """
    Args:
        scores: Ranking scores produced by the model (b)
        gt_scores: GT tomatometer scores
    Output:
        dcg: Discounted cumulative gain for the ranking
    """
    n = scores.shape[0]
    _, order = torch.sort(scores, descending=True)
    gt_scores = gt_scores[order]
    gt_scores = gt_scores / torch.max(gt_scores)
    v = 1.0 / torch.log2(torch.arange(n).double() + 2.0).to(device)
    dcg = torch.dot(gt_scores, v)

    return dcg

def run_iter(opts, data, model):
    s, s_len = data['sent'].to(device), data['sent_len'].to(device)
    genres, order = data['genre'].to(device), data['order'].to(device)
    scores = data['score'].to(device)

    out, loss = model.train_forward(s, s_len, genres, order)
    dcg = calculate_dcg(out, scores)

    return dcg, loss

def evaluate(opts, model, loader):
    model.eval()

    time_start = time.time()
    val_loss = 0.0
    val_dcg = 0.0
    num_batches = 0.0

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):

            dcg, loss = run_iter(opts, data, model)
            val_loss += loss.data.cpu().item()
            val_dcg += dcg
            num_batches += 1

    avg_valid_loss = val_loss / num_batches
    avg_valid_dcg = val_dcg / num_batches
    time_taken = time.time() - time_start

    return avg_valid_loss, avg_valid_dcg, time_taken


def train_rank(opts):

    glove_loader = GloveLoader(os.path.join(opts.data_dir, 'glove', opts.glove_emb_file))
    train_dataset = RottenTomatoesRankingDataset(opts.data_dir, 'train', glove_loader, opts.maxlen)
    train_loader = DataLoader(train_dataset, batch_size=opts.bsize, sampler=RankSampler(train_dataset), \
        collate_fn=rank_collate_func, num_workers=opts.nworkers)

    valid_dataset = RottenTomatoesRankingDataset(opts.data_dir, 'val', glove_loader, opts.maxlen)
    valid_loader = DataLoader(valid_dataset, batch_size=opts.bsize, sampler=RankSampler(valid_dataset), \
        collate_fn=rank_collate_func, num_workers=opts.nworkers)
    model = RankNet(opts.hidden_size, opts.dropout_p, glove_loader, opts.enc_arch, \
        num_genres=len(train_dataset.genres), pretrained_base=opts.pretrained_base)

    if opts.optim == 'adam':
        # raise NotImplementedError('update LR')
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.wd)
    else:
        raise NotImplementedError("Unknown optim type")

    start_n_iter = 0
    # for choosing the best model
    best_val_dcg = 0.0

    model_path = os.path.join(opts.save_path, 'model_latest.net')
    if opts.resume and os.path.exists(model_path):
        # restoring training from save_state
        print ('====> Resuming training from previous checkpoint')
        save_state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(save_state['state_dict'])
        start_n_iter = save_state['n_iter']
        best_val_dcg = save_state['best_val_dcg']
        opts = save_state['opts']
        opts.start_epoch = save_state['epoch'] + 1

    model = model.to(device)

    # for logging
    logger = TensorboardXLogger(opts.start_epoch, opts.log_iter, opts.log_dir)
    logger.set(['DCG', 'loss'])
    logger.n_iter = start_n_iter

    for epoch in range(opts.start_epoch, opts.epochs):
        model.train()
        logger.step()

        for batch_idx, data in enumerate(train_loader):
            dcg, loss = run_iter(opts, data, model)

            # optimizer step
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), opts.max_norm)
            optimizer.step()

            logger.update(dcg, loss)

        val_loss, val_dcg, time_taken = evaluate(opts, model, valid_loader)
        # log the validation losses
        logger.log_valid(time_taken, val_dcg, val_loss)
        print ('')

        # Save the model to disk
        if val_dcg >= best_val_dcg:
            best_val_dcg = val_dcg
            save_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'n_iter': logger.n_iter,
                'opts': opts,
                'val_dcg': val_dcg,
                'best_val_dcg': best_val_dcg
            }
            model_path = os.path.join(opts.save_path, 'model_best.net')
            torch.save(save_state, model_path)

        save_state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'n_iter': logger.n_iter,
            'opts': opts,
            'val_dcg': val_dcg,
            'best_val_dcg': best_val_dcg
        }
        model_path = os.path.join(opts.save_path, 'model_latest.net')
        torch.save(save_state, model_path)

