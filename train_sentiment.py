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
from models import SSEClassifier
from logger import TensorboardXLogger
from dataset import RottenTomatoesReviewDataset

use_cuda = torch.cuda.is_available()

def run_iter(opts, data, model, criterion, device):
    sent, sent_len, target = data['s'].to(device), data['s_len'].to(device), data['label'].to(device)
    output = model(sent, sent_len)

    loss = criterion(output, target)
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    acc = correct / sent.shape[0]

    return acc, loss

def evaluate(opts, model, loader, criterion, device):
    model.eval()

    time_start = time.time()
    val_loss = 0.0
    val_acc = 0.0
    num_batches = 0.0

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):

            acc, loss = run_iter(opts, data, model, criterion, device)
            val_loss += loss.data.cpu().item()
            val_acc += acc
            num_batches += 1

    avg_valid_loss = val_loss / num_batches
    avg_valid_acc = val_acc / num_batches
    time_taken = time.time() - time_start

    return avg_valid_loss, avg_valid_acc, time_taken


def train_sentiment(opts):
    
    device = torch.device("cuda" if use_cuda else "cpu")

    glove_loader = GloveLoader(os.path.join(opts.data_dir, 'glove', opts.glove_emb_file))
    train_loader = DataLoader(RottenTomatoesReviewDataset(opts.data_dir, 'train', glove_loader, opts.maxlen), \
        batch_size=opts.bsize, shuffle=True, num_workers=opts.nworkers)
    valid_loader = DataLoader(RottenTomatoesReviewDataset(opts.data_dir, 'val', glove_loader, opts.maxlen), \
        batch_size=opts.bsize, shuffle=False, num_workers=opts.nworkers)
    model = SSEClassifier(opts.hidden_size, opts.dropout_p, glove_loader, xavier_init=True)

    if opts.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.wd)
    else:
        raise NotImplementedError("Unknown optim type")

    criterion = nn.CrossEntropyLoss()

    start_n_iter = 0
    # for choosing the best model
    best_val_acc = 0.0

    model_path = os.path.join(opts.save_path, 'model_latest.net')
    if opts.resume and os.path.exists(model_path):
        # restoring training from save_state
        print ('====> Resuming training from previous checkpoint')
        save_state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(save_state['state_dict'])
        start_n_iter = save_state['n_iter']
        best_val_acc = save_state['best_val_acc']
        opts = save_state['opts']
        opts.start_epoch = save_state['epoch'] + 1

    model = model.to(device)

    # for logging
    logger = TensorboardXLogger(opts.start_epoch, opts.log_iter, opts.log_dir)
    logger.set(['acc', 'loss'])
    logger.n_iter = start_n_iter

    for epoch in range(opts.start_epoch, opts.epochs):
        model.train()
        logger.step()

        for batch_idx, data in enumerate(train_loader):
            acc, loss = run_iter(opts, data, model, criterion, device)

            # optimizer step
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), opts.max_norm)
            optimizer.step()

            logger.update(acc, loss)

        val_loss, val_acc, time_taken = evaluate(opts, model, valid_loader, criterion, device)
        # log the validation losses
        logger.log_valid(time_taken, val_acc, val_loss)
        print ('')

        # Save the model to disk
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'n_iter': logger.n_iter,
                'opts': opts,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc
            }
            model_path = os.path.join(opts.save_path, 'model_best.net')
            torch.save(save_state, model_path)

        save_state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'n_iter': logger.n_iter,
            'opts': opts,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc
        }
        model_path = os.path.join(opts.save_path, 'model_latest.net')
        torch.save(save_state, model_path)

