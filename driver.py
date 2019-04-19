import os
import pdb
from utils import *
from args import get_args
from train_sentiment import train_sentiment
from train_rank import train_rank

if __name__ == '__main__':

    opts = get_args()
    set_random_seeds(opts.seed)

    if opts.mode == 'train_sentiment':
        train_sentiment(opts)
    elif opts.mode == 'train_rank':
        train_rank(opts)
    else:
        raise NotImplementedError('unrecognized mode')

