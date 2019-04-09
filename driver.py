import os
import pdb
from utils import *
from args import get_args
from train_sentiment import train_sentiment

if __name__ == '__main__':

    opts = get_args()
    set_random_seeds(opts.seed)

    if opts.mode == 'train_sentiment':
        train_sentiment(opts)
    else:
        raise NotImplementedError('unrecognized mode')

