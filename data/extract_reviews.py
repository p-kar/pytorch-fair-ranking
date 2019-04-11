import os
import csv
import pdb
import json
import random
import shutil
import numpy as np

data_file = 'movie_reviews.json'

with open(data_file, 'r') as fp:
    reviews = json.load(fp)
    all_reviews = [r['review_text'] for r in reviews]
    all_labels = [r['review_flag'] for r in reviews]
    pos_reviews = [r for r in zip(all_reviews, all_labels) if r[1] == 'fresh']
    neg_reviews = [r for r in zip(all_reviews, all_labels) if r[1] == 'rotten']
    random.shuffle(pos_reviews)
    random.shuffle(neg_reviews)
    max_len = max(len(pos_reviews), len(neg_reviews))
    pos_reviews = pos_reviews[:max_len]
    neg_reviews = neg_reviews[:max_len]
    balanced_reviews = pos_reviews + neg_reviews
    all_reviews, all_labels = zip(*balanced_reviews)

    num_reviews = len(all_reviews)
    pairs = list(zip(all_reviews, all_labels))
    random.shuffle(pairs)
    all_reviews, all_labels = zip(*pairs)
    train_idx = int(0.75 * num_reviews)
    val_idx = int(0.85 * num_reviews)

    if os.path.isdir('./reviews'):
        shutil.rmtree('./reviews')
    os.makedirs('./reviews')

    with open(os.path.join('./reviews', 'train.csv'), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(zip(all_reviews[:train_idx], all_labels[:train_idx]))
    with open(os.path.join('./reviews', 'val.csv'), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(zip(all_reviews[train_idx:val_idx], all_labels[train_idx:val_idx]))
    with open(os.path.join('./reviews', 'test.csv'), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(zip(all_reviews[val_idx:], all_labels[val_idx:]))

