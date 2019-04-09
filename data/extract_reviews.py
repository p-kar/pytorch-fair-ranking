import os
import csv
import pdb
import json
import shutil
import numpy as np

data_file = 'rt_data.json'

with open(data_file, 'r') as fp:
    data = json.load(fp)
    all_reviews = []
    all_labels = []
    for movie in data:
        reviews = movie['reviews']
        all_reviews.extend([r['review_text'] for r in reviews])
        all_labels.extend([r['review_flag'] for r in reviews])

    num_reviews = len(all_reviews)
    perm = np.random.permutation(np.arange(num_reviews))
    all_reviews = [all_reviews[p] for p in perm]
    all_labels = [all_labels[p] for p in perm]
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

