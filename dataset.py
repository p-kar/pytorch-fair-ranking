import os
import csv
import pdb
import numpy as np
import random
from nltk import word_tokenize

import torch
from torchvision import transforms
from torch.utils.data import Dataset

def read_review_file(fname):
    """
    Args:
        fname: file containing the sentence pairs for the split
    Output:
        samples: Pairs containing sentence and the label
    """
    with open(fname, 'r') as csvfile:
        reader = csv.reader(csvfile)
        content = [row for row in reader]
    samples = []
    for line in content:
        if len(line) == 2:
            review = word_tokenize(line[0], preserve_line=True)
            label = 0 if line[1] == 'rotten' else 1
            samples.append([review, label])
    return samples

class RottenTomatoesReviewDataset(Dataset):
    """RottenTomatoes review dataset"""

    def __init__(self, root, split, glove_loader, maxlen):

        self.word_to_index = glove_loader.word_to_index
        self.split = split
        self.glove_vec_size = glove_loader.embed_size
        self.data_file = os.path.join(root, 'reviews/' + split + '.csv')
        self.data = read_review_file(self.data_file)
        self.maxlen = maxlen

    def __len__(self):
        return len(self.data)

    def _parse(self, sent):
        sent = [s.lower() if s.lower() in self.word_to_index else '<unk>' for s in sent]
        sent.append('<eos>')
        sent = sent[:self.maxlen]
        padding = ['<pad>' for i in range(max(0, self.maxlen - len(sent)))]
        sent.extend(padding)
        return np.array([self.word_to_index[s] for s in sent])

    def __getitem__(self, idx):
        raw_s = ' '.join(self.data[idx][0])
        label = self.data[idx][1]
        s = torch.LongTensor(self._parse(self.data[idx][0]))
        s_len = min(self.maxlen, len(self.data[idx][0]) + 1)

        return {'s': s, 's_len': s_len, 'label': label, 'raw_s': raw_s}

