import pdb
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import *

class SSEBase(nn.Module):
    """
    Model architecture similar to the Shortcut-Stacked Sentence Encoder as
    described in https://arxiv.org/pdf/1708.02312.pdf.
    """
    def __init__(self, hidden_size, dropout_p, glove_loader):
        """
        Args:
            hidden_size: Size of the intermediate linear layers
            dropout_p: Dropout probability for intermediate dropout layers
            glove_loader: GLoVe embedding loader
        """
        super(SSEBase, self).__init__()

        word_vectors = glove_loader.word_vectors
        word_vectors = np.vstack(word_vectors)
        vocab_size = word_vectors.shape[0]
        embed_size = word_vectors.shape[1]

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.load_state_dict({'weight': torch.Tensor(word_vectors)})
        
        self.encoder1 = nn.Sequential( \
            nn.Dropout(p=dropout_p), \
            nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=1, bidirectional=True))
        self.encoder2 = nn.Sequential( \
            nn.Dropout(p=dropout_p), \
            nn.LSTM(input_size=embed_size + 2*hidden_size, hidden_size=hidden_size, num_layers=1, bidirectional=True))
        self.encoder3 = nn.Sequential( \
            nn.Dropout(p=dropout_p), \
            nn.LSTM(input_size=embed_size + 4*hidden_size, hidden_size=hidden_size, num_layers=1, bidirectional=True))

    def forward(self, s, len_s):
        """
        Args:
            s: Tokenized sentence (b x L)
            len_s: Sentence length (b)
        Output:
            out: Output vector with concatenated avg. and max. pooled
                sentence encoding (b x (hidden_size * 4))
        """
        batch_size = s.shape[0]
        maxlen = s.shape[1]

        s = self.embedding(s).transpose(0, 1)
        # L x b x embed_size
        h1, _ = self.encoder1(s)
        h2, _ = self.encoder2(torch.cat((s, h1), dim=2))
        h3, _ = self.encoder3(torch.cat((s, h1, h2), dim=2))
        v = torch.transpose(h3, 0, 1)
        # b x L x (hidden_size * 2)

        mask = torch.arange(0, maxlen).expand(batch_size, maxlen)
        if torch.cuda.is_available():
            mask = mask.cuda()
        mask = mask < len_s.unsqueeze(-1)
        mask = mask.float()

        v_avg = torch.sum(torch.mul(v, mask.unsqueeze(-1)), dim=1)
        v_avg = torch.div(v_avg, torch.sum(mask, dim=1).unsqueeze(-1))
        # b x (hidden_size * 2)
        v_max = torch.max(torch.mul(v, mask.unsqueeze(-1)), dim=1)[0]
        # b x (hidden_size * 2)

        out = torch.cat((v_avg, v_max), dim=1)
        # b x (hidden_size * 4)
        return out

class SSEClassifier(nn.Module):
    """Classifier on top of the Shortcut-stacked encoder"""
    def __init__(self, hidden_size, dropout_p, glove_loader, xavier_init=True):
        """
        Args:
            hidden_size: Size of the intermediate linear layers
            dropout_p: Dropout probability for intermediate dropout layers
            glove_loader: GLoVe embedding loader
            xavier_init: Initialise network using Xavier init
        """
        super(SSEClassifier, self).__init__()

        self.encoder = SSEBase(hidden_size, dropout_p, glove_loader)

        # prediction layer for the sentiment analysis task
        self.sent_pred = nn.Sequential( \
            nn.Dropout(p=dropout_p), \
            nn.Linear(hidden_size * 4, hidden_size), \
            nn.ReLU(), \
            nn.Dropout(p=dropout_p), \
            nn.Linear(hidden_size, 2))

        if xavier_init:
            self.reset_parameters()

    def reset_parameters(self):
        """Initialize network weights using Xavier init (with bias 0.01)"""
        self.apply(ixvr)

    def forward(self, s, s_len):
        """
        Args:
            s: Tokenized sentence (b x L)
            s_len: Sentence length (b)
        """
        v = self.encoder(s, s_len)
        # b x L x (hidden_size * 4)
        out = self.sent_pred(v)
        # b x 2
        return out

