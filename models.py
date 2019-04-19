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

        self.linear = nn.Sequential( \
            nn.Dropout(p=dropout_p), \
            nn.Linear(hidden_size * 4, hidden_size), \
            nn.ReLU())

    def forward(self, s, len_s):
        """
        Args:
            s: Tokenized sentence (b x L)
            len_s: Sentence length (b)
        Output:
            out: Output vector with concatenated avg. and max. pooled
                sentence encoding (b x hidden_size)
        """
        batch_size = s.shape[0]
        maxlen = s.shape[1]

        s = self.embedding(s).transpose(0, 1)
        # L x b x embed_size
        h1, _ = self.encoder1(s)
        h2, _ = self.encoder2(torch.cat((s, h1), dim=2))
        v = torch.transpose(h2, 0, 1)
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
        out = self.linear(out)
        # b x hidden_size
        return out

class BiLSTMBase(nn.Module):
    """
    Simple sentence encoder using a BiLSTM
    """
    def __init__(self, hidden_size, dropout_p, glove_loader):
        super(BiLSTMBase, self).__init__()

        word_vectors = glove_loader.word_vectors
        word_vectors = np.vstack(word_vectors)
        vocab_size = word_vectors.shape[0]
        embed_size = word_vectors.shape[1]

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.load_state_dict({'weight': torch.Tensor(word_vectors)})
        
        self.encoder = nn.Sequential( \
            nn.Dropout(p=dropout_p), \
            nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=1, bidirectional=True))

        self.linear = nn.Sequential( \
            nn.Dropout(p=dropout_p), \
            nn.Linear(hidden_size * 4, hidden_size), \
            nn.ReLU())

    def forward(self, s, len_s):
        """
        Args:
            s: Tokenized sentence (b x L)
            len_s: Sentence length (b)
        Output:
            out: Output vector with concatenated avg. and max. pooled
                sentence encoding (b x hidden_size)
        """
        batch_size = s.shape[0]
        maxlen = s.shape[1]

        s = self.embedding(s).transpose(0, 1)
        # L x b x embed_size
        h, _ = self.encoder(s)
        v = torch.transpose(h, 0, 1)
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
        out = self.linear(out)
        # b x hidden_size
        return out


class Classifier(nn.Module):
    """Classifier on top of the sentence encoder"""
    def __init__(self, hidden_size, dropout_p, glove_loader, enc_arch):
        """
        Args:
            hidden_size: Size of the intermediate linear layers
            dropout_p: Dropout probability for intermediate dropout layers
            glove_loader: GLoVe embedding loader
            enc_arch: Sentence encoder architecture [bilstm | sse]
        """
        super(Classifier, self).__init__()

        if enc_arch == 'bilstm':
            self.encoder = BiLSTMBase(hidden_size, dropout_p, glove_loader)
        elif enc_arch == 'sse':
            self.encoder = SSEBase(hidden_size, dropout_p, glove_loader)
        else:
            raise NotImplementedError('unknown sentence encoder')

        # prediction layer for the sentiment analysis task
        self.sent_pred = nn.Sequential( \
            nn.Dropout(p=dropout_p), \
            nn.Linear(hidden_size, 2))

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
        # b x hidden_size
        out = self.sent_pred(v)
        # b x 2
        return out

class RankNet(nn.Module):
    """Rank net on top of the sentence encoder"""
    def __init__(self, hidden_size, dropout_p, glove_loader, enc_arch, num_genres, pretrained_base=None):
        """
        Args:
            hidden_size: Size of the intermediate linear layers
            dropout_p: Dropout probability for intermediate dropout layers
            glove_loader: GLoVe embedding loader
            enc_arch: Sentence encoder architecture [bilstm | sse]
            num_genres: Number of movie genres
            pretrained_base: Path to the pretrained model
        """
        super(RankNet, self).__init__()

        if enc_arch == 'bilstm':
            self.encoder = BiLSTMBase(hidden_size, dropout_p, glove_loader)
        elif enc_arch == 'sse':
            self.encoder = SSEBase(hidden_size, dropout_p, glove_loader)
        else:
            raise NotImplementedError('unknown sentence encoder')

        self.drop = nn.Dropout(p=dropout_p)
        self.rank_layer = nn.Linear(hidden_size + num_genres, 1)
        self.reset_parameters()

        if pretrained_base is not None:
            raise NotImplementedError()

    def reset_parameters(self):
        """Initialize network weights using Xavier init (with bias 0.01)"""
        self.apply(ixvr)

    def forward(self, s, s_len, genres):
        """
        Args:
            s: Tokenized sentence (b x L)
            s_len: Sentence length (b)
            genres: One hot encoding for the genre (b x num_genres)
        Output:
            out: Ranking scores (b x 1)
        """
        v = self.drop(self.encoder(s, s_len))
        # b x hidden_size
        v = torch.cat((v, genres), dim=1)
        # b x (hidden_size + num_genres)
        out = self.rank_layer(v)
        # b x 1
        return out

    def train_forward(self, s, s_len, genres, order):
        """
        Args:
            s: Tokenized sentence (b x L)
            s_len: Sentence length (b)
            genres: One hot encoding for the genre (b x num_genres)
            order: GT ranking order (b)
        Output:
            out: Ranking scores (b)
            loss: Ranknet loss
        """
        out = self.forward(s, s_len, genres).view(-1)
        loss = self.ranknet_loss(out, order)

        return out, loss

    def ranknet_loss(self, scores, order):
        """
        Args:
            scores: normalization scores should be detached from the
                computation graph (b)
            order: GT ranking order (b)
        Output:
            loss: RankNet loss
        """
        out = scores
        scores = scores.detach()
        batch_size = scores.shape[0]
        device = torch.device('cuda' if next(self.parameters()).is_cuda else 'cpu')

        s_i = torch.ones(batch_size, batch_size).to(device) * scores.view(1, -1)
        # b x b
        s_j = torch.ones(batch_size, batch_size).to(device) * scores.view(-1, 1)
        # b x b
        d = s_i - s_j
        # b x b
        P_ij = torch.triu(torch.ones(batch_size, batch_size), diagonal=1).to(device)[order,:][:,order]
        # b x b
        lambda_ij = torch.sigmoid(d) - P_ij
        # b x b
        lambda_i = lambda_ij.sum(dim=1) - lambda_ij.sum(dim=0)
        # b
        loss = torch.mul(lambda_i, out).mean()

        return loss

