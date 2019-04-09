import pdb
import torch
import random
import numpy as np
import torch.nn as nn

def set_random_seeds(seed):
    """
    Sets the random seeds for numpy, python, pytorch cpu and gpu
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_glove_file(fname):
    """
    Args:
        fname: File containing the GloVe vectors
    Output:
        word_to_index: dictionary mapping from word to emb. index
        index_to_word: dictionary mapping from emb. index to word
        word_vectors: list of GloVe vectors
    """

    with open(fname, 'r') as f:
        content = f.readlines()

    word_to_index = {}
    index_to_word = {}
    word_vectors = []

    for idx, line in enumerate(content):
        line = line.strip().split()
        word, vec = line[0], line[1:]
        vec = np.array([float(v) for v in vec])
        word_to_index[word] = idx
        index_to_word[idx] = word
        word_vectors.append(vec)

    extra_words = ['<sos>', '<eos>', '<pad>', '<unk>']
    num_words = len(word_vectors)
    glove_vec_size = word_vectors[0].shape[0]

    for word in extra_words:
        word_to_index[word] = num_words
        index_to_word[num_words] = word
        word_vectors.append(np.random.randn(glove_vec_size))
        num_words += 1

    return word_to_index, index_to_word, word_vectors

class GloveLoader:
    def __init__(self, glove_emb_file):
        self.word_to_index, self.index_to_word, self.word_vectors = load_glove_file(glove_emb_file)
        self.embed_size = self.word_vectors[0].shape[0]

def ixvr(input_layer, bias_val=0.01):
    ignore_layers = ["<class 'torch.nn.modules.batchnorm.BatchNorm2d'>", \
        "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>", \
        "<class 'torch.nn.modules.sparse.Embedding'>"]
    # If the layer is an LSTM
    if str(type(input_layer)) == "<class 'torch.nn.modules.rnn.LSTM'>":
        for i in range(input_layer.num_layers):
            nn.init.xavier_normal_(getattr(input_layer, 'weight_ih_l%d'%(i)))
            nn.init.xavier_normal_(getattr(input_layer, 'weight_hh_l%d'%(i)))
            nn.init.constant_(getattr(input_layer, 'bias_ih_l%d'%(i)), bias_val)
            nn.init.constant_(getattr(input_layer, 'bias_hh_l%d'%(i)), bias_val)
    elif not str(type(input_layer)) in ignore_layers:
        if hasattr(input_layer, 'weight'):
            nn.init.xavier_normal_(input_layer.weight);
        if hasattr(input_layer, 'bias'):
            nn.init.constant_(input_layer.bias, bias_val);

    return input_layer
