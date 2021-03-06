import os
import csv
import pdb
import json
import numpy as np
import random
from collections import Counter
from nltk import word_tokenize

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import default_collate
from lp_solver import lp_solver_func
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
            review = word_tokenize(line[0].lower(), preserve_line=True)
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

def read_movie_data_div_by_genres(fname, genres):
    """
    Args:
        fname: file containing the movie data with reviews
        genres: list containing distinct movie genres
    Output:
        movie_dict: Dictionary containing movies segregated based
            on genres and critic rating ('rotten' and 'fresh')
    """
    with open(fname, 'r') as fp:
        content = json.load(fp)
    movie_dict = {k: {'fresh': [], 'rotten': []} for k in genres}

    for movie in content:
        if len(movie['reviews']) == 0:
            continue
        if movie['tomatometer'] == 'unknown':
            continue
        if 'Genre:' not in movie['movie_metadata']:
            continue
        present_flags = [int(g in movie['movie_metadata']['Genre:']) for g in genres]
        # either none of the genres are present or more than 1 genre is present
        # then ignore this movie
        if sum(present_flags) != 1:
            continue
        # this list should contain only 1 value
        movie_genre = [g for g in genres if g in movie['movie_metadata']['Genre:']][0]
        tomatometer = float(movie['tomatometer'][:-1]) / 100.0
        rating = movie['critic_rating']

        movie_item = {
            'url': movie['url'],
            'movie_name': movie['movie_name'],
            'movie_genre': movie_genre,
            'rating': rating,
            'tomatometer': tomatometer,
            'reviews': [r for r in movie['reviews'] if r['review_flag'] == rating]
        }
        if len(movie_item['reviews']) == 0:
            continue
        movie_dict[movie_genre][rating].append(movie_item)

    counts = {g: {k: len(movie_dict[g][k]) for k in movie_dict[g].keys()} for g in genres}
    genre_counts = {g: sum([counts[g][k] for k in counts[g].keys()]) for g in genres}
    movie_count = sum([genre_counts[g] for g in genres])
    
    print('Dropped {} movies'.format(len(content) - movie_count))
    print('Found {} movies'.format(movie_count))
    print('Genre counts:')
    for g in genres:
        print('{:>10}: {}'.format(g, genre_counts[g]))

    return movie_dict

def read_movie_data_div_by_studio(fname, thresh=4):
    """
    Args:
        fname: file containing the movie data with reviews
    Output:
        movie_dict: Dictionary containing movies segregated based
            on studio size and critic rating ('rotten' and 'fresh')
    """
    with open(fname, 'r') as fp:
        content = json.load(fp)
    genres = ['big_studio', 'small_studio']
    tot_num_movies = len(content)
    content = [movie for movie in content if 'Studio:' in movie['movie_metadata'].keys()]
    pick_first = lambda x: x[0] if isinstance(x, list) else x
    studios = [pick_first(movie['movie_metadata']['Studio:']) for movie in content]
    big_studios = set([s[0] for s in Counter(studios).most_common() if s[1] >= thresh])

    movie_dict = {k: {'fresh': [], 'rotten': []} for k in genres}

    for movie in content:
        if len(movie['reviews']) == 0:
            continue
        if movie['tomatometer'] == 'unknown':
            continue
        assert 'Studio:' in movie['movie_metadata'].keys()

        studio = pick_first(movie['movie_metadata']['Studio:'])
        movie_genre = 'big_studio' if studio in big_studios else 'small_studio'
        tomatometer = float(movie['tomatometer'][:-1]) / 100.0
        rating = movie['critic_rating']

        movie_item = {
            'url': movie['url'],
            'movie_name': movie['movie_name'],
            'movie_genre': movie_genre,
            'rating': rating,
            'tomatometer': tomatometer,
            'reviews': [r for r in movie['reviews'] if r['review_flag'] == rating]
        }
        if len(movie_item['reviews']) == 0:
            continue
        movie_dict[movie_genre][rating].append(movie_item)

    counts = {g: {k: len(movie_dict[g][k]) for k in movie_dict[g].keys()} for g in genres}
    genre_counts = {g: sum([counts[g][k] for k in counts[g].keys()]) for g in genres}
    movie_count = sum([genre_counts[g] for g in genres])
    
    print('Dropped {} movies'.format(tot_num_movies - movie_count))
    print('Found {} movies'.format(movie_count))
    print('Genre counts:')
    for g in genres:
        print('{:>10}: {}'.format(g, genre_counts[g]))

    return movie_dict

class RottenTomatoesRankingDataset(Dataset):
    """RottenTomatoes movie ranking dataset"""

    def __init__(self, root, split, glove_loader, maxlen, div_by='genre', genres=['Drama', 'Horror']):
        """
        If div_by flag is set to 'studio', the genres argument is ignored
        """

        self.word_to_index = glove_loader.word_to_index
        self.split = split
        self.glove_vec_size = glove_loader.embed_size
        self.maxlen = maxlen
        self.data_file = os.path.join(root, 'rt_data.json')
        self.div_by = div_by
        if self.div_by == 'genre':
            self.genres = genres
            self.data = read_movie_data_div_by_genres(self.data_file, genres)
        elif self.div_by == 'studio':
            self.genres = ['big_studio', 'small_studio']
            self.data = read_movie_data_div_by_studio(self.data_file)
        else:
            raise NotImplementedError('unknown div_by type')
        self._split_movie_data()
        self.data_size = self._data_size()
        self.shuffle()

    def _split_movie_data(self):
        for g in self.data.keys():
            for k in self.data[g].keys():
                split_idx = int(0.75 * len(self.data[g][k]))
                if self.split == 'train':
                    sl = slice(0, split_idx)
                else:
                    sl = slice(split_idx, len(self.data[g][k]))
                self.data[g][k] = self.data[g][k][sl]

    def _data_size(self):
        size = 0
        for g in self.data.keys():
            for k in self.data[g].keys():
                size = max(size, len(self.data[g][k]))
        return size

    def _parse_sent(self, sent):
        sent = [s.lower() if s.lower() in self.word_to_index else '<unk>' for s in sent]
        sent.append('<eos>')
        sent = sent[:self.maxlen]
        padding = ['<pad>' for i in range(max(0, self.maxlen - len(sent)))]
        sent.extend(padding)
        return np.array([self.word_to_index[s] for s in sent])

    def _parse_movie(self, movie_item):
        ret_dict = {
            'url': movie_item['url'],
            'name': movie_item['movie_name'],
            'score': movie_item['tomatometer']
        }
        review = random.choice(movie_item['reviews'])
        sent = word_tokenize(review['review_text'].lower())

        ret_dict['sent'] = torch.LongTensor(self._parse_sent(sent))
        ret_dict['sent_len'] = min(self.maxlen, len(sent) + 1)
        ret_dict['sent_raw'] = ' '.join(sent)
        ret_dict['flag'] = 0 if review['review_flag'] == 'rotten' else 1

        genre = int(self.genres.index(movie_item['movie_genre']))
        ret_dict['genre'] = torch.zeros(len(self.genres))
        ret_dict['genre'][genre] = 1.0

        return ret_dict

    def shuffle(self):
        for g in self.data.keys():
            for k in self.data[g].keys():
                random.shuffle(self.data[g][k])

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        ret = []
        for g in self.data.keys():
            for k in self.data[g].keys():
                lidx = idx % len(self.data[g][k])
                ret.append(self._parse_movie(self.data[g][k][lidx]))
        return ret

class RankSampler(Sampler):
    """
    Calls the shuffle() function in RottenTomatoesRankingDataset
    before returning an iterator
    """
    def __init__(self, data_source):
        self.data_source = data_source
        self.replacement = False
        self._num_samples = len(self.data_source)

    @property
    def num_samples(self):
        return self._num_samples

    def __iter__(self):
        self.data_source.shuffle()
        n = len(self.data_source)
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self._num_samples

def rank_collate_func(batches):
    """
    Calculates the ranking for the batch without the
    fairness constraint
    """
    batch = [item for b in batches for item in b]
    batch_size = len(batch)
    ret_dict = default_collate(batch)
    _, order = torch.sort(ret_dict['score'], descending=True)
    ret_dict['order'] = order
    ret_dict['PIJ'] = torch.triu(torch.ones(batch_size, batch_size), diagonal=1)[order,:][:,order]

    return ret_dict

def rank_lp_func(constraint, batches):
    """
    Calculates rank based on the lp constraints
    """
    #print ("Using rank_lp")
    #constraint = 'DemoParity'
    batch = [item for b in batches for item in b]
    ret_dict = default_collate(batch)
    genres = ret_dict['genre'].max(dim=1)[1].numpy()
    scores = ret_dict['score'].numpy()
    dcg, P = lp_solver_func(scores, genres, constraint)
    order = np.argmax(P,axis=0 ) #argmax across column for position id
    ret_dict['order'] = torch.from_numpy(order).long()

    T = np.triu(np.ones(P.shape), k=1)
    # S - Sij -> probability of doc i after pos j
    S = np.matmul(T, P.transpose()).transpose()
    # PP - pair preference matrix probability, PPij -> probability of doc i occurring before doc j
    PP = np.matmul(P, S.transpose())
    # Pairwise positioning matrix PIJ_{ij} -> denotes if doc i should be ranked higher than doc j
    PIJ = (PP > PP.transpose()).astype(np.float64)
    ret_dict['PIJ'] = torch.from_numpy(PIJ)

    return ret_dict
