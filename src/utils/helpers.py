import json
import code
import collections
import re

import gensim
import ftfy
import torch
import numpy as np


def load_partial_pretrained_word_embedding_as_dict(vocab, embedding_path, embedding_type):
    word_embedding = {}
    if embedding_type in ["glove"]:
        with open(embedding_path) as f:
            for line in f.readlines():
                items = line.strip().split(" ")
                word = items[0]
                vec = items[1:]
                if word in vocab:
                    word_embedding[word] = [float(val) for val in vec]
    elif embedding_type in ["gensim_bin", "gensim_txt"]:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(
            embedding_path,
            binary=True if embedding_type == "gensim_bin" else False
        )
        for word in vocab.keys():
            if word in word2vec.vocab:
                word_embedding[word] = [float(v) for v in list(word2vec[word])]
    else:
        raise Exception(f"Unknown embedding type: {embedding_type}")
    return word_embedding


def standardize_english_text(string):
    """
    string cleaning for English
    """
    string = ftfy.fix_text(string)
    string = re.sub(r'—|–|―', '-', string)
    string = re.sub(r'…', '...', string)
    string = re.sub(r'[`´]', "'", string)
    string = re.sub(r"[^A-Za-z0-9,!?/()\'\.\<\>\"]", " ", string)
    string = re.sub(r"\.{3}", " ...", string)
    string = string.replace("\'m", " \'m")
    string = string.replace("\'s", " \'s")
    string = string.replace("\'re", " \'re")
    string = string.replace("n\'t", " n\'t")
    string = string.replace("\'ve", " \'ve")
    string = string.replace("\'d", " \'d")
    string = string.replace("\'ll", " \'ll")
    string = string.replace("\"", " \" ")
    string = string.replace(",", " , ")
    string = string.replace("!", " ! ")
    string = string.replace("?", " ? ")
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip()
    string = string.lower()
    return string


def repackage_hidden_states(hidden_states):
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states.detach()
    else:
        hidden_states = tuple(repackage_hidden_states(h) for h in hidden_states)
    return hidden_states


class StatisticsReporter:
    def __init__(self):
        self.statistics = collections.defaultdict(list)

    def update_data(self, d):
        for k, v in d.items():
            if isinstance(v, (int, float, np.int, np.int32, np.int64, np.float, np.float32, np.float64)):
                self.statistics[k] += [v]

    def clear(self):
        self.statistics = collections.defaultdict(list)

    def to_string(self):
        string_list = []
        for k, v in sorted(list(self.statistics.items()), key=lambda x: x[0]):
            mean = np.mean(v)
            string_list.append("{}: {:.5g}".format(k, mean))
        return ", ".join(string_list)

    def get_value(self, key):
        if key in self.statistics:
            value = np.mean(self.statistics[key])
            return value
        else:
            return None

    def items(self):
        for k, v in self.statistics.items():
            yield k, v
