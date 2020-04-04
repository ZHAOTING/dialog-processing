import copy
import collections
import math
import random
import json
import code

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataSource():

    def __init__(self, data, config, tokenizer):
        # Attributes
        # Attributes from config
        self.dataset_path = config.dataset_path
        self.max_uttr_len = config.max_uttr_len
        # Other attributes
        self.tokenizer = tokenizer
        self.statistics = {"n_sents": 0, "n_tokens": 0}

        # Process sentences
        sents = data
        for sent in sents:
            text = sent["text"]
            tokens = self.tokenizer.convert_string_to_tokens(text)[:self.max_uttr_len]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens, bos_and_eos=True)

            sent.update({
                "tokens": tokens,
                "token_ids": token_ids
            })
        self._sents = sents

        # Calculate basic statistics
        self.statistics["n_sents"] = len(sents)
        for sent in sents:
            tokens = sent["text"].split(" ")
            self.statistics["n_tokens"] += len(tokens)

    def epoch_init(self, shuffle=False):
        self.cur_sent_idx = 0
        if shuffle:
            self.sents = copy.deepcopy(self._sents)
            random.shuffle(self.sents)
        else:
            self.sents = self._sents

    def __len__(self):
        return len(self._sents)

    def next(self, batch_size):
        # Return None when running out of sentences
        if self.cur_sent_idx == len(self.sents):
            return None

        # Data to fill in
        X = []
        while self.cur_sent_idx < len(self.sents):
            if len(X) == batch_size:
                break

            sent = self.sents[self.cur_sent_idx]
            self.cur_sent_idx += 1

            X.append(sent["token_ids"])

        X = self.tokenizer.convert_batch_ids_to_tensor(X)
        X = torch.LongTensor(X).to(DEVICE)
        
        batch_data_dict = {
            "X": X
        }

        return batch_data_dict
