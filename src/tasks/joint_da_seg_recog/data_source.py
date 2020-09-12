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

    def __init__(self, data, config, tokenizer, label_tokenizer):
        # Attributes
        # Attributes from config
        self.dataset_path = config.dataset_path
        self.max_uttr_len = config.max_uttr_len
        self.history_len = config.history_len
        # Other attributes
        self.label_tokenizer = label_tokenizer
        self.tokenizer = tokenizer
        self.pad_label_id = label_tokenizer.pad_token_id
        self.bos_label_id = label_tokenizer.bos_token_id
        self.eos_label_id = label_tokenizer.eos_token_id
        self.statistics = {"n_sessions": 0, "n_uttrs": 0, "n_tokens": 0, "n_segments": 0, "n_fragments": 0}

        sessions = data

        # Process sessions
        for sess in sessions:
            sess["processed_utterances"] = []
            for uttr in sess["utterances"]:
                uttr_tokens = []
                uttr_labels = []
                for segment in uttr:
                    text = segment["text"]
                    floor = segment["floor"]
                    dialog_act = segment["segment_meta"]["dialog_act"]
                    tokens = self.tokenizer.convert_string_to_tokens(text)[:self.max_uttr_len]
                    uttr_tokens += tokens
                    uttr_labels += ["I"] * (len(tokens) - 1) + ["E_"+dialog_act]

                uttr_token_ids = self.tokenizer.convert_tokens_to_ids(uttr_tokens, bos_and_eos=True)
                uttr_label_ids = [self.bos_label_id] + \
                    [self.label_tokenizer.word2id[label] for label in uttr_labels] + \
                    [self.eos_label_id]
                uttr_floor_id = ["A", "B"].index(floor)
                
                sess["processed_utterances"].append({
                    "token_ids": uttr_token_ids,
                    "label_ids": uttr_label_ids,
                    "floor_id": uttr_floor_id
                })  

        # Get segments
        self._fragments = []
        for sess in sessions:
            uttrs = sess["processed_utterances"]
            for uttr_end_idx in range(0, len(uttrs)):
                uttr_start_idx = max(0, uttr_end_idx-self.history_len+1)
                fragment = {
                    "utterances": uttrs[uttr_start_idx:uttr_end_idx+1],
                }
                self._fragments.append(fragment)

        # Calculate basic statistics
        self.statistics["n_sessions"] = len(sessions)
        self.statistics["n_fragments"] = len(self._fragments)
        for sess in sessions:
            self.statistics["n_uttrs"] += len(sess["utterances"])
            for uttr in sess["utterances"]:
                self.statistics["n_segments"] += len(uttr)
                for segment in uttr:
                    tokens = segment["text"].split(" ")
                    self.statistics["n_tokens"] += len(tokens)

    def epoch_init(self, shuffle=False):
        self.cur_fragment_idx = 0
        if shuffle:
            self.fragments = copy.deepcopy(self._fragments)
            random.shuffle(self.fragments)
        else:
            self.fragments = self._fragments

    def __len__(self):
        return len(self._fragments)

    def next(self, batch_size):
        # Return None when running out of segments
        if self.cur_fragment_idx == len(self.fragments):
            return None

        # Data to fill in
        X, Y = [], []
        X_floor = []

        empty_sent = ""
        empty_tokens = self.tokenizer.convert_string_to_tokens(empty_sent)
        empty_ids = self.tokenizer.convert_tokens_to_ids(empty_tokens, bos_and_eos=False)
        padding_segment = {
            "tokens": empty_tokens,
            "token_ids": empty_ids,
            "floor_id": 0,
            "label_ids": [self.pad_label_id] * len(empty_ids)
        }

        while self.cur_fragment_idx < len(self.fragments):
            if len(Y) == batch_size:
                break

            fragment = self.fragments[self.cur_fragment_idx]
            segments = fragment["utterances"]
            self.cur_fragment_idx += 1

            # First non-padding segments
            for segment in segments:
                X.append(segment["token_ids"])
                X_floor.append(segment["floor_id"])
            
            segment = segments[-1]
            Y.append(segment["label_ids"])

            # Then padding segments
            for _ in range(self.history_len-len(segments)):
                segment = padding_segment
                X.append(segment["token_ids"])
                X_floor.append(segment["floor_id"])

        X = self.tokenizer.convert_batch_ids_to_tensor(X)
        max_segment_len = X.size(1)
        Y = [y + [self.pad_label_id]*(max_segment_len-len(y)) for y in Y]

        batch_size = len(Y)
        history_len = X.size(0)//batch_size

        X = torch.LongTensor(X).to(DEVICE).view(batch_size, history_len, -1)
        X_floor = torch.LongTensor(X_floor).to(DEVICE).view(batch_size, history_len)
        Y = torch.LongTensor(Y).to(DEVICE).view(batch_size, -1)

        batch_data_dict = {
            "X": X,
            "X_floor": X_floor,
            "Y": Y
        }

        return batch_data_dict
