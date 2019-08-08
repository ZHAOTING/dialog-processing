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

    def __init__(self, config, tokenizer, dataset):
        ## Attributes
        # Attributes from config
        self.dataset_path = config.dataset_path
        self.max_uttr_len = config.max_uttr_len
        self.history_len = config.history_len
        self.dialog_acts = config.dialog_acts
        # Other attributes
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.num_segments = 0
        self.statistics = {"n_sessions": 0, "n_uttrs": 0, "n_tokens": 0}

        ## Load dataset from json file
        print("Reading {} dataset...".format(dataset))
        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = json.load(f)
        if dataset == "train":
            sessions = self.data["train"]
        elif dataset == "dev":
            sessions = self.data["dev"]
        elif dataset == "test":
            sessions = self.data["test"]

        ## Process sessions
        for sess in sessions:
            for uttr in sess["utterances"]:
                text = uttr["text"]
                floor = uttr["floor"]
                dialog_act = uttr["utterance_meta"]["dialog_act"]
                tokens = self.tokenizer.convert_sent_to_tokens(text)[:self.max_uttr_len]
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens, bos_and_eos=True)
                floor_id = ["A", "B"].index(floor)
                dialog_act_id = self.dialog_acts.index(dialog_act)
                
                uttr.update({
                    "tokens": tokens,
                    "token_ids": token_ids,
                    "floor_id": floor_id,
                    "dialog_act_id": dialog_act_id
                })

        ## Get segments
        self._segments = []
        for sess in sessions:
            uttrs = sess["utterances"]
            for segment_end_idx in range(1, len(uttrs)):
                segment_start_idx = max(0, segment_end_idx-self.history_len)
                segment = {
                    "utterances": uttrs[segment_start_idx:segment_end_idx+1],
                    "segment_meta": {},
                }
                self._segments.append(segment)

        ## Calculate basic statistics
        self.statistics["n_sessions"] = len(sessions)
        for sess in sessions:
            self.statistics["n_uttrs"] += len(sess["utterances"])
            for uttr in sess["utterances"]:
                self.statistics["n_tokens"] += len(tokens)
        print(self.statistics)

    def epoch_init(self, shuffle=True):
        self.cur_segment_idx = 0
        if shuffle:
            self.segments = copy.deepcopy(self._segments)
            random.shuffle(self.segments)
        else:
            self.segments = self._segments

        print("{} datapoints in total in {} dataset".format(len(self.segments), self.dataset))

    def num_batches(self, batch_size):
        return len(self._segments)//batch_size

    def next(self, batch_size):
        ## Return None when running out of segments
        if self.cur_segment_idx == len(self.segments):
            return None

        ## Data to fill in
        X, Y_da = [], []
        X_floor, Y_floor = [], []
        X_da, Y_da = [], []

        while self.cur_segment_idx < len(self.segments):
            if len(Y_da) == batch_size:
                break

            segment = self.segments[self.cur_segment_idx]
            segment_uttrs = segment["utterances"]
            self.cur_segment_idx += 1

            empty_sent = ""
            empty_tokens = self.tokenizer.convert_sent_to_tokens(empty_sent)
            empty_ids = self.tokenizer.convert_tokens_to_ids(empty_tokens, bos_and_eos=True)
            padding_uttr = {
                "tokens": empty_tokens,
                "token_ids": empty_ids,
                "floor_id": 0,
                "dialog_act_id": 0
            }

            ## First non-padding input uttrs
            for uttr in segment_uttrs:
                X.append(uttr["token_ids"])
                X_floor.append(uttr["floor_id"])

            ## Then padding input uttrs
            for _ in range(self.history_len-len(segment_uttrs)+1):
                uttr = padding_uttr
                X.append(uttr["token_ids"])
                X_floor.append(uttr["floor_id"])

            ## Target uttr
            uttr = segment_uttrs[-1]
            Y_da.append(uttr["dialog_act_id"])
            Y_floor.append(uttr["floor_id"])

        X = self.tokenizer.convert_batch_ids_to_tensor(X)

        batch_size = len(Y_da)
        history_len = X.size(0)//batch_size

        X = torch.LongTensor(X).to(DEVICE).view(batch_size, history_len, -1)
        X_floor = torch.LongTensor(X_floor).to(DEVICE).view(batch_size, history_len)
        Y_floor = torch.LongTensor(Y_floor).to(DEVICE)
        Y_da = torch.LongTensor(Y_da).to(DEVICE)

        batch_data_dict = {
            "X": X,
            "X_floor": X_floor,
            "Y_floor": Y_floor, 
            "Y_da": Y_da,
        }

        return batch_data_dict
