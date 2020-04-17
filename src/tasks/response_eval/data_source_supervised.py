import copy
import collections
import math
import random
import json
import code

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataSourceSupervised():

    def __init__(self, data, config, tokenizer):
        # Attributes
        # Attributes from config
        self.dataset_path = config.dataset_path
        self.max_uttr_len = config.max_uttr_len
        self.history_len = config.history_len
        self.human_score_names = config.human_score_names
        self.target_score_name = config.target_score_name
        # Other attributes
        self.tokenizer = tokenizer
        self.statistics = {"n_segments": 0, "n_uttrs": 0, "n_tokens": 0}

        dialogs = data

        # Process dialogs
        for dialog in dialogs:
            for uttr in dialog["utterances"]:
                text = uttr["text"]
                floor = uttr["floor"]
                tokens = self.tokenizer.convert_string_to_tokens(text)[:self.max_uttr_len]
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens, bos_and_eos=True)
                floor_id = ["A", "B"].index(floor)

                uttr.update({
                    "tokens": tokens,
                    "token_ids": token_ids,
                    "floor_id": floor_id,
                })

                if "utterance_meta" in uttr:
                    if "reference_text" in uttr["utterance_meta"]:
                        ref_text = uttr["utterance_meta"]["reference_text"]
                        ref_tokens = self.tokenizer.convert_string_to_tokens(ref_text)[:self.max_uttr_len]
                        ref_token_ids = self.tokenizer.convert_tokens_to_ids(ref_tokens, bos_and_eos=True)
                        uttr.update({
                            "ref_tokens": ref_tokens,
                            "ref_token_ids": ref_token_ids
                        })
        self._segments = dialogs

        # Calculate basic statistics
        self.statistics["n_segments"] = len(self._segments)
        for dialog in self._segments:
            self.statistics["n_uttrs"] += len(dialog["utterances"])
            for uttr in dialog["utterances"]:
                tokens = uttr["text"].split(" ")
                self.statistics["n_tokens"] += len(tokens)

    def epoch_init(self, shuffle=False):
        self.cur_dialog_idx = 0
        if shuffle:
            self.dialogs = copy.deepcopy(self._segments)
            random.shuffle(self.dialogs)
        else:
            self.dialogs = self._segments

    def __len__(self):
        return len(self._segments)

    def next(self, batch_size):
        # Return None when running out of dialogs
        if self.cur_dialog_idx == len(self.dialogs):
            return None

        # Data to fill in
        X, Y, Y_ref = [], [], []
        X_floor, Y_floor = [], []
        Y_score = []
        Y_tgt_score = []
        Y_meta_dict = []
        D_meta_dict = []

        empty_sent = ""
        empty_tokens = self.tokenizer.convert_string_to_tokens(empty_sent)
        empty_ids = self.tokenizer.convert_tokens_to_ids(empty_tokens, bos_and_eos=True)
        padding_uttr = {
            "tokens": empty_tokens,
            "token_ids": empty_ids,
            "floor_id": 0,
        }

        while self.cur_dialog_idx < len(self.dialogs):
            if len(Y) == batch_size:
                break

            dialog = self.dialogs[self.cur_dialog_idx]
            uttrs = dialog["utterances"]
            self.cur_dialog_idx += 1

            # First non-padding input uttrs
            for uttr in uttrs[:-1]:
                X.append(uttr["token_ids"])
                X_floor.append(uttr["floor_id"])

            # Then padding input uttrs
            for _ in range(self.history_len-len(uttrs)+1):
                uttr = padding_uttr
                X.append(uttr["token_ids"])
                X_floor.append(uttr["floor_id"])

            # Last output uttr
            uttr = uttrs[-1]
            Y.append(uttr["token_ids"])
            Y_ref.append(uttr["ref_token_ids"])
            Y_floor.append(uttr["floor_id"])

            # Get scores
            scores = []
            for score_name in self.human_score_names:
                score = uttr["utterance_meta"]["human_scores"][score_name]
                scores.append(score)
                if score_name == self.target_score_name:
                    Y_tgt_score.append(score)
            Y_score.append(scores)

            # Get meta information dictionary
            Y_meta_dict.append(uttr["utterance_meta"])
            D_meta_dict.append(dialog["dialog_meta"])

        X = self.tokenizer.convert_batch_ids_to_tensor(X)
        Y = self.tokenizer.convert_batch_ids_to_tensor(Y)
        Y_ref = self.tokenizer.convert_batch_ids_to_tensor(Y_ref)

        batch_size = Y.size(0)
        history_len = X.size(0)//batch_size

        X = torch.LongTensor(X).to(DEVICE).view(batch_size, history_len, -1)
        X_floor = torch.LongTensor(X_floor).to(DEVICE).view(batch_size, history_len)

        Y = torch.LongTensor(Y).to(DEVICE)
        Y_ref = torch.LongTensor(Y_ref).to(DEVICE)
        Y_floor = torch.LongTensor(Y_floor).to(DEVICE)
        Y_score = torch.FloatTensor(Y_score).to(DEVICE)
        Y_tgt_score = torch.FloatTensor(Y_tgt_score).to(DEVICE)

        batch_data_dict = {
            "X": X,
            "X_floor": X_floor,
            "Y": Y,
            "Y_ref": Y_ref,
            "Y_floor": Y_floor,
            "Y_score": Y_score,
            "Y_tgt_score": Y_tgt_score,
            "Y_meta_dict": Y_meta_dict,
            "D_meta_dict": D_meta_dict
        }

        return batch_data_dict
