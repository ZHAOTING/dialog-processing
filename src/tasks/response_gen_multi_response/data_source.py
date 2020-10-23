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

    def __init__(self, data, config, tokenizer, hyp_source_model_names=[], use_ground_truth=True, use_flattened_hyps=False, use_nested_hyps=False, n_nested_hyps=0, deduplicate_hyps=False):
        # Attributes
        # Attributes from config
        self.dataset_path = config.dataset_path
        self.max_uttr_len = config.max_uttr_len
        self.history_len = config.history_len
        self.n_hyps = config.n_hyps if hasattr(config, "n_hyps") else 0
        # Other attributes
        self.tokenizer = tokenizer
        self.statistics = {"n_sessions": 0, "n_uttrs": 0, "n_tokens": 0, "n_segments": 0}
        self.use_ground_truth = use_ground_truth
        self.use_flattened_hyps = use_flattened_hyps
        self.use_nested_hyps = use_nested_hyps
        self.n_nested_hyps = n_nested_hyps

        assert (not self.use_flattened_hyps) or (not self.use_nested_hyps) is True

        sessions = data

        # Process sessions
        for sess in sessions:
            for uttr in sess["utterances"]:
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

                if "model2hypothesis" in uttr["utterance_meta"]:
                    uttr["utterance_meta"]["hypotheses"] = []
                    for model_name, hyps in uttr["utterance_meta"]["model2hypothesis"].items():
                        if model_name not in hyp_source_model_names:
                            continue
                        for hyp in hyps:
                            hyp_text = hyp["text"]
                            tokens = self.tokenizer.convert_string_to_tokens(hyp_text)[:self.max_uttr_len]
                            token_ids = self.tokenizer.convert_tokens_to_ids(tokens, bos_and_eos=True)
                            hyp["token_ids"] = token_ids
                            hyp["floor_id"] = uttr["floor_id"]
                            uttr["utterance_meta"]["hypotheses"].append(hyp)
                    uttr["utterance_meta"]["hypotheses"] = uttr["utterance_meta"]["hypotheses"][:self.n_hyps]

                    hyps_to_use = [hyp for hyp in uttr["utterance_meta"]["hypotheses"]]
                    if self.use_ground_truth:
                        hyps_to_use.append(uttr)
                    assert len(hyps_to_use) >= self.n_nested_hyps

                    if deduplicate_hyps:
                        deduplicated = []
                        added_texts = set()
                        for hyp in hyps_to_use:
                            if hyp["text"] in added_texts:
                                continue
                            else:
                                deduplicated.append(hyp)
                                added_texts.add(hyp["text"])
                        hyps_to_use = deduplicated

                    hyp_prob_sum = 0
                    for hyp in hyps_to_use:
                        if "sentence_prob" in hyp["utterance_meta"]:
                            hyp_prob = hyp["utterance_meta"]["sentence_prob"] + 1e-10
                            hyp_prob_sum += hyp_prob
                    for hyp in hyps_to_use:
                        if "sentence_prob" in hyp["utterance_meta"]:
                            hyp_prob = hyp["utterance_meta"]["sentence_prob"] + 1e-10
                            hyp_weight = len(hyps_to_use)*hyp_prob/hyp_prob_sum
                            hyp["utterance_meta"]["sentence_weight"] = hyp_weight

                    uttr["utterance_meta"]["hypotheses"] = hyps_to_use

        # code.interact(local=locals())

        # Get segments
        self._segments = []
        for sess in sessions:
            uttrs = sess["utterances"]
            for segment_end_idx in range(1, len(uttrs)):
                segment_start_idx = max(0, segment_end_idx-self.history_len)

                if self.use_flattened_hyps:
                    ctx_start_idx = segment_start_idx
                    ctx_end_idx = segment_end_idx-1
                    response = uttrs[segment_end_idx]
                    for hyp in response["utterance_meta"]["hypotheses"]:
                        segment_uttrs = [
                            {
                                "token_ids": uttr["token_ids"],
                                "floor_id": uttr["floor_id"],
                                "utterance_meta": {
                                    "sentence_id": uttr["utterance_meta"]["sentence_id"],
                                    "sentence_weight": uttr["utterance_meta"].get("sentence_weight", None)
                                }
                            } for uttr in uttrs[ctx_start_idx:ctx_end_idx+1] + [hyp]
                        ]
                        segment = {
                            "utterances": segment_uttrs,
                            "segment_meta": sess["dialog_meta"]
                        }
                        self._segments.append(segment)

                    if self.use_ground_truth:
                        segment_uttrs = [
                            {
                                "token_ids": uttr["token_ids"],
                                "floor_id": uttr["floor_id"],
                                "utterance_meta": {
                                    "sentence_id": uttr["utterance_meta"]["sentence_id"],
                                    "sentence_weight": uttr["utterance_meta"].get("sentence_weight", None)
                                }
                            } for uttr in uttrs[segment_start_idx:segment_end_idx+1]
                        ]
                        segment = {
                            "utterances": segment_uttrs,
                            "segment_meta": sess["dialog_meta"]
                        }
                        self._segments.append(segment)
                elif self.use_nested_hyps:
                    segment_uttrs = [
                        {
                            "token_ids": uttr["token_ids"],
                            "floor_id": uttr["floor_id"],
                            "utterance_meta": {
                                "sentence_id": uttr["utterance_meta"]["sentence_id"]
                            }
                        } for uttr in uttrs[segment_start_idx:segment_end_idx]
                    ]
                    response = uttrs[segment_end_idx]
                    hypotheses = []
                    if self.use_ground_truth:
                        hypotheses.append({
                            "token_ids": response["token_ids"]
                        })
                    if "hypotheses" in response["utterance_meta"]:
                        for hyp in response["utterance_meta"]["hypotheses"]:
                            hypotheses.append({
                                "token_ids": hyp["token_ids"]
                            })
                    segment_uttrs.append(
                        {
                            "token_ids": response["token_ids"],
                            "floor_id": response["floor_id"],
                            "utterance_meta": {
                                "sentence_id": response["utterance_meta"]["sentence_id"],
                                "hypotheses": hypotheses
                            }
                        }
                    )
                    segment = {
                        "utterances": segment_uttrs,
                        "segment_meta": sess["dialog_meta"]
                    }
                    self._segments.append(segment)
                else:
                    segment_uttrs = [
                        {
                            "token_ids": uttr["token_ids"],
                            "floor_id": uttr["floor_id"],
                            "utterance_meta": {
                                "sentence_id": uttr["utterance_meta"]["sentence_id"]
                            }
                        } for uttr in uttrs[segment_start_idx:segment_end_idx+1]
                    ]
                    segment = {
                        "utterances": segment_uttrs,
                        "segment_meta": sess["dialog_meta"]
                    }
                    self._segments.append(segment)

        # Calculate basic statistics
        self.statistics["n_sessions"] = len(sessions)
        self.statistics["n_segments"] = len(self._segments)
        for sess in sessions:
            self.statistics["n_uttrs"] += len(sess["utterances"])
            for uttr in sess["utterances"]:
                tokens = uttr["text"].split(" ")
                self.statistics["n_tokens"] += len(tokens)

    def epoch_init(self, shuffle=False):
        self.cur_segment_idx = 0
        if shuffle:
            random.shuffle(self._segments)
        self.segments = self._segments

    def __len__(self):
        return len(self._segments)

    def next(self, batch_size):
        # Return None when running out of segments
        if self.cur_segment_idx == len(self.segments):
            return None

        # Data to fill in
        X, Y = [], []
        X_floor, Y_floor = [], []
        Y_id = []
        Y_weight = []
        if self.use_nested_hyps:
            Y_nested = []

        empty_sent = ""
        empty_tokens = self.tokenizer.convert_string_to_tokens(empty_sent)
        empty_ids = self.tokenizer.convert_tokens_to_ids(empty_tokens, bos_and_eos=True)
        padding_uttr = {
            "tokens": empty_tokens,
            "token_ids": empty_ids,
            "floor_id": 0,
        }

        while self.cur_segment_idx < len(self.segments):
            if len(Y) == batch_size:
                break

            segment = self.segments[self.cur_segment_idx]
            segment_uttrs = segment["utterances"]
            self.cur_segment_idx += 1

            # First non-padding input uttrs
            for uttr in segment_uttrs[:-1]:
                X.append(uttr["token_ids"])
                X_floor.append(uttr["floor_id"])

            # Then padding input uttrs
            for _ in range(self.history_len-len(segment_uttrs)+1):
                uttr = padding_uttr
                X.append(uttr["token_ids"])
                X_floor.append(uttr["floor_id"])

            # Last output uttr
            uttr = segment_uttrs[-1]
            Y.append(uttr["token_ids"])
            Y_floor.append(uttr["floor_id"])
            Y_id.append(uttr["utterance_meta"]["sentence_id"])
            if self.use_nested_hyps:
                uttr = segment_uttrs[-1]

                hyps = uttr["utterance_meta"]["hypotheses"]
                random.shuffle(hyps)
                for hyp in hyps[:self.n_nested_hyps]:
                    Y_nested.append(hyp["token_ids"])
            if "sentence_weight" in uttr["utterance_meta"]:
                if uttr["utterance_meta"]["sentence_weight"] is not None:
                    Y_weight.append(uttr["utterance_meta"]["sentence_weight"])

        X = self.tokenizer.convert_batch_ids_to_tensor(X)
        Y = self.tokenizer.convert_batch_ids_to_tensor(Y)

        batch_size = Y.size(0)
        history_len = X.size(0)//batch_size

        X = torch.LongTensor(X).to(DEVICE).view(batch_size, history_len, -1)
        X_floor = torch.LongTensor(X_floor).to(DEVICE).view(batch_size, history_len)

        Y = torch.LongTensor(Y).to(DEVICE)
        Y_floor = torch.LongTensor(Y_floor).to(DEVICE)

        if len(Y_weight) == Y.size(0):
            Y_weight = torch.FloatTensor(Y_weight).to(DEVICE)

        if self.use_nested_hyps:
            Y_nested = self.tokenizer.convert_batch_ids_to_tensor(Y_nested)
            Y_nested = torch.LongTensor(Y_nested).to(DEVICE).view(batch_size, self.n_nested_hyps, -1)

        batch_data_dict = {
            "X": X,
            "X_floor": X_floor,
            "Y": Y,
            "Y_floor": Y_floor,
            "Y_id": Y_id,
            "Y_weight": Y_weight
        }

        if self.use_nested_hyps:
            batch_data_dict.update({
                "Y_nested": Y_nested,
            })

        return batch_data_dict
