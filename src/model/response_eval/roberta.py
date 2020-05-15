import re
import code
import math
import json
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.modules.utils import init_module_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Roberta(nn.Module):
    def __init__(self, config, tokenizer):
        super(Roberta, self).__init__()

        # Attributes
        # Other attributes
        self.tokenizer = tokenizer
        self.sep_token_id = tokenizer.sep_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.speaker1_token_id = tokenizer.speaker1_token_id
        self.speaker2_token_id = tokenizer.speaker2_token_id

        # Load pretrained gpt2 model
        model_size = config.model_size
        assert model_size in ["base", "large", "large-mnli"]
        from transformers import RobertaForSequenceClassification
        pretrained = RobertaForSequenceClassification.from_pretrained(f'roberta-{model_size}')

        pretrained.resize_token_embeddings(len(tokenizer))
        self.roberta = pretrained.roberta

        # some model-relying attributes
        self.hidden_dim = self.roberta.config.hidden_size
        self.hidden_dropout_prob = self.roberta.config.hidden_dropout_prob

        # Embedding componenets
        self.embeddings = self.roberta.embeddings

        # Output layer
        # regressor for unsupervised training of is-next-sentence prediction
        self.regressor_pipe = nn.ModuleList(
            [
                nn.Dropout(self.hidden_dropout_prob),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Dropout(self.hidden_dropout_prob),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            ]
        )

        # Initialization
        self._init_weights()

    def _init_weights(self):
        init_module_weights(self.regressor_pipe)

    def _construct_input_output(self, inputs, input_floors, outputs, output_floors):
        """Convert a list of sentences into a single sequence for each dialog
        """

        # Use lists instead of tensors to speed up
        input_lens = (inputs != self.pad_token_id).sum(-1)
        dial_lens = (input_lens > 0).sum(dim=1).tolist()
        inputs = inputs.tolist()
        input_lens = input_lens.tolist()
        input_floors = input_floors.tolist()
        output_floors = output_floors.tolist()
        output_lens = (outputs != self.pad_token_id).sum(-1)
        outputs = outputs.tolist()
        output_lens = output_lens.tolist()

        # build sequences
        input_token_id_seqs = []
        for dial_idx in range(len(inputs)):
            # merge history sentences in context (sentences except for the last sentence)
            ctx_input_ids = []
            for sent_idx in range(dial_lens[dial_idx]):
                sent_len = input_lens[dial_idx][sent_idx]
                sent_token_ids = inputs[dial_idx][sent_idx][:sent_len]

                src_speaker = input_floors[dial_idx][sent_idx]
                tgt_speaker = output_floors[dial_idx]
                speaker_token_id = self.speaker1_token_id if src_speaker == tgt_speaker else self.speaker2_token_id

                ctx_input_ids += ([speaker_token_id] + sent_token_ids)

            # response token ids (the last sentence)
            output_len = output_lens[dial_idx]
            output_token_ids = outputs[dial_idx][:output_len]
            response_input_ids = [self.speaker1_token_id] + output_token_ids

            # concat context and response
            input_token_id_seq = [self.cls_token_id] + ctx_input_ids + [self.sep_token_id] + [self.sep_token_id] + response_input_ids + [self.sep_token_id]
            input_token_id_seqs.append(input_token_id_seq)

        # Get lengths
        seq_lens = [len(seq) for seq in input_token_id_seqs]
        max_seq_len = max(seq_lens)

        # Get attention masks
        attention_masks = [[1]*len(seq) + [0]*(max_seq_len-len(seq)) for seq in input_token_id_seqs]

        # Pad sequences and produce tensors
        input_token_id_seqs = [seq + [self.pad_token_id]*(max_seq_len-len(seq)) for seq in input_token_id_seqs]
        input_token_id_seqs = torch.LongTensor(input_token_id_seqs).to(DEVICE)
        attention_masks = torch.LongTensor(attention_masks).to(DEVICE)

        return {
            "input_ids": input_token_id_seqs,
            "seq_lens": seq_lens,
            "attention_masks": attention_masks,
        }

    def _compute_is_next_prob(self, input_ids, attention_masks):
        # forward Roberta
        outputs = self.roberta(input_ids, attention_mask=attention_masks)
        sequence_output = outputs[0]

        # compute probability
        cls_hidden = sequence_output[:, 0, :]
        x = cls_hidden
        for m in self.regressor_pipe:
            x = m(x)
        probs = x.view(-1)

        return probs

    def _compute_scores(self, input_ids, attention_masks):
        # forward Roberta
        outputs = self.roberta(input_ids, attention_mask=attention_masks)
        sequence_output = outputs[0]

        # compute score
        cls_hidden = sequence_output[:, 0, :]
        x = cls_hidden
        for m in self.regressor_pipe:
            x = m(x)
        probs = x.view(-1)

        # scaling probability to score
        scores = probs*4 + 1

        return scores

    def load_model(self, model_path):
        """Load pretrained model weights from model_path

        Arguments:
            model_path {str} -- path to pretrained model weights
        """
        pretrained_state_dict = torch.load(
            model_path,
            map_location=lambda storage, loc: storage
        )

        own_state_dict = self.state_dict()
        num_loaded_params = 0
        for name, param in pretrained_state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].data = param.data
                num_params = 1
                for dim in param.size():
                    num_params *= dim
                num_loaded_params += num_params

        self.load_state_dict(own_state_dict)

    def unsupervised_train_step(self, data):
        """One training step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of context sentences
                'Y' {LongTensor [batch_size, max_y_sent_len]} -- token ids of response sentence
                'Y_floor' {LongTensor [batch_size]} -- floor of response sentence
                'Y_is_next' {BoolTensor [batch_size]} -- is_next label (is_next: True, not_next: False) of response sentence

            lr {float} -- learning rate

        Returns:
            dict of data -- returned keys and values
                'loss' {FloatTensor []} -- loss to backword
            dict of statistics -- returned keys and values
                'loss' {float} -- batch loss
        """
        X, Y = data["X"], data["Y"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_is_next = data["Y_is_next"]

        batch_size = X.size(0)

        # construct inputs and outputs
        data_dict = self._construct_input_output(
            inputs=X,
            input_floors=X_floor,
            outputs=Y,
            output_floors=Y_floor,
        )
        input_ids = data_dict["input_ids"]
        attention_masks = data_dict["attention_masks"]

        # forward
        probs = self._compute_is_next_prob(
            input_ids=input_ids,
            attention_masks=attention_masks
        )

        # loss
        loss = F.mse_loss(
            probs,
            Y_is_next.float(),
            reduction="mean"
        )

        # return dicts
        ret_data = {
            "loss": loss
        }
        ret_stat = {
            "loss": loss.item()
        }

        return ret_data, ret_stat

    def unsupervised_evaluate_step(self, data):
        """One evaluation step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of context sentences
                'Y' {LongTensor [batch_size, max_y_sent_len]} -- token ids of response sentence
                'Y_floor' {LongTensor [batch_size]} -- floor of response sentence
                'Y_is_next' {BoolTensor [batch_size]} -- is_next label (is_next: True, not_next: False) of response sentence

        Returns:
            dict of data -- returned keys and values
                'probs' {FloatTensor [batch_size]} -- probablities of Y being the next sentence
            dict of statistics -- returned keys and values
                'monitor' {float} -- a monitor number for learning rate scheduling
        """
        X, Y = data["X"], data["Y"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_is_next = data["Y_is_next"]

        batch_size = X.size(0)

        with torch.no_grad():
            # construct inputs and outputs
            data_dict = self._construct_input_output(
                inputs=X,
                input_floors=X_floor,
                outputs=Y,
                output_floors=Y_floor,
            )
            input_ids = data_dict["input_ids"]
            attention_masks = data_dict["attention_masks"]

            # forward
            probs = self._compute_is_next_prob(
                input_ids=input_ids,
                attention_masks=attention_masks
            )

            # loss
            loss = F.mse_loss(
                probs,
                Y_is_next.float(),
                reduction="mean"
            )

        # return dicts
        ret_data = {
            "probs": probs,
        }
        ret_stat = {
            "monitor": loss.item()
        }

        return ret_data, ret_stat

    def unsupervised_test_step(self, data):
        """One test step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of context sentences
                'Y' {LongTensor [batch_size, max_y_sent_len]} -- token ids of response sentence
                'Y_floor' {LongTensor [batch_size]} -- floor of response sentence

        Returns:
            dict of data -- returned keys and values
                'probs' {FloatTensor [batch_size]} -- probablities of Y being the next sentence
            dict of statistics -- returned keys and values
        """
        X, Y = data["X"], data["Y"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]

        with torch.no_grad():
            # construct inputs and outputs
            data_dict = self._construct_input_output(
                inputs=X,
                input_floors=X_floor,
                outputs=Y,
                output_floors=Y_floor,
            )
            input_ids = data_dict["input_ids"]
            attention_masks = data_dict["attention_masks"]

            # forward
            probs = self._compute_is_next_prob(
                input_ids=input_ids,
                attention_masks=attention_masks
            )

        ret_data = {
            "probs": probs,
        }
        ret_stat = {}

        return ret_data, ret_stat

    def supervised_train_step(self, data):
        """One training step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of context sentences
                'Y' {LongTensor [batch_size, max_y_sent_len]} -- token ids of response sentence
                'Y_floor' {LongTensor [batch_size]} -- floor of response sentence
                'Y_tgt_score' {FloatTensor [batch_size]} -- target score of response sentence

        Returns:
            dict of data -- returned keys and values
                'loss' {FloatTensor []} -- loss to backword
            dict of statistics -- returned keys and values
                'loss' {float} -- batch loss
        """
        X, Y = data["X"], data["Y"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_tgt_score = data["Y_tgt_score"]

        batch_size = X.size(0)

        # construct inputs and outputs
        data_dict = self._construct_input_output(
            inputs=X,
            input_floors=X_floor,
            outputs=Y,
            output_floors=Y_floor,
        )
        input_ids = data_dict["input_ids"]
        attention_masks = data_dict["attention_masks"]

        # forward
        scores = self._compute_scores(
            input_ids=input_ids,
            attention_masks=attention_masks
        )

        # loss
        loss = F.mse_loss(
            scores,
            Y_tgt_score,
            reduction="mean"
        )

        # return dicts
        ret_data = {
            "loss": loss
        }
        ret_stat = {
            "loss": loss.item()
        }

        return ret_data, ret_stat

    def supervised_evaluate_step(self, data):
        """One evaluation step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of context sentences
                'Y' {LongTensor [batch_size, max_y_sent_len]} -- token ids of response sentence
                'Y_floor' {LongTensor [batch_size]} -- floor of response sentence
                'Y_tgt_score' {FloatTensor [batch_size]} -- target score of response sentence

        Returns:
            dict of data -- returned keys and values

            dict of statistics -- returned keys and values
                'monitor' {float} -- a monitor number for learning rate scheduling
        """
        X, Y = data["X"], data["Y"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_tgt_score = data["Y_tgt_score"]

        batch_size = X.size(0)

        with torch.no_grad():
            # construct inputs and outputs
            data_dict = self._construct_input_output(
                inputs=X,
                input_floors=X_floor,
                outputs=Y,
                output_floors=Y_floor,
            )
            input_ids = data_dict["input_ids"]
            attention_masks = data_dict["attention_masks"]

            # forward
            scores = self._compute_scores(
                input_ids=input_ids,
                attention_masks=attention_masks
            )

            # loss
            loss = F.mse_loss(
                scores,
                Y_tgt_score,
                reduction="mean"
            )

        # return dicts
        ret_data = {}
        ret_stat = {
            "monitor": loss.item()
        }

        return ret_data, ret_stat

    def supervised_test_step(self, data):
        """One test step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of context sentences
                'Y' {LongTensor [batch_size, max_y_sent_len]} -- token ids of response sentence
                'Y_floor' {LongTensor [batch_size]} -- floor of response sentence

        Returns:
            dict of data -- returned keys and values
                'scores' {FloatTensor [batch_size]} -- predicted score of response
            dict of statistics -- returned keys and values

        """
        X, Y = data["X"], data["Y"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]

        with torch.no_grad():
            # construct inputs and outputs
            data_dict = self._construct_input_output(
                inputs=X,
                input_floors=X_floor,
                outputs=Y,
                output_floors=Y_floor
            )
            input_ids = data_dict["input_ids"]
            attention_masks = data_dict["attention_masks"]

            # forward
            scores = self._compute_scores(
                input_ids=input_ids,
                attention_masks=attention_masks
            )

        ret_data = {
            "scores": scores
        }
        ret_stat = {}

        return ret_data, ret_stat

    def predict(self, batch_ctx, batch_hyp):
        """Predict a batch of data

        Arguments:
            batch_ctx {list batch_size*[history_len*(text, floor)]} -- context utterances and floors
            batch_hyp {list batch_size*[(text, floor)]} -- response utterances and floors

        Returns:
            list of scores {list of int [batch_size]} - scores of input (context, hypothesis) pairs (score scale: 1-5)

        Example:
            inputs:
                batch_ctx - [
                    [
                        ["hello there .", "A"],
                        ["hi .", "B"],
                    ],
                    [
                        ["it has been a terrible year .", "B"],
                        ["any plan after graduation ?", "A"],
                    ],
                    [
                        ["hi, how much is the guitar ?", "A"],
                        ["it only takes five thousand bucks .", "B"],
                        ["that is a lot !", "A"]
                    ]
                ]

                batch_hyp - [
                    ["how is it going ?", "A"],
                    ["not really .", "B"],
                    ["that is funny .", "B"]
                ]

            outputs:
                scores - [
                    4.38,
                    4.56,
                    2.64
                ]
        """
        input_token_id_seqs = []
        for ctx, hyp in zip(batch_ctx, batch_hyp):
            hyp_text, hyp_floor = hyp
            assert hyp_floor in ["A", "B"]

            ctx_input_ids = []
            for text, floor in ctx:
                assert floor in ["A", "B"]
                tokens = self.tokenizer.convert_string_to_tokens(text)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens, bos_and_eos=True)
                speaker_token_id = self.speaker1_token_id if floor == hyp_floor else self.speaker2_token_id
                ctx_input_ids += ([speaker_token_id] + token_ids)

            tokens = self.tokenizer.convert_string_to_tokens(hyp_text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens, bos_and_eos=True)
            response_input_ids = [self.speaker1_token_id] + token_ids

            input_id_seq = [self.cls_token_id] + ctx_input_ids + [self.sep_token_id] + [self.sep_token_id] + response_input_ids + [self.sep_token_id]
            input_token_id_seqs.append(input_id_seq)

        # Get lengths
        seq_lens = [len(seq) for seq in input_token_id_seqs]
        max_seq_len = max(seq_lens)

        # Get attention masks
        attention_masks = [[1]*len(seq) + [0]*(max_seq_len-len(seq)) for seq in input_token_id_seqs]

        # Pad sequences and produce tensors
        input_token_id_seqs = [seq + [self.pad_token_id]*(max_seq_len-len(seq)) for seq in input_token_id_seqs]
        input_token_id_seqs = torch.LongTensor(input_token_id_seqs).to(DEVICE)
        attention_masks = torch.LongTensor(attention_masks).to(DEVICE)

        with torch.no_grad():
            scores = self._compute_scores(input_token_id_seqs, attention_masks)

        return scores.tolist()
