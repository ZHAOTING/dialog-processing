import re
import code
import math
import json
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.modules.encoders import EncoderRNN
from model.modules.submodules import AbsFloorEmbEncoder, RelFloorEmbEncoder
from model.modules.utils import init_module_weights, init_word_embedding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RUBER(nn.Module):
    def __init__(self, config, tokenizer):
        super(RUBER, self).__init__()

        # Attributes
        # Attributes from config
        self.word_embedding_dim = config.word_embedding_dim
        self.attr_embedding_dim = config.attr_embedding_dim
        self.sent_encoder_hidden_dim = config.sent_encoder_hidden_dim
        self.n_sent_encoder_layers = config.n_sent_encoder_layers
        self.dial_encoder_hidden_dim = config.dial_encoder_hidden_dim
        self.n_dial_encoder_layers = config.n_dial_encoder_layers
        self.rnn_type = config.rnn_type
        self.word_embedding_path = config.word_embedding_path
        self.floor_encoder_type = config.floor_encoder
        self.metric = config.metric_type
        # Optional attributes from config
        self.dropout = config.dropout if hasattr(config, "dropout") else 0.0
        self.use_pretrained_word_embedding = config.use_pretrained_word_embedding if hasattr(config, "use_pretrained_word_embedding") else False
        # Other attributes
        self.id2word = tokenizer.id2word
        self.vocab_size = len(tokenizer.word2id)
        self.pad_token_id = tokenizer.pad_token_id
        self.margin = 0.5

        # Input components
        self.word_embedding = nn.Embedding(
            self.vocab_size,
            self.word_embedding_dim,
            padding_idx=self.pad_token_id,
            _weight=init_word_embedding(
                load_pretrained_word_embedding=self.use_pretrained_word_embedding,
                pretrained_word_embedding_path=self.word_embedding_path,
                id2word=self.id2word,
                word_embedding_dim=self.word_embedding_dim,
                vocab_size=self.vocab_size,
                pad_token_id=self.pad_token_id
            ),
        )

        # Encoding components
        self.sent_encoder = EncoderRNN(
            input_dim=self.word_embedding_dim,
            hidden_dim=self.sent_encoder_hidden_dim,
            n_layers=self.n_sent_encoder_layers,
            dropout_emb=self.dropout,
            dropout_input=self.dropout,
            dropout_hidden=self.dropout,
            dropout_output=self.dropout,
            bidirectional=True,
            embedding=self.word_embedding,
            rnn_type=self.rnn_type,
        )
        self.dial_encoder = EncoderRNN(
            input_dim=self.sent_encoder_hidden_dim,
            hidden_dim=self.dial_encoder_hidden_dim,
            n_layers=self.n_dial_encoder_layers,
            dropout_emb=self.dropout,
            dropout_input=self.dropout,
            dropout_hidden=self.dropout,
            dropout_output=self.dropout,
            bidirectional=True,
            rnn_type=self.rnn_type,
        )

        # Output components
        # regressor for unsupervised training of is-next-sentence prediction
        self.M = nn.Parameter(
            torch.FloatTensor(self.dial_encoder_hidden_dim, self.sent_encoder_hidden_dim)
        )
        self.unref_fc = nn.ModuleList(
            [
                nn.Dropout(self.dropout),
                nn.Linear(self.dial_encoder_hidden_dim+self.sent_encoder_hidden_dim+1, self.dial_encoder_hidden_dim),
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.dial_encoder_hidden_dim, 1),
                nn.Sigmoid()
            ]
        )

        # Extra components
        # floor encoding
        if self.floor_encoder_type == "abs":
            self.floor_encoder = AbsFloorEmbEncoder(
                input_dim=self.sent_encoder_hidden_dim,
                embedding_dim=self.attr_embedding_dim
            )
        elif self.floor_encoder_type == "rel":
            self.floor_encoder = RelFloorEmbEncoder(
                input_dim=self.sent_encoder_hidden_dim,
                embedding_dim=self.attr_embedding_dim
            )
        else:
            self.floor_encoder = None

        # Initialization
        self._init_weights()

    def _init_weights(self):
        init_module_weights(self.unref_fc)

    def _encode_response(self, inputs):
        input_lens = (inputs != self.pad_token_id).sum(-1)
        _, _, hyp_sent_encodings = self.sent_encoder(inputs, input_lens)  # [batch_size, sent_encoder_dim]

        return hyp_sent_encodings

    def _encode_context(self, inputs, input_floors, output_floors):
        batch_size, history_len, max_x_sent_len = inputs.size()

        flat_inputs = inputs.view(batch_size*history_len, max_x_sent_len)
        input_lens = (inputs != self.pad_token_id).sum(-1)
        flat_input_lens = input_lens.view(batch_size*history_len)
        _, _, sent_encodings = self.sent_encoder(flat_inputs, flat_input_lens)
        sent_encodings = sent_encodings.view(batch_size, history_len, -1)

        if self.floor_encoder is not None:
            src_floors = input_floors.view(-1)
            tgt_floors = output_floors.unsqueeze(1).repeat(1, history_len).view(-1)
            sent_encodings = sent_encodings.view(batch_size*history_len, -1)
            sent_encodings = self.floor_encoder(
                sent_encodings,
                src_floors=src_floors,
                tgt_floors=tgt_floors
            )
            sent_encodings = sent_encodings.view(batch_size, history_len, -1)

        ctx_lens = (input_lens > 0).long().sum(1)  # equals number of non-padding sents
        _, _, ctx_encodings = self.dial_encoder(sent_encodings, ctx_lens)  # [batch_size, dial_encoder_dim]

        return ctx_encodings

    def _compute_ref_metric(self, hypotheses, references):
        hyp_word_embeddings = self.word_embedding(hypotheses)  # [batch_size, seq_len, word_embedding_dim]
        ref_word_embeddings = self.word_embedding(references)  # [batch_size, seq_len, word_embedding_dim]
        hyp_max_embedding, _ = hyp_word_embeddings.max(dim=1)  # [batch_size, word_embedding_dim]
        ref_max_embedding, _ = ref_word_embeddings.max(dim=1)  # [batch_size, word_embedding_dim]
        ref_metric = F.cosine_similarity(hyp_max_embedding, ref_max_embedding, dim=1)  # [batch_size]
        ref_metric = (ref_metric+1)/2

        return ref_metric

    def _compute_unref_metric(self, hyp_encodings, ctx_encodings):
        quad = torch.matmul(ctx_encodings, self.M)  # [batch_size, sent_encoder_dim]
        quad = (quad*hyp_encodings).sum(1).view(-1, 1)  # [batch_size, 1]
        x = torch.cat([ctx_encodings, hyp_encodings, quad], dim=1)
        for m in self.unref_fc:
            x = m(x)
        unref_metric = x.view(-1)  # [batch_size]

        return unref_metric

    def _compute_hybrid_metric(self, ref_metric, unref_metric):
        if self.metric == "hybrid":
            return (ref_metric+unref_metric)/2
        elif self.metric == "ref":
            return ref_metric
        elif self.metric == "unref":
            return unref_metric

    def _compute_score(self, metric):
        return 4*metric + 1

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
        for name, param in pretrained_state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].data = param.data

        self.load_state_dict(own_state_dict)

    def unsupervised_train_step(self, data):
        """One training step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of context sentences
                'Y' {LongTensor [batch_size, max_y_sent_len]} -- token ids of negtive-sampling response sentence
                'Y_ref' {LongTensor [batch_size, max_y_sent_len]} -- token ids of reference response sentence
                'Y_floor' {LongTensor [batch_size]} -- floor of response sentence

        Returns:
            dict of data -- returned keys and values
                'loss' {FloatTensor []} -- loss to backword
            dict of statistics -- returned keys and values
                'loss' {float} -- batch loss
        """
        X, Y_neg, Y_ref = data["X"], data["Y"], data["Y_ref"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]

        # Forward
        neg_encodings = self._encode_response(Y_neg)
        ref_encodings = self._encode_response(Y_ref)
        ctx_encodings = self._encode_context(X, X_floor, Y_floor)
        neg_unref_metric = self._compute_unref_metric(neg_encodings, ctx_encodings)
        ref_unref_metric = self._compute_unref_metric(ref_encodings, ctx_encodings)

        # Compute loss
        batch_size = X.size(0)
        loss = F.margin_ranking_loss(
            ref_unref_metric,
            neg_unref_metric,
            torch.ones(batch_size).long().to(DEVICE),
            margin=self.margin,
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
                'Y' {LongTensor [batch_size, max_y_sent_len]} -- token ids of hypothesis response sentence
                'Y_ref' {LongTensor [batch_size, max_y_sent_len]} -- token ids of reference response sentence
                'Y_floor' {LongTensor [batch_size]} -- floor of response sentence
                'Y_is_next' {BoolTensor [batch_size]} -- is_next label (is_next: True, not_next: False) of response sentence

        Returns:
            dict of data -- returned keys and values
                'probs' {FloatTensor [batch_size]} -- probablities of Y being the next sentence
            dict of statistics -- returned keys and values
                'monitor' {float} -- a monitor number for learning rate scheduling
        """
        X, Y_hyp, Y_ref = data["X"], data["Y"], data["Y_ref"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_is_next = data["Y_is_next"]

        with torch.no_grad():
            # Forward
            hyp_encodings = self._encode_response(Y_hyp)
            ref_encodings = self._encode_response(Y_ref)
            ctx_encodings = self._encode_context(X, X_floor, Y_floor)
            hyp_ref_metric = self._compute_ref_metric(Y_hyp, Y_ref)
            hyp_unref_metric = self._compute_unref_metric(hyp_encodings, ctx_encodings)
            hyp_hybrid_metric = self._compute_hybrid_metric(hyp_ref_metric, hyp_unref_metric)

            # loss
            loss = F.mse_loss(
                hyp_hybrid_metric,
                Y_is_next.float(),
                reduction="mean"
            )

        # return dicts
        ret_data = {
            "probs": hyp_hybrid_metric,
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
                'Y' {LongTensor [batch_size, max_y_sent_len]} -- token ids of hypothesis response sentence
                'Y_ref' {LongTensor [batch_size, max_y_sent_len]} -- token ids of reference response sentence
                'Y_floor' {LongTensor [batch_size]} -- floor of response sentence

        Returns:
            dict of data -- returned keys and values
                'probs' {FloatTensor [batch_size]} -- probablities of Y being the next sentence
            dict of statistics -- returned keys and values
        """
        X, Y_hyp, Y_ref = data["X"], data["Y"], data["Y_ref"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]

        with torch.no_grad():
            # Forward
            hyp_encodings = self._encode_response(Y_hyp)
            ref_encodings = self._encode_response(Y_ref)
            ctx_encodings = self._encode_context(X, X_floor, Y_floor)
            ref_metric = self._compute_ref_metric(Y_hyp, Y_ref)
            unref_metric = self._compute_unref_metric(hyp_encodings, ctx_encodings)
            hybrid_metric = self._compute_hybrid_metric(ref_metric, unref_metric)

        ret_data = {
            "probs": hybrid_metric,
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
                'Y_ref' {LongTensor [batch_size, max_y_sent_len]} -- token ids of reference response sentence
                'Y_floor' {LongTensor [batch_size]} -- floor of response sentence
                'Y_tgt_score' {FloatTensor [batch_size]} -- target score of response sentence

        Returns:
            dict of data -- returned keys and values
                'loss' {FloatTensor []} -- loss to backword
            dict of statistics -- returned keys and values
                'loss' {float} -- batch loss
        """
        X, Y_hyp, Y_ref = data["X"], data["Y"], data["Y_ref"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_tgt_score = data["Y_tgt_score"]

        batch_size = X.size(0)

        # Forward
        hyp_encodings = self._encode_response(Y_hyp)
        ref_encodings = self._encode_response(Y_ref)
        ctx_encodings = self._encode_context(X, X_floor, Y_floor)
        ref_metric = self._compute_ref_metric(Y_hyp, Y_ref)
        unref_metric = self._compute_unref_metric(hyp_encodings, ctx_encodings)
        hybrid_metric = self._compute_hybrid_metric(ref_metric, unref_metric)
        scores = self._compute_score(hybrid_metric)

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

    def supervised_test_step(self, data):
        """One test step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of context sentences
                'Y' {LongTensor [batch_size, max_y_sent_len]} -- token ids of hypothesis response sentence
                'Y_ref' {LongTensor [batch_size, max_y_sent_len]} -- token ids of reference response sentence
                'Y_floor' {LongTensor [batch_size]} -- floor of response sentence

        Returns:
            dict of data -- returned keys and values
                'probs' {FloatTensor [batch_size]} -- probablities of Y being the next sentence
            dict of statistics -- returned keys and values
        """
        X, Y_hyp, Y_ref = data["X"], data["Y"], data["Y_ref"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]

        with torch.no_grad():
            # Forward
            hyp_encodings = self._encode_response(Y_hyp)
            ref_encodings = self._encode_response(Y_ref)
            ctx_encodings = self._encode_context(X, X_floor, Y_floor)
            ref_metric = self._compute_ref_metric(Y_hyp, Y_ref)
            unref_metric = self._compute_unref_metric(hyp_encodings, ctx_encodings)
            hybrid_metric = self._compute_hybrid_metric(ref_metric, unref_metric)
            scores = self._compute_score(hybrid_metric)

        ret_data = {
            "scores": scores
        }
        ret_stat = {}

        return ret_data, ret_stat
