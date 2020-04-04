import re
import code
import math
import json
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from sklearn.decomposition import PCA

from model.modules.encoders import EncoderRNN
from model.modules.decoders import DecoderRNN
from model.modules.submodules import AbsFloorEmbEncoder, RelFloorEmbEncoder
from model.modules.utils import init_module_weights, init_word_embedding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ADEM(nn.Module):
    def __init__(self, config, tokenizer):
        super(ADEM, self).__init__()

        # Attributes
        # Attributes from config
        self.word_embedding_dim = config.word_embedding_dim
        self.attr_embedding_dim = config.attr_embedding_dim
        self.sent_encoder_hidden_dim = config.sent_encoder_hidden_dim
        self.n_sent_encoder_layers = config.n_sent_encoder_layers
        self.dial_encoder_hidden_dim = config.dial_encoder_hidden_dim
        self.n_dial_encoder_layers = config.n_dial_encoder_layers
        assert self.sent_encoder_hidden_dim == self.dial_encoder_hidden_dim
        self.latent_variable_dim = config.latent_dim
        self.tie_weights = config.tie_weights
        self.rnn_type = config.rnn_type
        self.word_embedding_path = config.word_embedding_path
        self.floor_encoder_type = config.floor_encoder
        self.metric = config.metric_type
        # Optional attributes from config
        self.dropout_emb = config.dropout if hasattr(config, "dropout") else 0.0
        self.dropout_input = config.dropout if hasattr(config, "dropout") else 0.0
        self.dropout_hidden = config.dropout if hasattr(config, "dropout") else 0.0
        self.dropout_output = config.dropout if hasattr(config, "dropout") else 0.0
        self.use_pretrained_word_embedding = config.use_pretrained_word_embedding if hasattr(config, "use_pretrained_word_embedding") else False
        # Other attributes
        self.id2word = tokenizer.id2word
        self.vocab_size = len(tokenizer.word2id)
        self.pad_token_id = tokenizer.pad_token_id
        self.n_pca_components = 50

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
            dropout_emb=self.dropout_emb,
            dropout_input=self.dropout_input,
            dropout_hidden=self.dropout_hidden,
            dropout_output=self.dropout_output,
            bidirectional=True,
            embedding=self.word_embedding,
            rnn_type=self.rnn_type,
        )
        self.dial_encoder = EncoderRNN(
            input_dim=self.sent_encoder_hidden_dim,
            hidden_dim=self.dial_encoder_hidden_dim,
            n_layers=self.n_dial_encoder_layers,
            dropout_emb=self.dropout_emb,
            dropout_input=self.dropout_input,
            dropout_hidden=self.dropout_hidden,
            dropout_output=self.dropout_output,
            bidirectional=True,
            rnn_type=self.rnn_type,
        )
        self.pca = PCA(n_components=self.n_pca_components)

        # Scoring components
        self.M = nn.Linear(self.n_pca_components, self.n_pca_components)
        self.N = nn.Linear(self.n_pca_components, self.n_pca_components)
        self.alpha = None
        self.beta = None

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

    def _reset_optimizer_for_finetuning(self):
        params = list(self.M.parameters()) + list(self.N.parameters())

        if self.optimizer_type == "adam":
            self.optimizer = optim.AdamW(
                params,
                lr=self.init_lr,
                weight_decay=self.l2_penalty
            )
        elif self.optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                params,
                lr=self.init_lr,
                weight_decay=self.l2_penalty
            )

    def _init_weights(self):
        # diagonal initialization
        nn.init.eye(self.M.weight)
        self.M.bias.data.fill_(0)
        nn.init.eye(self.N.weight)
        self.N.bias.data.fill_(0)

    def _encode(self, inputs, input_floors, output_floors):
        batch_size, history_len, max_x_sent_len = inputs.size()

        flat_inputs = inputs.view(batch_size*history_len, max_x_sent_len)
        input_lens = (inputs != self.pad_token_id).sum(-1)
        flat_input_lens = input_lens.view(batch_size*history_len)
        word_encodings, _, sent_encodings = self.sent_encoder(flat_inputs, flat_input_lens)
        word_encodings = word_encodings.view(batch_size, history_len, max_x_sent_len, -1)
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

        dialog_lens = (input_lens > 0).long().sum(1)  # equals number of non-padding sents
        _, _, dialog_encodings = self.dial_encoder(sent_encodings, dialog_lens)  # [batch_size, dialog_encoder_dim]

        return word_encodings, sent_encodings, dialog_encodings

    def _compute_scores(self, inputs, outputs, references, input_floors, output_floors):
        batch_size = inputs.size(0)

        with torch.no_grad():
            # context encodings
            _, _, ctx_encodings = self._encode(inputs, input_floors, output_floors)

            # response encodings
            output_lens = (outputs != self.pad_token_id).sum(-1)
            _, _, response_encodings = self.sent_encoder(outputs, output_lens)
            reference_lens = (references != self.pad_token_id).sum(-1)
            _, _, reference_encodings = self.sent_encoder(references, reference_lens)

            # transform using PCA
            ctx_encodings = self.pca.transform(ctx_encodings.cpu())
            response_encodings = self.pca.transform(response_encodings.cpu())
            reference_encodings = self.pca.transform(reference_encodings.cpu())

            # convert np arrays back to tensors
            ctx_encodings = torch.FloatTensor(ctx_encodings).to(DEVICE)
            response_encodings = torch.FloatTensor(response_encodings).to(DEVICE)
            reference_encodings = torch.FloatTensor(reference_encodings).to(DEVICE)

        # compute scores
        m_response_encodings = self.M(response_encodings)
        n_response_encodings = self.N(response_encodings)
        pred1 = torch.bmm(
            ctx_encodings.view(batch_size, 1, -1),
            m_response_encodings.view(batch_size, -1, 1)
        ).view(batch_size)
        pred2 = torch.bmm(
            reference_encodings.view(batch_size, 1, -1),
            n_response_encodings.view(batch_size, -1, 1)
        ).view(batch_size)

        if self.metric == "hybrid":
            pred = pred1 + pred2
        elif self.metric == "ref":
            pred = pred2
        elif self.metric == "unref":
            pred = pred1

        scores = 3 + 4 * (pred - self.alpha) / self.beta

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
        for name, param in pretrained_state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].data = param.data

        self.load_state_dict(own_state_dict)

        self._reset_optimizer_for_finetuning()

    def estimate_pca_parameters(self, train_data_source):
        self.eval()
        C, R, R_hat = [], [], []
        with torch.no_grad():
            train_data_source.epoch_init(shuffle=False)
            while True:
                batch_data = train_data_source.next(100)
                if batch_data is None:
                    break

                X, Y, Y_ref = batch_data["X"], batch_data["Y"], batch_data["Y_ref"]
                X_floor, Y_floor = batch_data["X_floor"], batch_data["Y_floor"]

                # context encodings
                _, _, ctx_encodings = self._encode(X, X_floor, Y_floor)

                # response encodings
                Y_len = (Y != self.pad_token_id).sum(-1)
                _, _, response_encodings = self.sent_encoder(Y, Y_len)
                Y_ref_len = (Y_ref != self.pad_token_id).sum(-1)
                _, _, reference_encodings = self.sent_encoder(Y_ref, Y_ref_len)

                C += ctx_encodings.tolist()
                R += reference_encodings.tolist()
                R_hat += response_encodings.tolist()

        if self.metric == "hybrid":
            embeddings = np.array(C+R+R_hat)
        elif self.metric == "ref":
            embeddings = np.array(R+R_hat)
        elif self.metric == "unref":
            embeddings = np.array(C+R_hat)
        self.pca.fit(embeddings)

        print("Estimated PCA Variance:")
        print(f"  var ratio = {self.pca.explained_variance_ratio_}")
        print(f"  sum = {np.sum(self.pca.explained_variance_ratio_)}")

    def estimate_scaling_constants(self, train_data_source):
        self.eval()
        C, R, R_hat = [], [], []
        with torch.no_grad():
            train_data_source.epoch_init(shuffle=False)
            while True:
                batch_data = train_data_source.next(100)
                if batch_data is None:
                    break

                X, Y, Y_ref = batch_data["X"], batch_data["Y"], batch_data["Y_ref"]
                X_floor, Y_floor = batch_data["X_floor"], batch_data["Y_floor"]

                # context encodings
                _, _, ctx_encodings = self._encode(X, X_floor, Y_floor)

                # response encodings
                Y_len = (Y != self.pad_token_id).sum(-1)
                _, _, response_encodings = self.sent_encoder(Y, Y_len)
                Y_ref_len = (Y_ref != self.pad_token_id).sum(-1)
                _, _, reference_encodings = self.sent_encoder(Y_ref, Y_ref_len)

                # transform using PCA
                ctx_encodings = self.pca.transform(ctx_encodings.cpu())
                response_encodings = self.pca.transform(response_encodings.cpu())
                reference_encodings = self.pca.transform(reference_encodings.cpu())

                C += ctx_encodings.tolist()
                R += reference_encodings.tolist()
                R_hat += response_encodings.tolist()

        C = np.array(C)
        R = np.array(R)
        R_hat = np.array(R_hat)

        prod_list = []
        for i in range(len(C)):
            term = 0
            term += np.dot(C[i], R_hat[i])
            term += np.dot(R[i], R_hat[i])
            prod_list.append(term)
        self.alpha = np.mean(prod_list)
        self.beta = max(prod_list) - min(prod_list)

        print(f"Estimated scaling factors alpha = {self.alpha}, beta = {self.beta}")

    def supervised_train_step(self, data):
        """One training step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of context sentences
                'Y' {LongTensor [batch_size, max_y_sent_len]} -- token ids of response sentence
                'Y_floor' {LongTensor [batch_size]} -- floor of response sentence
                'Y_ref' {LongTensor [batch_size, max_y_ref_sent_len]} -- token ids of referencee response sentence
                'Y_tgt_score' {FloatTensor [batch_size]} -- target score of response sentence

        Returns:
            dict of data -- returned keys and values
                'loss' {FloatTensor []} -- loss to backword
            dict of statistics -- returned keys and values
                'loss' {float} -- batch loss
        """
        X, Y, Y_ref = data["X"], data["Y"], data["Y_ref"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_tgt_score = data["Y_tgt_score"]

        # forward
        scores = self._compute_scores(
            inputs=X,
            outputs=Y,
            references=Y_ref,
            input_floors=X_floor,
            output_floors=Y_floor
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
                'Y_ref' {LongTensor [batch_size, max_y_ref_sent_len]} -- token ids of referencee response sentence
                'Y_tgt_score' {FloatTensor [batch_size]} -- target score of response sentence

        Returns:
            dict of data -- returned keys and values

            dict of statistics -- returned keys and values
                'monitor' {float} -- a monitor number for learning rate scheduling
        """
        X, Y, Y_ref = data["X"], data["Y"], data["Y_ref"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_tgt_score = data["Y_tgt_score"]

        batch_size = X.size(0)

        with torch.no_grad():
            # forward
            scores = self._compute_scores(
                inputs=X,
                outputs=Y,
                references=Y_ref,
                input_floors=X_floor,
                output_floors=Y_floor
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
                'Y_ref' {LongTensor [batch_size, max_y_ref_sent_len]} -- token ids of referencee response sentence

        Returns:
            dict of data -- returned keys and values
                'scores' {FloatTensor [batch_size]} -- predicted score of response
            dict of statistics -- returned keys and values

        """
        X, Y, Y_ref = data["X"], data["Y"], data["Y_ref"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]

        with torch.no_grad():
            # forward
            scores = self._compute_scores(
                inputs=X,
                outputs=Y,
                references=Y_ref,
                input_floors=X_floor,
                output_floors=Y_floor
            )

        ret_data = {
            "scores": scores
        }
        ret_stat = {}

        return ret_data, ret_stat
