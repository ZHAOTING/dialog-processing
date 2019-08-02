import re
import code
import math
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from ..modules.encoders import EncoderRNN
from ..modules.submodules import AbsFloorEmbEncoder, RelFloorEmbEncoder
from ..modules.utils import init_module_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HRE(nn.Module):
    def __init__(self, config, tokenizer):
        super(HRE, self).__init__()

        ## Attributes
        # Attributes from config
        self.num_labels = len(config.dialog_acts)
        self.word_embedding_dim = config.word_embedding_dim
        self.attr_embedding_dim = config.attr_embedding_dim
        self.sent_encoder_hidden_dim = config.sent_encoder_hidden_dim
        self.n_sent_encoder_layers = config.n_sent_encoder_layers
        self.dial_encoder_hidden_dim = config.dial_encoder_hidden_dim
        self.n_dial_encoder_layers = config.n_dial_encoder_layers
        self.dropout_emb = config.dropout
        self.dropout_input = config.dropout
        self.dropout_hidden = config.dropout
        self.dropout_output = config.dropout
        self.rnn_type = config.rnn_type
        self.optimizer_type = config.optimizer
        self.gradient_clip = config.gradient_clip
        self.l2_penalty = config.l2_penalty
        self.use_pretrained_word_embedding = config.use_pretrained_word_embedding
        self.word_embedding_path = config.word_embedding_path
        self.floor_encoder_type = config.floor_encoder
        # Other attributes
        self.word2id = tokenizer.word2id
        self.id2word = tokenizer.id2word
        self.vocab_size = len(tokenizer.word2id)
        self.pad_id = tokenizer.pad_id

        ## Input components
        self.word_embedding = nn.Embedding(
            self.vocab_size,
            self.word_embedding_dim,
            padding_idx=self.pad_id,
            _weight=self._init_word_embedding(),
        )

        ## Encoding components
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
            bidirectional=False,
            rnn_type=self.rnn_type,
        )

        ## Classification components
        self.output_fc = nn.Linear(
            self.dial_encoder_hidden_dim,
            self.num_labels
        )

        ## Extra components
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

        ## Initialization
        self._set_optimizer()
        self._init_weights()
        self._print_model_stats()

    def _set_optimizer(self):
        if self.optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.parameters(),
                lr=0.0,
                weight_decay=self.l2_penalty
            )
        elif self.optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.parameters(),
                lr=0.0,
                weight_decay=self.l2_penalty
            )

    def _print_model_stats(self):
        total_parameters = 0
        for name, param in self.named_parameters():
            # shape is an array of tf.Dimension
            shape = param.size()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            print("Trainable %s with %d parameters" % (name, variable_parameters))
            total_parameters += variable_parameters
        print("Total number of trainable parameters is %d" % total_parameters)

    def _init_weights(self):
        init_module_weights(self.output_fc)

    def _init_word_embedding(self):
        if self.use_pretrained_word_embedding:
            embeddings = []
            pretrained_embeddings = json.load(open(self.word_embedding_path))
            in_vocab_cnt = 0
            for word_id in range(len(self.id2word)):
                word = self.id2word[word_id]
                if word in pretrained_embeddings:
                    embeddings.append(pretrained_embeddings[word])
                    in_vocab_cnt += 1
                else:
                    embeddings.append([0.0]*self.word_embedding_dim)
            weights = nn.Parameter(torch.FloatTensor(embeddings).to(DEVICE))
            print("{}/{} pretrained word embedding in vocab".
                  format(in_vocab_cnt, self.vocab_size))
        else:
            weights = nn.Parameter(
                torch.FloatTensor(
                    self.vocab_size,
                    self.word_embedding_dim
                ).to(DEVICE)
            )
            torch.nn.init.uniform_(weights, -1.0, 1.0)
        weights[self.pad_id].data.fill_(0)
        return weights

    def _encode(self, inputs, input_floors, output_floors):
        batch_size, history_len, max_x_sent_len = inputs.size()

        flat_inputs = inputs.view(batch_size*history_len, max_x_sent_len)
        input_lens = (inputs != self.pad_id).sum(-1)
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

    def load_model(self, model_path):
        """Load pretrained model weights from model_path

        Arguments:
            model_path {str} -- path to pretrained model weights
        """
        if DEVICE == "cuda":
            pretrained_state_dict = torch.load(model_path)
        else:
            pretrained_state_dict = torch.load(model_path,
                                               map_location=lambda storage, loc: storage)
        self.load_state_dict(pretrained_state_dict)

    def train_step(self, data, lr):
        """One training step

        Arguments:
            data {dict of data} -- required keys and values:
                X {LongTensor [batch_size, history_len+1, max_x_sent_len]} -- token ids of context and target sentences
                X_floor {LongTensor [batch_size, history_len+1]} -- floors of context and target sentences
                Y_floor {LongTensor [batch_size]} -- floor of target sentence
                Y_da {LongTensor [batch_size]} -- dialog acts of target sentence

            lr {float} -- learning rate

        Returns:
            dict of statistics -- returned keys and values
                loss {float} -- batch loss
        """
        X = data["X"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_da = data["Y_da"]

        ## Forward
        word_encodings, sent_encodings, dial_encodings = self._encode(
            inputs=X,
            input_floors=X_floor,
            output_floors=Y_floor
        )
        logits = self.output_fc(dial_encodings)

        ## Calculate loss
        loss = F.cross_entropy(
            logits.view(-1, self.num_labels),
            Y_da.view(-1),
            reduction="mean"
        )

        ## Return statistics
        ret_statistics = {
            "loss": loss.item()
        }

        ## Backward
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)
        self.optimizer.step()

        return ret_statistics

    def evaluate_step(self, data):
        """One evaluation step

        Arguments:
            data {dict of data} -- required keys and values:
                X {LongTensor [batch_size, history_len+1, max_x_sent_len]} -- token ids of context and target sentences
                X_floor {LongTensor [batch_size, history_len+1]} -- floors of context and target sentences
                Y_floor {LongTensor [batch_size]} -- floor of target sentence
                Y_da {LongTensor [batch_size]} -- dialog acts of target sentence

        Returns:
            dict of outputs -- returned keys and values
                labels {LongTensor [batch_size]} -- predicted label of target sentence
            dict of statistics -- returned keys and values
                monitor {float} -- a monitor number for learning rate scheduling
        """
        X = data["X"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_da = data["Y_da"]

        with torch.no_grad():
            ## Forward
            word_encodings, sent_encodings, dial_encodings = self._encode(
                inputs=X,
                input_floors=X_floor,
                output_floors=Y_floor
            )
            logits = self.output_fc(dial_encodings)
            _, labels = torch.max(logits, dim=1)

            ## Loss
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                Y_da.view(-1),
                reduction="mean"
            )

        # return outputs
        ret_outputs = {
            "labels": labels
        }

        # return statistics
        ret_statistics = {
            "monitor": loss.item()
        }

        return ret_outputs, ret_statistics
