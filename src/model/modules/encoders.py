import code
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .submodules import DynamicRNN, LockedDropout
from .utils import init_module_weights, embedded_dropout

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers,
                    embedding=None, bidirectional=True,
                    dropout_emb=0.0, dropout_input=0.0, dropout_hidden=0.0, dropout_output=0.0,
                    rnn_type="gru"):
        super(EncoderRNN, self).__init__()

        # attributes
        self.dropout_emb = dropout_emb
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output
        self.rnn_type = rnn_type

        # input components
        self.embedding = embedding

        # rnn components
        self.rnn = DynamicRNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout_hidden,
            rnn_type=rnn_type
        )

        # variational dropout
        self.lockdrop = LockedDropout()

        # initialization
        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, inputs, input_lens=None, init_hiddens=None):
        if self.embedding is not None:
            # dropout between words and embedding matrix
            # get word embeddings
            rnn_inputs = embedded_dropout(
                embed=self.embedding,
                words=inputs,
                dropout=self.dropout_emb if self.training else 0
            )

            # dropout between rnn inputs and rnn
            rnn_inputs = self.lockdrop(rnn_inputs, self.dropout_input)
        else:
            rnn_inputs = inputs

        # get rnn outputs
        step_outputs, final_hiddens, encodings = self.rnn(rnn_inputs, input_lens, init_hiddens)

        # dropout between rnn outputs and subsequent layers
        step_outputs = self.lockdrop(step_outputs, self.dropout_output)
        encodings = self.lockdrop(encodings, self.dropout_output)

        return step_outputs, final_hiddens, encodings

