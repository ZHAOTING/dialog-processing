import math
import code

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .utils import init_module_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout):
        if not self.training or not dropout:
            return x
        if x.dim() == 3:
            m = x.data.new(1, 1, x.size(2)).bernoulli_(1 - dropout)
        elif x.dim() == 2:
            m = x.data.new(1, x.size(1)).bernoulli_(1 - dropout)
        else:
            raise Exception("Trying applying variational dropout to input with unsupported dimension {}".format(x.dim()))
        mask = m / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

class DynamicRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers,
        bidirectional=False, dropout=0.0, rnn_type="gru"):
        super(DynamicRNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        if rnn_type == "gru":
            self.rnn_cell = nn.GRU
        elif rnn_type == "lstm":
            self.rnn_cell = nn.LSTM
        self.rnn = self.rnn_cell(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            #dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.init_weights()

    def init_weights(self):
        init_module_weights(self.rnn)

    def forward(self, inputs, input_lens=None, init_hiddens=None):
        # set values
        B = inputs.size(0)

        # for full length batch data
        if input_lens is None:
            L = inputs.size(1)
            input_lens = torch.LongTensor([L]*B).to(DEVICE)
        else:
            # avoid zero-length sequence
            input_lens.data[input_lens == 0] = 1

        # pack
        packed_inputs = pack_padded_sequence(inputs, input_lens, batch_first=True, enforce_sorted=False)

        # process
        if init_hiddens is None:
            packed_outputs, final_hiddens = self.rnn(packed_inputs)
        else:
            packed_outputs, final_hiddens = self.rnn(packed_inputs, init_hiddens)

        # unpack
        unpacked_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        if self.bidirectional:
            unpacked_outputs = unpacked_outputs.view(B, -1, self.hidden_dim, 2)
            unpacked_outputs = unpacked_outputs.mean(3) # mean over directions

        # last encoding
        if self.rnn_type == "gru":
            encodings = final_hiddens.mean(0) # mean over directions and layers
        elif self.rnn_type == "lstm":
            encodings = torch.cat(final_hiddens, dim=0).mean(0)

        return unpacked_outputs, final_hiddens, encodings


class Attention(nn.Module):
    r"""Applies an attention mechanism on the output features from the decoder.

        math:
            \begin{array}{ll}
            e_i^t = W_e tanh ( W_q * q + W_c * c + b ), i is query step, t is context step
            a^t = softmax ( e^t )
            \end{array}

    Inputs:
        query {FloatTensor} -- (batch, query_len, query_dim)
        context {FloatTensor} -- (batch, context_len, context_dim)
        mask {ByteTensor} -- (batch, context_len)

    Returns:
        [type] -- [description]
    """

    def __init__(self, query_dim, context_dim, hidden_dim, coverage=False):
        super(Attention, self).__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.coverage = coverage

        self.w_query = nn.Linear(query_dim, hidden_dim)
        self.w_context = nn.Linear(context_dim, hidden_dim)
        self.w_emit = nn.Linear(hidden_dim, 1)
        self.linear_out = nn.Linear(query_dim+context_dim, query_dim)

        self.init_weights()

    def init_weights(self):
        init_module_weights(self.w_query, 0.1)
        init_module_weights(self.w_context, 0.1)
        init_module_weights(self.w_emit, 0.1)
        init_module_weights(self.linear_out, 0.1)

    def forward(self, query, context, mask=None):
        batch_size = query.size(0)
        query_len = query.size(1)
        context_len = context.size(1)

        # see math formula
        mapped_context = self.w_context(context) # [batch_size, context_len, hidden_dim]
        mapped_query = self.w_query(query) # [batch_size, query_len, hidden_dim]
        tiled_context = mapped_context.unsqueeze(1) # [batch_size, 1, context_len, hidden_dim]
        tiled_query = mapped_query.unsqueeze(2).repeat(1, 1, context_len, 1) # [batch_size, query_len, context_len, hidden_dim]
        emission_input = torch.tanh(tiled_context+tiled_query) # [batch_size, query_len, context_len, hidden_dim]
        emission = self.w_emit(emission_input).squeeze(-1)  # [batch_size, query_len, context_len]

        # mask out unneeded contexts
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, query_len, 1)
            ignore_mask = 1 - mask
            emission.data.masked_fill_(ignore_mask, -float('inf'))

        # calculate attention
        attn = F.softmax(emission, dim=2)  # [batch_size, query_len, context_len]

        # prevent nan when all contexts are float(-inf)
        nan_mask = (attn != attn)
        attn.data.masked_fill_(nan_mask, 0.0)

        # get weighted context
        weighted_context = torch.bmm(attn, context)  # [batch, query_len, context_dim]

        # combined weighted_context and query
        combined = torch.cat((weighted_context, query), dim=2) # [batch, query_len, context_dim+query_dim]
        output = torch.tanh(self.linear_out(combined))  # [batch, query_len, query_dim]

        return output, attn

## Floor encoders
class AbsFloorOneHotEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AbsFloorOneHotEncoder, self).__init__()
        self.input_dim = input_dim

        self.linear = nn.Linear(input_dim+2, input_dim)

    def init_weights(self):
        init_module_weights(self.linear, 0.1)

    def forward(self, encodings, src_floors, **args):
        B = encodings.size(0)

        floors_one_hot = torch.stack([src_floors, 1-src_floors], dim=1).float()
        encodings = torch.cat([encodings, floors_one_hot], dim=1)
        outputs = self.linear(encodings)

        return outputs

class RelFloorOneHotEncoder(nn.Module):
    def __init__(self, input_dim):
        super(RelFloorOneHotEncoder, self).__init__()
        self.input_dim = input_dim

        self.linear = nn.Linear(input_dim+2, input_dim)

    def init_weights(self):
        init_module_weights(self.linear, 0.1)

    def forward(self, encodings, src_floors, tgt_floors):
        B = encodings.size(0)

        same_floors = (src_floors == tgt_floors).float()
        floors_one_hot = torch.stack([same_floors, 1-same_floors], dim=1)
        # [1, 0] for same floor, [0, 1] for different floors
        encodings = torch.cat([encodings, floors_one_hot], dim=1)
        outputs = self.linear(encodings)

        return outputs

class AbsFloorEmbEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(AbsFloorEmbEncoder, self).__init__()
        self.input_dim = input_dim

        self.embedding = nn.Embedding(2, embedding_dim)
        self.linear = nn.Linear(input_dim+embedding_dim, input_dim)

        self.init_weights()

    def init_weights(self):
        init_module_weights(self.embedding, 1.0)
        init_module_weights(self.linear, 0.1)

    def forward(self, encodings, src_floors, **args):
        B = encodings.size(0)

        floor_embeddings = self.embedding(src_floors)
        encodings = torch.cat([encodings, floor_embeddings], dim=1)
        outputs = self.linear(encodings)

        return outputs

class RelFloorEmbEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(RelFloorEmbEncoder, self).__init__()
        self.input_dim = input_dim

        self.embedding = nn.Embedding(2, embedding_dim)
        self.linear = nn.Linear(input_dim+embedding_dim, input_dim)

        self.init_weights()

    def init_weights(self):
        init_module_weights(self.embedding, 1.0)
        init_module_weights(self.linear, 0.1)

    def forward(self, encodings, src_floors, tgt_floors):
        B = encodings.size(0)

        same_floors = (src_floors == tgt_floors).long()
        floor_embeddings = self.embedding(same_floors)
        encodings = torch.cat([encodings, floor_embeddings], dim=1)
        outputs = self.linear(encodings)

        return outputs

## Variational modules
class GaussianVariation(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(GaussianVariation, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.ctx_fc = nn.Linear(input_dim, z_dim)
        self.ctx2mu = nn.Linear(z_dim, z_dim)
        self.ctx2var = nn.Linear(z_dim, z_dim)

        self.init_weights()

    def init_weights(self):
        init_module_weights(self.ctx_fc)
        init_module_weights(self.ctx2mu)
        init_module_weights(self.ctx2var)

    def forward(self, context):
        batch_size, _ = context.size()
        context = torch.tanh(self.ctx_fc(context))
        mu = self.ctx2mu(context)
        var = F.softplus(self.ctx2var(context))
        std = torch.sqrt(var)

        epsilon = torch.randn([batch_size, self.z_dim]).to(DEVICE)
        z = epsilon * std + mu
        return z, mu, var
