import code
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.submodules import DynamicRNN, LockedDropout
from model.modules.utils import init_module_weights, embedded_dropout

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

#######################
# Transformer encoders
#######################

class SelfAttnBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.0):
        super(SelfAttnBlock, self).__init__()

        self.norm1 = nn.LayerNorm(input_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(input_dim)
        self.ffn = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        ])
        self.dropout2 = nn.Dropout(dropout)

        ## Initialization
        self._init_weights()

    def _init_weights(self):
        init_module_weights(self.self_attn)
        init_module_weights(self.ffn)

    def forward(self, inputs, attn_mask=None):
        key_padding_mask = None if attn_mask is None else ~attn_mask

        res_x = inputs
        x = self.norm1(inputs)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask
        )
        x = res_x + self.dropout1(x)
        
        res_x = x
        x = self.norm2(x)
        for m in self.ffn:
            x = m(x)
        x = res_x + self.dropout2(x)

        return x, (attn,)


class SelfAttnEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_blocks, dropout=0.0):
        super(SelfAttnEncoder, self).__init__()

        self.blocks = nn.ModuleList([
            SelfAttnBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, inputs, attn_mask):
        inputs = inputs.transpose(0, 1).contiguous()
        x = inputs
        for block in self.blocks:
            x, (attn) = block(x, attn_mask)
        x = self.norm(x)
        x = x.transpose(0, 1).contiguous()
        return x, (attn,)

