import code

import torch
import torch.nn as nn
import torch.nn.functional as F

from .submodules import Attention, DynamicRNN, LockedDropout
from .utils import init_module_weights, init_rnn_hidden_states, embedded_dropout

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecoderRNN(nn.Module):
    MODE_TEACHER_FORCE = "teacher forcing"
    MODE_FREE_RUN = "free running"
    GEN_GREEDY = "greedy"
    GEN_BEAM = "beam"
    GEN_SAMPLE = "sample"
    GEN_TOP = "top"

    def __init__(self, vocab_size, input_dim, hidden_dim, n_layers,
            bos_id, eos_id, pad_id, max_len,
            dropout_emb=0.0, dropout_input=0.0, dropout_hidden=0.0, dropout_output=0.0,
            use_attention=False, attn_dim=0, feat_dim=0,
            embedding=None, tie_weights=False, rnn_type="gru"):
        super(DecoderRNN, self).__init__()

        # attributes
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_len = max_len
        self.dropout_emb = dropout_emb
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output
        self.use_attention = use_attention
        self.attn_dim = attn_dim
        self.feat_dim = feat_dim
        self.tie_weights = tie_weights
        self.rnn_type = rnn_type

        # input components
        if embedding is None:
            self.embedding = nn.Embedding(
                vocab_size,
                input_dim,
                padding_idx=pad_id
            )
        else:
            self.embedding = embedding
        if feat_dim > 0:
            self.feat_fc = nn.Linear(
                input_dim+feat_dim,
                input_dim
            )

        # rnn components
        if rnn_type == "gru":
            self.rnn_cell = nn.GRU
        elif rnn_type == "lstm":
            self.rnn_cell = nn.LSTM
        self.rnn = self.rnn_cell(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            #dropout=dropout_hidden,
        )

        # output components
        if use_attention:
            self.attention = Attention(
                query_dim=hidden_dim,
                context_dim=attn_dim,
                hidden_dim=hidden_dim
            )
        self.project_fc = nn.Linear(
            hidden_dim,
            input_dim
        )
        self.word_classifier = nn.Linear(
            input_dim,
            vocab_size
        )
        if tie_weights:
            self.word_classifier.weight = self.embedding.weight

        # variational dropout
        self.lockdrop = LockedDropout()

        # initialization
        self.init_weights()

    def init_weights(self):
        if self.feat_dim > 0:
            init_module_weights(self.feat_fc, 0.1)
        init_module_weights(self.rnn)
        init_module_weights(self.project_fc, 0.1)
        if not self.tie_weights:
            init_module_weights(self.word_classifier, 0.1)

    def init_hidden_states(self, batch_size, init_type):
        return init_rnn_hidden_states(
            batch_size=batch_size,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            bidirectional=False,
            rnn_type=self.rnn_type,
            init_type=init_type
        )

    def _step(self, inputs, hiddens, feats=None, attn_ctx=None, attn_mask=None):

        # dropout between words and embedding matrix
        # get word embeddings
        embedded = embedded_dropout(
            embed=self.embedding,
            words=inputs,
            dropout=self.dropout_emb if self.training else 0
        )

        # get rnn inputs
        rnn_inputs = embedded
        if self.feat_dim > 0:
            rnn_inputs = torch.cat([rnn_inputs, feats], dim=2)
            rnn_inputs = self.feat_fc(rnn_inputs)

        # dropout between rnn inputs and rnn
        rnn_inputs = self.lockdrop(rnn_inputs, self.dropout_input)

        # get rnn outputs
        outputs, new_hiddens = self.rnn(rnn_inputs, hiddens)

        # dropout between rnn outputs and subsequent layers
        outputs = self.lockdrop(outputs, self.dropout_output)

        # compute attentions
        attns = None
        if self.use_attention:
            outputs, attns = self.attention(
                query=outputs,
                context=attn_ctx,
                mask=attn_mask
            )

        # get logits
        projected = self.project_fc(outputs)
        logits = self.word_classifier(projected)

        return {
            "outputs": outputs,
            "logits": logits,
            "hiddens": new_hiddens,
            "attns": attns,
        }

    def _step_decode(self, logits, gen_type, top_k=0, top_p=0.0, temp=1.0):
        logits = logits/temp

        if gen_type == DecoderRNN.GEN_GREEDY:
            symbol = logits.topk(1)[1]
        elif gen_type == DecoderRNN.GEN_SAMPLE:
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            symbol = dist.sample().unsqueeze(1)
        elif gen_type == DecoderRNN.GEN_TOP:
            top_k = min(top_k, logits.size(-1))
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float("inf")
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                #sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for batch_idx in range(logits.size(0)):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    logits[batch_idx, indices_to_remove] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            symbol = dist.sample().unsqueeze(1)

        elif gen_type == DecoderRNN.GEN_BEAM:
            raise Exception("unsupported generation type {}".format(gen_type))

        return {
            "symbol": symbol,
        }

    def forward(self, batch_size, inputs=None, hiddens=None, feats=None,
        attn_ctx=None, attn_mask=None,
        mode="teacher forcing", gen_type="greedy", beam_size=1,
        top_k=1, top_p=1.0, temp=1.0):

        ret_dict = {
            "outputs": None,
            "logits": None,
            "hiddens": None,
            "attn": None
        }

        ## unrolling over steps
        # use automatic unrolling in standard teacher forcing
        if mode == DecoderRNN.MODE_TEACHER_FORCE:
            step_ret_dict = self._step(
                inputs=inputs,
                hiddens=hiddens,
                feats=feats,
                attn_ctx=attn_ctx,
                attn_mask=attn_mask
            )
            ret_dict["outputs"] = step_ret_dict["outputs"]
            ret_dict["logits"] = step_ret_dict["logits"]
            ret_dict["hiddens"] = step_ret_dict["hiddens"]
            ret_dict["attns"] = step_ret_dict["attns"]
        # manual unrolling in free running mode
        elif mode == DecoderRNN.MODE_FREE_RUN:
            # lists for collecting step products
            step_output_lst = []
            step_logit_lst = []
            step_attn_lst = []
            step_symbol_lst = []

            # bos input for the first step
            bos_input = torch.LongTensor([self.bos_id]).to(DEVICE)
            bos_input.requires_grad_(False)
            step_input = bos_input.expand(batch_size, 1)

            # hidden for the first step
            step_hidden = hiddens

            # unrolling
            for step_idx in range(self.max_len):
                # some inputs to the current step
                if feats is None:
                    step_feat = None
                else:
                    step_feat = feats[:, step_idx, :].unsqueeze(1)
                # take a step
                step_ret_dict = self._step(
                    inputs=step_input,
                    hiddens=step_hidden,
                    feats=step_feat,
                    attn_ctx=attn_ctx,
                    attn_mask=attn_mask
                )
                # get inputs to next step
                decode_dict = self._step_decode(
                    logits=step_ret_dict["logits"].squeeze(1),
                    gen_type=gen_type,
                    top_k=top_k,
                    top_p=top_p,
                    temp=temp,
                )
                step_input = decode_dict["symbol"]
                step_hidden = step_ret_dict["hiddens"]

                # collect step productions
                step_output_lst.append(step_ret_dict["outputs"])
                step_logit_lst.append(step_ret_dict["logits"])
                step_attn_lst.append(step_ret_dict["attns"])
                step_symbol_lst.append(step_input)

            # organize step products
            outputs = torch.cat(step_output_lst, dim=1)
            logits = torch.cat(step_logit_lst, dim=1)
            if self.use_attention:
                attns = torch.cat(step_attn_lst, dim=1)
            else:
                attns = None
            symbols = torch.cat(step_symbol_lst, dim=1)
            ret_dict["outputs"] = outputs
            ret_dict["logits"] = logits
            ret_dict["hiddens"] = step_hidden
            ret_dict["attns"] = attns
            ret_dict["symbols"] = symbols

        return ret_dict










