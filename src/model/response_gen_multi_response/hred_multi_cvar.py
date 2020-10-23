import math
import code
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.response_gen_multi_response.hred import HRED

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HRED_Multi_CVaR(HRED):
    def __init__(self, config, tokenizer):
        super(HRED_Multi_CVaR, self).__init__(config, tokenizer)

        self.n_samples = config.n_z_samples if hasattr(config, "n_z_samples") else 1
        self.alpha = 0.3
        self.tokenizer = tokenizer

    def train_step(self, data):
        """One training step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of context sentences
                'Y_nested' {LongTensor [batch_size, n_ref, max_y_sent_len]]} -- token ids of multiple response sentences
                'Y_floor' {LongTensor [batch_size]} -- floor of response sentence

        Returns:
            dict of data -- returned keys and values
                'loss' {FloatTensor []} -- loss to backword
            dict of statistics -- returned keys and values
                'ppl' {float} -- perplexity
                'loss' {float} -- batch loss
        """
        X = data["X"]
        Y = data["Y_nested"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]

        batch_size, n_refs, _ = Y.size()

        Y = Y.view(batch_size*n_refs, -1)
        Y_in = Y[:, :-1].contiguous()
        Y_out = Y[:, 1:].contiguous()

        max_y_len = Y_out.size(1)

        # Forward
        # -- encode
        word_encodings, sent_encodings, dial_encodings = self._encode(
            inputs=X,
            input_floors=X_floor,
            output_floors=Y_floor
        )
        # -- repeat encodings
        repeat_dial_encodings = dial_encodings.unsqueeze(1).repeat(1, n_refs, 1).view(batch_size*n_refs, -1)  # [batch_size*n_refs, dial_hidden_dim]
        repeat_word_encodings = word_encodings.unsqueeze(1).repeat(1, n_refs, 1, 1, 1).view(batch_size*n_refs, *word_encodings.size()[1:])  # [batch_size*n_refs, history_len, max_x_len, emb_dim]
        repeat_X = X.unsqueeze(1).repeat(1, n_refs, 1, 1).view(batch_size*n_refs, *X.size()[1:])  # [batch_size*n_refs, history_len, max_x_len]
        # -- decode
        attn_ctx = repeat_word_encodings.view(batch_size*n_refs, -1, repeat_word_encodings.size(-1))
        attn_mask = self._get_attn_mask(repeat_X.view(batch_size*n_refs, -1))
        decoder_ret_dict = self._decode(
            inputs=Y_in,
            context=repeated_dial_encodings,
            attn_ctx=attn_ctx,
            attn_mask=attn_mask
        )

        # Loss
        loss = 0
        # -- CVaR
        word_nll = F.cross_entropy(
            decoder_ret_dict["logits"].view(-1, self.vocab_size),
            Y_out.view(-1),
            ignore_index=self.decoder.pad_token_id,
            reduction="none"
        ).view(batch_size*n_refs, max_y_len)
        sent_nll = word_nll.sum(-1)
        neglog_alpha = (-1) * math.log(self.alpha)
        CVaR_losses = []
        cur_start_idx = 0
        for batch_idx in range(batch_size):
            N_r = len(Y_multi_lst[batch_idx])

            cur_sent_nll = sent_nll[cur_start_idx:cur_start_idx+N_r]
            worst_sent_nll = torch.FloatTensor([0.0]).to(DEVICE)
            for nll in cur_sent_nll:
                if nll >= neglog_alpha:
                    worst_sent_nll += nll
            CVaR_losses.append(worst_sent_nll)
            
            cur_start_idx += N_r
        CVaR_losses = torch.cat(CVaR_losses, dim=0)
        CVaR_loss = CVaR_losses.mean(0) / (1-self.alpha)
        loss += CVaR_loss
        # -- ppl
        with torch.no_grad():
            ppl = F.cross_entropy(
                decoder_ret_dict["logits"].view(-1, self.vocab_size),
                Y_out.view(-1),
                ignore_index=self.decoder.pad_token_id,
                reduction="mean"
            ).exp()

        # return dicts
        ret_data = {
            "loss": loss
        }
        ret_stat = {
            "ppl": ppl.item(),
            "CVaR_loss": CVaR_loss.item(),
        }

        return ret_data, ret_stat

