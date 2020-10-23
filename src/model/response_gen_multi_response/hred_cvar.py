import math
import code
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.response_gen_multi_response.hred import HRED

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HRED_CVaR(HRED):
    def __init__(self, config, tokenizer):
        super(HRED_CVaR, self).__init__(config, tokenizer)

        self.alpha = 0.3
        self.tokenizer = tokenizer

    def train_step(self, data):
        """One training step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of context sentences
                'Y' {LongTensor [batch_size, max_y_sent_len]} -- token ids of response sentence
                'Y_floor' {LongTensor [batch_size]} -- floor of response sentence

        Returns:
            dict of data -- returned keys and values
                'loss' {FloatTensor []} -- loss to backword
            dict of statistics -- returned keys and values
                'ppl' {float} -- perplexity
                'loss' {float} -- batch loss
        """
        X, Y = data["X"], data["Y"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_in = Y[:, :-1].contiguous()
        Y_out = Y[:, 1:].contiguous()

        batch_size = X.size(0)
        max_y_len = Y_out.size(1)

        # Forward
        word_encodings, sent_encodings, dial_encodings = self._encode(
            inputs=X,
            input_floors=X_floor,
            output_floors=Y_floor
        )
        attn_ctx = word_encodings.view(batch_size, -1, word_encodings.size(-1))
        attn_mask = self._get_attn_mask(X.view(batch_size, -1))
        decoder_ret_dict = self._decode(
            inputs=Y_in,
            context=dial_encodings,
            attn_ctx=attn_ctx,
            attn_mask=attn_mask
        )

        # Calculate loss
        loss = 0
        logits = decoder_ret_dict["logits"]
        word_losses = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            Y_out.view(-1),
            ignore_index=self.pad_token_id,
            reduction="none"
        ).view(batch_size, max_y_len)
        sent_nlls = word_losses.sum(1)
        neglog_alpha = (-1) * math.log(self.alpha)
        CVaR_losses = []
        for sent_nll in sent_nlls:
            if sent_nll >= neglog_alpha:
                CVaR_losses.append(sent_nlls)
        CVaR_losses = torch.stack(CVaR_losses)
        CVaR_loss = CVaR_losses.mean() / (1-self.alpha)
        loss += CVaR_loss
        with torch.no_grad():
            ppl = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                Y_out.view(-1),
                ignore_index=self.pad_token_id,
                reduction="mean"
            ).exp()

        # return dicts
        ret_data = {
            "loss": loss
        }
        ret_stat = {
            "ppl": ppl.item(),
            "CVaR_loss": CVaR_loss.item(),
            "loss": loss.item()
        }

        return ret_data, ret_stat
