import re
import code
import math
import json
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.response_gen_multi_response.vhred import VHRED
from model.modules.utils import gaussian_kld

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VHREDMultiAvg(VHRED):
    def __init__(self, config, tokenizer):
        super(VHREDMultiAvg, self).__init__(config, tokenizer)

        self.mbow_factor = 1.0

    def _annealing_coef_term(self, step):
        return min(1.0, 1.0*step/self.n_step_annealing)

    def train_step(self, data, step):
        """One training step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of context sentences
                'Y_nested' {LongTensor [batch_size, n_refs, max_y_sent_len]} -- token ids of multiple response sentences
                'Y_floor' {LongTensor [batch_size]} -- floor of response sentence

            step {int} -- the n-th optimization step

        Returns:
            dict of data -- returned keys and values
                'loss' {FloatTensor []} -- loss to backword
            dict of statistics -- returned keys and values
                'ppl' {float} -- perplexity
                'kld' {float} -- KLD
                'kld_term' {float} -- KLD annealing coefficient
                'bow_loss' {float} -- bag-of-word loss
                'loss' {float} -- batch loss
        """
        X, Y = data["X"], data["Y_nested"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_in = Y[:, :, :-1].contiguous()
        Y_out = Y[:, :, 1:].contiguous()

        batch_size = X.size(0)
        n_refs = Y.size(1)
        max_y_len = Y_out.size(2)

        Y_flat = Y.view(batch_size*n_refs, -1)  # [batch_size*n_refs, max_y_len]
        Y_in_flat = Y_in.view(batch_size*n_refs, -1)  # [batch_size*n_refs, max_y_len]
        Y_out_flat = Y_out.view(batch_size*n_refs, -1)  # [batch_size*n_refs, max_y_len]

        # Forward
        # -- encode
        word_encodings, sent_encodings, dial_encodings = self._encode_dial(
            inputs=X,
            input_floors=X_floor,
            output_floors=Y_floor
        )
        repeat_dial_encodings = dial_encodings.unsqueeze(1).repeat(1, n_refs, 1).view(batch_size*n_refs, -1)  # [batch_size*n_refs, dial_hidden_dim]
        repeat_word_encodings = word_encodings.unsqueeze(1).repeat(1, n_refs, 1, 1, 1).view(batch_size*n_refs, *word_encodings.size()[1:])  # [batch_size*n_refs, history_len, max_x_len, emb_dim]
        repeat_X = X.unsqueeze(1).repeat(1, n_refs, 1, 1).view(batch_size*n_refs, *X.size()[1:])  # [batch_size*n_refs, history_len, max_x_len]
        # -- get prior z
        prior_net_input = dial_encodings
        prior_z, prior_mu, prior_var = self._get_prior_z(
            prior_net_input=prior_net_input
        )  # [batch_size, *]
        repeated_prior_mu = prior_mu.unsqueeze(1).repeat(1, n_refs, 1).view(batch_size*n_refs, -1)
        repeated_prior_var = prior_var.unsqueeze(1).repeat(1, n_refs, 1).view(batch_size*n_refs, -1)
        # -- get post z
        post_sent_encodings = self._encode_sent(Y_flat)  # [batch_size*n_refs, sent_hidden_dim]
        post_net_input = post_sent_encodings
        post_net_input = torch.cat([post_sent_encodings, repeat_dial_encodings], dim=1)  # [batch_size*n_refs, sent_hidden_dim+dial_hidden_dim]
        post_z, post_mu, post_var = self._get_post_z(post_net_input)  # [batch_size*n_refs, *]
        # -- decode
        ctx_encodings = self._get_ctx_for_decoder(post_z, repeat_dial_encodings)
        attn_ctx = repeat_word_encodings.view(batch_size*n_refs, -1, repeat_word_encodings.size(-1))
        attn_mask = self._get_attn_mask(repeat_X.view(batch_size*n_refs, -1))
        decoder_ret_dict = self._decode(
            inputs=Y_in_flat,
            context=ctx_encodings,
            attn_ctx=attn_ctx,
            attn_mask=attn_mask
        )

        # Loss
        loss = 0
        # Reconstruction
        logits = decoder_ret_dict["logits"]
        word_losses = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            Y_out_flat.view(-1),
            ignore_index=self.pad_token_id,
            reduction="none"
        ).view(batch_size*n_refs, max_y_len)
        sent_loss = word_losses.sum(1).mean(0)
        loss += sent_loss
        with torch.no_grad():
            ppl = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                Y_out_flat.view(-1),
                ignore_index=self.pad_token_id,
                reduction="mean"
            ).exp()
        # KLD
        kld_coef = self._annealing_coef_term(step)
        kld_losses = gaussian_kld(
            post_mu,
            post_var,
            repeated_prior_mu,
            repeated_prior_var,
        )
        avg_kld = kld_losses.mean()
        loss += avg_kld*kld_coef
        # BOW
        if self.use_bow_loss:
            bow_losses = []

            bow_input = torch.cat([post_z, repeat_dial_encodings], dim=1)  # [batch_size*n_refs, *]
            bow_input = bow_input.view(batch_size, n_refs, -1)
            for batch_idx in range(batch_size):
                group_bow_input = bow_input[batch_idx]  # [n_refs, *]
                group_Y_out = Y_out[batch_idx]  # [n_refs, max_y_len]
                group_Y_out_mask = (group_Y_out != self.pad_token_id).float()

                bow_logits = self.latent_to_bow(group_bow_input)  # [n_refs, vocab_size]

                for src_ref_idx in range(n_refs):
                    src_bow_logits = bow_logits[src_ref_idx].unsqueeze(0)  # [1, vocab_size]
                    repeated_src_bow_logits = src_bow_logits.repeat(n_refs, 1)  # [n_refs, vocab_size]

                    bow_probs = F.softmax(repeated_src_bow_logits, dim=1).gather(1, group_Y_out) * group_Y_out_mask  # [n_refs, max_y_len]

                    src_bow_probs = bow_probs[src_ref_idx]  # [max_y_len]
                    src_bow_loss = -(src_bow_probs+1e-10).log().sum()

                    comp_bow_loss = 0
                    for comp_ref_idx in range(n_refs):
                        if comp_ref_idx != src_ref_idx:
                            comp_bow_probs = bow_probs[comp_ref_idx]
                            comp_bow_loss -= (1-comp_bow_probs+1e-10).log().sum()

                    bow_losses.append(src_bow_loss+(self.mbow_factor/n_refs)*comp_bow_loss)

            bow_losses = torch.stack(bow_losses)
            bow_loss = bow_losses.mean()
            loss += bow_loss

        # return dicts
        ret_data = {
            "loss": loss
        }
        ret_stat = {
            "ppl": ppl.item(),
            "kld_term": kld_coef,
            "kld": avg_kld.item(),
            "prior_abs_mu_mean": prior_mu.abs().mean().item(),
            "prior_var_mean": prior_var.mean().item(),
            "post_abs_mu_mean": post_mu.abs().mean().item(),
            "post_var_mean": post_var.mean().item(),
            "loss": loss.item()
        }
        if self.use_bow_loss:
            ret_stat["bow_loss"] = bow_loss.item()

        return ret_data, ret_stat

