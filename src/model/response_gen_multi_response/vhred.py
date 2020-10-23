import re
import code
import math
import json
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.modules.encoders import EncoderRNN
from model.modules.decoders import DecoderRNN
from model.modules.submodules import AbsFloorEmbEncoder, RelFloorEmbEncoder
from model.modules.submodules import GaussianVariation, GMMVariation, LGMVariation
from model.modules.utils import init_module_weights, init_word_embedding, gaussian_kld

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VHRED(nn.Module):
    def __init__(self, config, tokenizer):
        super(VHRED, self).__init__()

        # Attributes
        # Attributes from config
        self.word_embedding_dim = config.word_embedding_dim
        self.attr_embedding_dim = config.attr_embedding_dim
        self.sent_encoder_hidden_dim = config.sent_encoder_hidden_dim
        self.n_sent_encoder_layers = config.n_sent_encoder_layers
        self.dial_encoder_hidden_dim = config.dial_encoder_hidden_dim
        self.n_dial_encoder_layers = config.n_dial_encoder_layers
        self.latent_variable_dim = config.latent_dim
        self.decoder_hidden_dim = config.decoder_hidden_dim
        self.n_decoder_layers = config.n_decoder_layers
        self.use_attention = config.use_attention
        self.decode_max_len = config.decode_max_len
        self.tie_weights = config.tie_weights
        self.rnn_type = config.rnn_type
        self.gen_type = config.gen_type
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.temp = config.temp
        self.word_embedding_path = config.word_embedding_path
        self.floor_encoder_type = config.floor_encoder
        self.gaussian_mix_type = config.gaussian_mix_type
        # Optional attributes from config
        self.use_bow_loss = config.use_bow_loss if hasattr(config, "use_bow_loss") else True
        self.dropout = config.dropout if hasattr(config, "dropout") else 0.0
        self.dropout_emb = config.dropout if hasattr(config, "dropout") else 0.0
        self.dropout_input = config.dropout if hasattr(config, "dropout") else 0.0
        self.dropout_hidden = config.dropout if hasattr(config, "dropout") else 0.0
        self.dropout_output = config.dropout if hasattr(config, "dropout") else 0.0
        self.use_pretrained_word_embedding = config.use_pretrained_word_embedding if hasattr(config, "use_pretrained_word_embedding") else False
        self.n_step_annealing = config.n_step_annealing if hasattr(config, "n_step_annealing") else 1
        self.n_components = config.n_components if hasattr(config, "n_components") else 1
        # Other attributes
        self.vocab_size = len(tokenizer.word2id)
        self.word2id = tokenizer.word2id
        self.id2word = tokenizer.id2word
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

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
            bidirectional=False,
            rnn_type=self.rnn_type,
        )

        # Variational components
        if config.n_components == 1:
            self.prior_net = GaussianVariation(
                input_dim=self.dial_encoder_hidden_dim,
                z_dim=self.latent_variable_dim,
                # large_mlp=True
            )
        elif config.n_components > 1:
            if self.gaussian_mix_type == "gmm":
                self.prior_net = GMMVariation(
                    input_dim=self.dial_encoder_hidden_dim,
                    z_dim=self.latent_variable_dim,
                    n_components=self.n_components,
                )
            elif self.gaussian_mix_type == "lgm":
                self.prior_net = LGMVariation(
                    input_dim=self.dial_encoder_hidden_dim,
                    z_dim=self.latent_variable_dim,
                    n_components=self.n_components,
                )
        self.post_net = GaussianVariation(
            input_dim=self.sent_encoder_hidden_dim+self.dial_encoder_hidden_dim,
            z_dim=self.latent_variable_dim,
        )
        self.latent_to_bow = nn.Sequential(
            nn.Linear(
                self.latent_variable_dim+self.dial_encoder_hidden_dim,
                self.latent_variable_dim
            ),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(
                self.latent_variable_dim,
                self.vocab_size
            )
        )
        self.ctx_fc = nn.Sequential(
            nn.Linear(
                self.latent_variable_dim+self.dial_encoder_hidden_dim,
                self.dial_encoder_hidden_dim,
            ),
            nn.Tanh(),
            nn.Dropout(self.dropout)
        )

        # Decoding components
        self.enc2dec_hidden_fc = nn.Linear(
            self.dial_encoder_hidden_dim,
            self.n_decoder_layers*self.decoder_hidden_dim if self.rnn_type == "gru" else self.n_decoder_layers*self.decoder_hidden_dim*2
        )
        self.decoder = DecoderRNN(
            vocab_size=len(self.word2id),
            input_dim=self.word_embedding_dim,
            hidden_dim=self.decoder_hidden_dim,
            feat_dim=self.dial_encoder_hidden_dim,
            n_layers=self.n_decoder_layers,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            max_len=self.decode_max_len,
            dropout_emb=self.dropout_emb,
            dropout_input=self.dropout_input,
            dropout_hidden=self.dropout_hidden,
            dropout_output=self.dropout_output,
            embedding=self.word_embedding,
            tie_weights=self.tie_weights,
            rnn_type=self.rnn_type,
            use_attention=self.use_attention,
            attn_dim=self.sent_encoder_hidden_dim
        )

        # Extra components
        # Floor encoding
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
        init_module_weights(self.enc2dec_hidden_fc)
        init_module_weights(self.latent_to_bow)
        init_module_weights(self.ctx_fc)

    def _init_dec_hiddens(self, context):
        batch_size = context.size(0)

        hiddens = self.enc2dec_hidden_fc(context)
        if self.rnn_type == "gru":
            hiddens = hiddens.view(
                batch_size,
                self.n_decoder_layers,
                self.decoder_hidden_dim
            ).transpose(0, 1).contiguous()  # (n_layers, batch_size, hidden_dim)
        elif self.rnn_type == "lstm":
            hiddens = hiddens.view(
                batch_size,
                self.n_decoder_layers,
                self.decoder_hidden_dim,
                2
            )
            h = hiddens[:,:,:,0].transpose(0, 1).contiguous()  # (n_layers, batch_size, hidden_dim)
            c = hiddens[:,:,:,1].transpose(0, 1).contiguous()  # (n_layers, batch_size, hidden_dim)
            hiddens = (h, c)

        return hiddens

    def _encode_sent(self, outputs):
        output_lens = (outputs != self.pad_token_id).sum(-1)
        _, _, sent_encodings = self.sent_encoder(outputs, output_lens)
        return sent_encodings

    def _encode_dial(self, inputs, input_floors, output_floors):
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

        dial_lens = (input_lens > 0).long().sum(1)  # equals number of non-padding sents
        _, _, dial_encodings = self.dial_encoder(sent_encodings, dial_lens)  # [batch_size, dial_encoder_dim]

        return word_encodings, sent_encodings, dial_encodings

    def _get_prior_z(self, prior_net_input, assign_k=None, return_pi=False):
        ret = self.prior_net(
            context=prior_net_input,
            assign_k=assign_k,
            return_pi=return_pi
        )

        return ret

    def _get_post_z(self, post_net_input):
        z, mu, var = self.post_net(post_net_input)
        return z, mu, var

    def _get_ctx_for_decoder(self, z, dial_encodings):
        ctx_encodings = self.ctx_fc(torch.cat([z, dial_encodings], dim=1))
        # ctx_encodings = self.ctx_fc(z)
        return ctx_encodings

    def _decode(self, inputs, context, attn_ctx=None, attn_mask=None):
        batch_size = context.size(0)
        hiddens = self._init_dec_hiddens(context)
        feats = None
        feats = context.unsqueeze(1).repeat(1, inputs.size(1), 1)
        ret_dict = self.decoder.forward(
            batch_size=batch_size,
            inputs=inputs,
            hiddens=hiddens,
            feats=feats,
            attn_ctx=attn_ctx,
            attn_mask=attn_mask,
            mode=DecoderRNN.MODE_TEACHER_FORCE
        )

        return ret_dict

    def _sample(self, context, attn_ctx=None, attn_mask=None):
        batch_size = context.size(0)
        hiddens = self._init_dec_hiddens(context)
        feats = None
        feats = context.unsqueeze(1).repeat(1, self.decode_max_len, 1)
        ret_dict = self.decoder.forward(
            batch_size=batch_size,
            hiddens=hiddens,
            feats=feats,
            attn_ctx=attn_ctx,
            attn_mask=attn_mask,
            mode=DecoderRNN.MODE_FREE_RUN,
            gen_type=self.gen_type,
            temp=self.temp,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        return ret_dict

    def _get_attn_mask(self, attn_keys):
        attn_mask = (attn_keys != self.pad_token_id)
        return attn_mask

    def _annealing_coef_term(self, step):
        return min(1.0, 1.0*step/self.n_step_annealing)

    def compute_active_units(self, data_source, batch_size, delta=0.01):
        with torch.no_grad():
            cnt = 0

            data_source.epoch_init()
            prior_mu_sum = 0
            post_mu_sum = 0
            while True:
                batch_data = data_source.next(batch_size)
                if batch_data is None:
                    break
                X, Y = batch_data["X"], batch_data["Y"]
                X_floor, Y_floor = batch_data["X_floor"], batch_data["Y_floor"]
                # encode
                word_encodings, sent_encodings, dial_encodings = self._encode_dial(
                    inputs=X,
                    input_floors=X_floor,
                    output_floors=Y_floor
                )
                # prior
                prior_net_input = dial_encodings
                _, prior_mu, _ = self._get_prior_z(prior_net_input)
                # post
                post_sent_encodings = self._encode_sent(Y)
                post_net_input = torch.cat([post_sent_encodings, dial_encodings], dim=1)
                _, post_mu, _ = self._get_post_z(post_net_input)
                # record
                cnt += prior_mu.size(0)
                prior_mu_sum += prior_mu.sum(0)
                post_mu_sum += post_mu.sum(0)
            prior_mu_mean = (prior_mu_sum / cnt).unsqueeze(0)  # [1, latent_dim]
            post_mu_mean = (post_mu_sum / cnt).unsqueeze(0)  # [1, latent_dim]

            data_source.epoch_init()
            prior_mu_var_sum = 0
            post_mu_var_sum = 0
            while True:
                batch_data = data_source.next(batch_size)
                if batch_data is None:
                    break
                X, Y = batch_data["X"], batch_data["Y"]
                X_floor, Y_floor = batch_data["X_floor"], batch_data["Y_floor"]
                # encode
                word_encodings, sent_encodings, dial_encodings = self._encode_dial(
                    inputs=X,
                    input_floors=X_floor,
                    output_floors=Y_floor
                )
                # prior
                prior_net_input = dial_encodings
                _, prior_mu, _ = self._get_prior_z(prior_net_input)
                # post
                post_sent_encodings = self._encode_sent(Y)
                post_net_input = torch.cat([post_sent_encodings, dial_encodings], dim=1)
                _, post_mu, _ = self._get_post_z(post_net_input)
                # record
                prior_mu_var_sum += ((prior_mu - prior_mu_mean)**2).sum(0)
                post_mu_var_sum += ((post_mu - post_mu_mean)**2).sum(0)
            prior_mu_var_mean = prior_mu_var_sum / (cnt-1)  # [latent_dim]
            post_mu_var_mean = post_mu_var_sum / (cnt-1)  # [latent_dim]

            prior_au = (prior_mu_var_mean >= delta).sum().item()
            post_au = (post_mu_var_mean >= delta).sum().item()
            prior_au_ratio = prior_au / self.latent_variable_dim
            post_au_ratio = post_au / self.latent_variable_dim
            return {
                "prior_au": prior_au,
                "post_au": post_au,
                "prior_au_ratio": prior_au_ratio,
                "post_au_ratio": post_au_ratio,
            }

    def load_model(self, model_path):
        """Load pretrained model weights from model_path

        Arguments:
            model_path {str} -- path to pretrained model weights
        """
        pretrained_state_dict = torch.load(
            model_path,
            map_location=lambda storage, loc: storage
        )
        self.load_state_dict(pretrained_state_dict)

    def train_step(self, data, step):
        """One training step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of context sentences
                'Y' {LongTensor [batch_size, max_y_sent_len]} -- token ids of response sentence
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
        X, Y = data["X"], data["Y"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_in = Y[:, :-1].contiguous()
        Y_out = Y[:, 1:].contiguous()

        batch_size = X.size(0)
        max_y_len = Y_out.size(1)

        # Forward
        # Get prior z
        word_encodings, sent_encodings, dial_encodings = self._encode_dial(
            inputs=X,
            input_floors=X_floor,
            output_floors=Y_floor
        )
        prior_net_input = dial_encodings
        prior_z, prior_mu, prior_var = self._get_prior_z(prior_net_input)
        # Get post z
        post_sent_encodings = self._encode_sent(Y)
        post_net_input = torch.cat([post_sent_encodings, dial_encodings], dim=1)
        post_z, post_mu, post_var = self._get_post_z(post_net_input)
        # Decode
        ctx_encodings = self._get_ctx_for_decoder(post_z, dial_encodings)
        attn_ctx = word_encodings.view(batch_size, -1, word_encodings.size(-1))
        attn_mask = self._get_attn_mask(X.view(batch_size, -1))
        decoder_ret_dict = self._decode(
            inputs=Y_in,
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
            Y_out.view(-1),
            ignore_index=self.pad_token_id,
            reduction="none"
        ).view(batch_size, max_y_len)
        sent_loss = word_losses.sum(1).mean(0)
        loss += sent_loss
        with torch.no_grad():
            ppl = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                Y_out.view(-1),
                ignore_index=self.pad_token_id,
                reduction="mean"
            ).exp()
        # KLD
        kld_coef = self._annealing_coef_term(step)
        kld_losses = gaussian_kld(
            post_mu,
            post_var,
            prior_mu,
            prior_var,
        )
        avg_kld = kld_losses.mean()
        loss += avg_kld*kld_coef
        # BOW
        if self.use_bow_loss:
            Y_out_mask = (Y_out != self.pad_token_id).float()
            bow_input = torch.cat([post_z, dial_encodings], dim=1)
            bow_logits = self.latent_to_bow(bow_input)  # [batch_size, vocab_size]
            bow_loss = -F.log_softmax(bow_logits, dim=1).gather(1, Y_out) * Y_out_mask
            bow_loss = bow_loss.sum(1).mean()
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

    def evaluate_step(self, data, assign_k=None):
        """One training step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of context sentences
                'Y' {LongTensor [batch_size, max_y_sent_len]} -- token ids of response sentence
                'Y_floor' {LongTensor [batch_size]} -- floor of response sentence

        Returns:
            dict of data -- returned keys and values

            dict of statistics -- returned keys and values
                'ppl' {float} -- perplexity
                'kld' {float} -- KLD
                'monitor' {float} -- a monitor number for learning rate scheduling
        """
        X, Y = data["X"], data["Y"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_in = Y[:, :-1].contiguous()
        Y_out = Y[:, 1:].contiguous()

        batch_size = X.size(0)
        max_y_len = Y_out.size(1)

        with torch.no_grad():
            # Forward
            # Get prior z
            word_encodings, sent_encodings, dial_encodings = self._encode_dial(
                inputs=X,
                input_floors=X_floor,
                output_floors=Y_floor
            )
            prior_net_input = dial_encodings
            prior_z, prior_mu, prior_var = self._get_prior_z(
                prior_net_input=prior_net_input,
                assign_k=assign_k
            )
            # Get post z
            post_sent_encodings = self._encode_sent(Y)
            post_net_input = torch.cat([post_sent_encodings, dial_encodings], dim=1)
            post_z, post_mu, post_var = self._get_post_z(post_net_input)
            # Decode from post z
            attn_ctx = word_encodings.view(batch_size, -1, word_encodings.size(-1))
            attn_mask = self._get_attn_mask(X.view(batch_size, -1))
            post_ctx_encodings = self._get_ctx_for_decoder(post_z, dial_encodings)
            post_decoder_ret_dict = self._decode(
                inputs=Y_in,
                context=post_ctx_encodings,
                attn_ctx=attn_ctx,
                attn_mask=attn_mask
            )
            # Decode from prior z
            prior_ctx_encodings = self._get_ctx_for_decoder(prior_z, dial_encodings)
            prior_decoder_ret_dict = self._decode(
                inputs=Y_in,
                context=prior_ctx_encodings,
                attn_ctx=attn_ctx,
                attn_mask=attn_mask
            )

            # Loss
            # Reconstruction
            post_word_losses = F.cross_entropy(
                post_decoder_ret_dict["logits"].view(-1, self.vocab_size),
                Y_out.view(-1),
                ignore_index=self.decoder.pad_token_id,
                reduction="none"
            ).view(batch_size, max_y_len)
            post_sent_loss = post_word_losses.sum(1).mean(0)
            post_ppl = F.cross_entropy(
                post_decoder_ret_dict["logits"].view(-1, self.vocab_size),
                Y_out.view(-1),
                ignore_index=self.decoder.pad_token_id,
                reduction="mean"
            ).exp()
            # Generation
            prior_word_losses = F.cross_entropy(
                prior_decoder_ret_dict["logits"].view(-1, self.vocab_size),
                Y_out.view(-1),
                ignore_index=self.decoder.pad_token_id,
                reduction="none"
            ).view(batch_size, max_y_len)
            prior_sent_loss = prior_word_losses.sum(1).mean(0)
            prior_ppl = F.cross_entropy(
                prior_decoder_ret_dict["logits"].view(-1, self.vocab_size),
                Y_out.view(-1),
                ignore_index=self.decoder.pad_token_id,
                reduction="mean"
            ).exp()
            # KLD
            kld_losses = gaussian_kld(
                post_mu,
                post_var,
                prior_mu,
                prior_var
            )
            avg_kld = kld_losses.mean()
            # monitor
            monitor_loss = post_sent_loss + avg_kld

        # return dicts
        ret_data = {}
        ret_stat = {
            "post_ppl": post_ppl.item(),
            "prior_ppl": prior_ppl.item(),
            "kld": avg_kld.item(),
            "post_abs_mu_mean": post_mu.abs().mean().item(),
            "post_var_mean": post_var.mean().item(),
            "prior_abs_mu_mean": prior_mu.abs().mean().item(),
            "prior_var_mean": prior_var.mean().item(),
            "monitor": monitor_loss.item()
        }

        return ret_data, ret_stat

    def test_step(self, data, assign_k=None):
        """One test step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of context sentences
                'Y_floor' {LongTensor [batch_size]} -- floor of response sentence

        Returns:
            dict of data -- returned keys and values
                'symbols' {LongTensor [batch_size, max_decode_len]} -- token ids of response hypothesis
            dict of statistics -- returned keys and values

        """
        X = data["X"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        batch_size = X.size(0)

        with torch.no_grad():
            # Forward
            # Get prior z
            word_encodings, sent_encodings, dial_encodings = self._encode_dial(
                inputs=X,
                input_floors=X_floor,
                output_floors=Y_floor
            )
            prior_net_input = dial_encodings
            prior_z, prior_mu, prior_var, prior_pi = self._get_prior_z(
                prior_net_input=prior_net_input,
                assign_k=assign_k,
                return_pi=True
            )
            # Decode
            ctx_encodings = self._get_ctx_for_decoder(prior_z, dial_encodings)
            attn_ctx = word_encodings.view(batch_size, -1, word_encodings.size(-1))
            attn_mask = self._get_attn_mask(X.view(batch_size, -1))
            decoder_ret_dict = self._sample(
                context=ctx_encodings,
                attn_ctx=attn_ctx,
                attn_mask=attn_mask
            )

        ret_data = {
            "symbols": decoder_ret_dict["symbols"],
            "pi": prior_pi
        }
        ret_stat = {}

        return ret_data, ret_stat
