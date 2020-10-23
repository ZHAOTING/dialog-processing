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
from model.modules.utils import init_module_weights, init_word_embedding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Mechanism_HRED(nn.Module):
    def __init__(self, config, tokenizer):
        super(Mechanism_HRED, self).__init__()

        # Attributes
        # Attributes from config
        self.word_embedding_dim = config.word_embedding_dim
        self.attr_embedding_dim = config.attr_embedding_dim
        self.sent_encoder_hidden_dim = config.sent_encoder_hidden_dim
        self.n_sent_encoder_layers = config.n_sent_encoder_layers
        self.dial_encoder_hidden_dim = config.dial_encoder_hidden_dim
        self.n_dial_encoder_layers = config.n_dial_encoder_layers
        self.decoder_hidden_dim = config.decoder_hidden_dim
        self.n_decoder_layers = config.n_decoder_layers
        self.n_mechanisms = config.n_mechanisms
        self.latent_dim = config.latent_dim
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
        # Optional attributes from config
        self.dropout_emb = config.dropout if hasattr(config, "dropout") else 0.0
        self.dropout_input = config.dropout if hasattr(config, "dropout") else 0.0
        self.dropout_hidden = config.dropout if hasattr(config, "dropout") else 0.0
        self.dropout_output = config.dropout if hasattr(config, "dropout") else 0.0
        self.use_pretrained_word_embedding = config.use_pretrained_word_embedding if hasattr(config, "use_pretrained_word_embedding") else False
        # Other attributes
        self.word2id = tokenizer.word2id
        self.id2word = tokenizer.id2word
        self.vocab_size = len(tokenizer.word2id)
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

        # Mechanism components
        self.mechanism_embeddings = nn.Embedding(
            self.n_mechanisms,
            self.latent_dim,
        )
        self.ctx2mech_fc = nn.Linear(
            self.dial_encoder_hidden_dim,
            self.latent_dim
        )
        self.score_bilinear = nn.Parameter(
            torch.FloatTensor(self.latent_dim, self.latent_dim)
        )
        self.ctx_mech_combine_fc = nn.Linear(
            self.dial_encoder_hidden_dim+self.latent_dim,
            self.dial_encoder_hidden_dim
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

    def _init_weights(self):
        init_module_weights(self.mechanism_embeddings)
        init_module_weights(self.ctx2mech_fc)
        init_module_weights(self.score_bilinear)
        init_module_weights(self.ctx_mech_combine_fc)
        init_module_weights(self.enc2dec_hidden_fc)

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
            h = hiddens[:, :, :, 0].transpose(0, 1).contiguous()  # (n_layers, batch_size, hidden_dim)
            c = hiddens[:, :, :, 1].transpose(0, 1).contiguous()  # (n_layers, batch_size, hidden_dim)
            hiddens = (h, c)

        return hiddens

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

        dial_lens = (input_lens > 0).long().sum(1)  # equals number of non-padding sents
        _, _, dial_encodings = self.dial_encoder(sent_encodings, dial_lens)  # [batch_size, dial_encoder_dim]

        return word_encodings, sent_encodings, dial_encodings

    def _compute_mechanism_probs(self, ctx_encodings):
        ctx_mech = self.ctx2mech_fc(ctx_encodings)  # [batch_size, latent_dim]
        mech_scores = torch.matmul(
            torch.matmul(ctx_mech, self.score_bilinear),
            self.mechanism_embeddings.weight.T
        )  # [batch_size, n_mechanisms]
        mech_probs = F.softmax(mech_scores, dim=1)

        return mech_probs

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

        # Forward
        # -- encode
        word_encodings, sent_encodings, dial_encodings = self._encode(
            inputs=X,
            input_floors=X_floor,
            output_floors=Y_floor
        )
        attn_ctx = word_encodings.view(batch_size, -1, word_encodings.size(-1))
        attn_mask = self._get_attn_mask(X.view(batch_size, -1))
        mech_probs = self._compute_mechanism_probs(dial_encodings)  # [batch_size, n_mechanisms]
        mech_embed_inputs = torch.LongTensor(list(range(self.n_mechanisms))).to(DEVICE)  # [n_mechanisms]
        repeated_mech_embed_inputs = mech_embed_inputs.unsqueeze(0).repeat(batch_size, 1).view(-1)  # [batch_size*n_mechanisms]
        repeated_mech_embeds = self.mechanism_embeddings(repeated_mech_embed_inputs)  # [batch_size*n_mechanisms, latent_dim]
        # -- repeat for each mechanism
        repeated_Y_in = Y_in.unsqueeze(1).repeat(1, self.n_mechanisms, 1)  # [batch_size, n_mechanisms, len]
        repeated_Y_in = repeated_Y_in.view(batch_size*self.n_mechanisms, -1)
        repeated_Y_out = Y_out.unsqueeze(1).repeat(1, self.n_mechanisms, 1)  # [batch_size, n_mechanisms, len]
        repeated_Y_out = repeated_Y_out.view(batch_size*self.n_mechanisms, -1)
        dial_encodings = dial_encodings.unsqueeze(1).repeat(1, self.n_mechanisms, 1)  # [batch_size, n_mechanisms, hidden_dim]
        dial_encodings = dial_encodings.view(batch_size*self.n_mechanisms, self.dial_encoder_hidden_dim)
        attn_ctx = attn_ctx.unsqueeze(1).repeat(1, self.n_mechanisms, 1, 1)
        attn_ctx = attn_ctx.view(batch_size*self.n_mechanisms, -1, attn_ctx.size(-1))
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_mechanisms, 1)
        attn_mask = attn_mask.view(batch_size*self.n_mechanisms, -1)
        # -- decode
        dec_ctx = self.ctx_mech_combine_fc(torch.cat([dial_encodings, repeated_mech_embeds], dim=1))
        decoder_ret_dict = self._decode(
            inputs=repeated_Y_in,
            context=dec_ctx,
            attn_ctx=attn_ctx,
            attn_mask=attn_mask
        )

        # Calculate loss
        loss = 0
        word_neglogll = F.cross_entropy(
            decoder_ret_dict["logits"].view(-1, self.vocab_size),
            repeated_Y_out.view(-1),
            ignore_index=self.decoder.pad_token_id,
            reduction="none"
        ).view(batch_size, self.n_mechanisms, -1)
        sent_logll = word_neglogll.sum(2) * (-1)
        mech_logll = (mech_probs+1e-10).log()
        sent_mech_logll = sent_logll + mech_logll
        target_logll = torch.logsumexp(sent_mech_logll, dim=1)
        target_neglogll = target_logll * (-1)
        loss = target_neglogll.mean()

        with torch.no_grad():
            ppl = F.cross_entropy(
                decoder_ret_dict["logits"].view(-1, self.vocab_size),
                repeated_Y_out.view(-1),
                ignore_index=self.decoder.pad_token_id,
                reduction="mean"
            ).exp()

        # return dicts
        ret_data = {
            "loss": loss
        }
        ret_stat = {
            "ppl": ppl.item(),
            "loss": loss.item(),
            "mech_prob_std": mech_probs.std(1).mean().item(),
            "mech_prob_max": mech_probs.max(1)[0].mean().item()
        }

        return ret_data, ret_stat

    def evaluate_step(self, data):
        """One evaluation step

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
                'monitor' {float} -- a monitor number for learning rate scheduling
        """
        X, Y = data["X"], data["Y"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_in = Y[:, :-1].contiguous()
        Y_out = Y[:, 1:].contiguous()

        batch_size = X.size(0)

        with torch.no_grad():
            # Forward
            word_encodings, sent_encodings, dial_encodings = self._encode(
                inputs=X,
                input_floors=X_floor,
                output_floors=Y_floor
            )
            attn_ctx = word_encodings.view(batch_size, -1, word_encodings.size(-1))
            attn_mask = self._get_attn_mask(X.view(batch_size, -1))
            mech_probs = self._compute_mechanism_probs(dial_encodings)  # [batch_size, n_mechanisms]
            mech_embed_inputs = mech_probs.argmax(1)  # [batch_size]
            mech_embeds = self.mechanism_embeddings(mech_embed_inputs)
            dec_ctx = self.ctx_mech_combine_fc(torch.cat([dial_encodings, mech_embeds], dim=1))
            decoder_ret_dict = self._decode(
                inputs=Y_in,
                context=dec_ctx,
                attn_ctx=attn_ctx,
                attn_mask=attn_mask
            )

            # Loss
            word_loss = F.cross_entropy(
                decoder_ret_dict["logits"].view(-1, self.vocab_size),
                Y_out.view(-1),
                ignore_index=self.decoder.pad_token_id,
                reduction="mean"
            )
            ppl = torch.exp(word_loss)

        # return dicts
        ret_data = {}
        ret_stat = {
            "ppl": ppl.item(),
            "monitor": ppl.item()
        }

        return ret_data, ret_stat

    def test_step(self, data, sample_from="dist"):
        """One test step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of context sentences
                'Y_floor' {LongTensor [batch_size]} -- floor of response sentence

            sample_from {str} -- "dist": sample mechanism from computed probabilities
                                "random": sample mechanisms uniformly
        Returns:
            dict of data -- returned keys and values
                'symbols' {LongTensor [batch_size, max_decode_len]} -- token ids of response hypothesis
            dict of statistics -- returned keys and values

        """
        X, Y = data["X"], data["Y"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        batch_size = X.size(0)

        with torch.no_grad():
            # Forward
            word_encodings, sent_encodings, dial_encodings = self._encode(
                inputs=X,
                input_floors=X_floor,
                output_floors=Y_floor
            )
            attn_ctx = word_encodings.view(batch_size, -1, word_encodings.size(-1))
            attn_mask = self._get_attn_mask(X.view(batch_size, -1))
            if sample_from == "dist":
                mech_probs = self._compute_mechanism_probs(dial_encodings)  # [batch_size, n_mechanisms]
                mech_dist = torch.distributions.Categorical(mech_probs)
                mech_embed_inputs = mech_dist.sample()  # [batch_size]
            elif sample_from == "random":
                mech_embed_inputs = [random.randint(0, self.n_mechanisms-1) for _ in range(batch_size)]
                mech_embed_inputs = torch.LongTensor(mech_embed_inputs).to(DEVICE)  # [batch_size]
            mech_embeds = self.mechanism_embeddings(mech_embed_inputs)
            dec_ctx = self.ctx_mech_combine_fc(torch.cat([dial_encodings, mech_embeds], dim=1))
            decoder_ret_dict = self._sample(
                context=dec_ctx,
                attn_ctx=attn_ctx,
                attn_mask=attn_mask
            )

        ret_data = {
            "symbols": decoder_ret_dict["symbols"]
        }
        ret_stat = {}

        return ret_data, ret_stat
