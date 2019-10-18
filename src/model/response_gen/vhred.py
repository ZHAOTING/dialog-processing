import re
import code
import math
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from model.modules.encoders import EncoderRNN
from model.modules.decoders import DecoderRNN
from model.modules.submodules import AbsFloorEmbEncoder, RelFloorEmbEncoder
from model.modules.submodules import GaussianVariation
from model.modules.utils import init_module_weights, init_word_embedding, gaussian_kld

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VHRED(nn.Module):
    def __init__(self, config, tokenizer):
        super(VHRED, self).__init__()

        ## Attributes
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
        self.dropout_emb = config.dropout
        self.dropout_input = config.dropout
        self.dropout_hidden = config.dropout
        self.dropout_output = config.dropout
        self.tie_weights = config.tie_weights
        self.rnn_type = config.rnn_type
        self.gen_type = config.gen_type
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.temp = config.temp
        self.optimizer_type = config.optimizer
        self.init_lr = config.init_lr
        self.gradient_clip = config.gradient_clip
        self.l2_penalty = config.l2_penalty
        self.use_pretrained_word_embedding = config.use_pretrained_word_embedding
        self.word_embedding_path = config.word_embedding_path
        self.floor_encoder_type = config.floor_encoder
        # Other attributes
        self.vocab_size = len(tokenizer.word2id)
        self.word2id = tokenizer.word2id
        self.id2word = tokenizer.id2word
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.kld_anneal_till_n_step = 10000

        ## Input components
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

        ## Variational components
        self.prior_net = GaussianVariation(
            input_dim=self.dial_encoder_hidden_dim,
            z_dim=self.latent_variable_dim
        )
        self.post_net = GaussianVariation(
            input_dim=self.sent_encoder_hidden_dim+self.dial_encoder_hidden_dim,
            z_dim=self.latent_variable_dim
        )
        self.latent_to_bow = nn.Linear(
            self.latent_variable_dim,
            self.vocab_size
        )
        self.ctx_fc = nn.Linear(
            self.latent_variable_dim+self.dial_encoder_hidden_dim,
            self.dial_encoder_hidden_dim
        )

        ## Decoding components
        self.enc2dec_hidden_fc = nn.Linear(
            self.dial_encoder_hidden_dim,
            self.n_decoder_layers*self.decoder_hidden_dim if self.rnn_type == "gru"
                else self.n_decoder_layers*self.decoder_hidden_dim*2
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

        ## Extra components
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

        ## Initialization
        self._set_optimizer()
        self._print_model_stats()
        self._init_weights()

    def _set_optimizer(self):
        if self.optimizer_type == "adam":
            self.optimizer = optim.AdamW(
                self.parameters(),
                lr=self.init_lr,
                weight_decay=self.l2_penalty
            )
        elif self.optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.parameters(),
                lr=self.init_lr,
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
        init_module_weights(self.enc2dec_hidden_fc)
        init_module_weights(self.latent_to_bow)
        init_module_weights(self.ctx_fc)

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
            print("{}/{} pretrained word embedding in vocab".\
                format(in_vocab_cnt, self.vocab_size))
        else:
            weights = nn.Parameter(
                torch.FloatTensor(
                    self.vocab_size,
                    self.word_embedding_dim
                ).to(DEVICE)
            )
            torch.nn.init.uniform_(weights, -1.0, 1.0)
        weights[self.pad_token_id].data.fill_(0)
        return weights

    def _init_dec_hiddens(self, context):
        batch_size = context.size(0)

        hiddens = self.enc2dec_hidden_fc(context)
        if self.rnn_type == "gru":
            hiddens = hiddens.view(
                batch_size,
                self.n_decoder_layers,
                self.decoder_hidden_dim
            ).transpose(0, 1).contiguous() # (n_layers, batch_size, hidden_dim)
        elif self.rnn_type == "lstm":
            hiddens = hiddens.view(
                batch_size,
                self.n_decoder_layers,
                self.decoder_hidden_dim,
                2
            )
            h = hiddens[:,:,:,0].transpose(0, 1).contiguous() # (n_layers, batch_size, hidden_dim)
            c = hiddens[:,:,:,1].transpose(0, 1).contiguous() # (n_layers, batch_size, hidden_dim)
            hiddens = (h, c)

        return hiddens

    def _encode_post(self, outputs):
        output_lens = (outputs != self.pad_token_id).sum(-1)
        _, _, sent_encodings = self.sent_encoder(outputs, output_lens)
        return sent_encodings

    def _encode_prior(self, inputs, input_floors, output_floors):
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

        dialog_lens = (input_lens > 0).long().sum(1) # equals number of non-padding sents
        _, _, dialog_encodings = self.dial_encoder(sent_encodings, dialog_lens) # [batch_size, dialog_encoder_dim]

        return word_encodings, sent_encodings, dialog_encodings

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
        attn_mask = (attn_keys > 0)
        return attn_mask

    def _kld_coef_term(self, step):
        return min(1.0, 1.0*step/self.kld_anneal_till_n_step)

    def load_model(self, model_path):
        """Load pretrained model weights from model_path

        Arguments:
            model_path {str} -- path to pretrained model weights
        """
        if DEVICE == "cuda":
            pretrained_state_dict = torch.load(model_path)
        else:
            pretrained_state_dict = torch.load(model_path, \
                map_location=lambda storage, loc: storage)
        self.load_state_dict(pretrained_state_dict)

    def train_step(self, data, step):
        """One training step

        Arguments:
            data {dict of data} -- required keys and values:
                X {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                X_floor {LongTensor [batch_size, history_len]} -- floors of context sentences
                Y {LongTensor [batch_size, max_y_sent_len]} -- token ids of response sentence
                Y_floor {LongTensor [batch_size]} -- floor of response sentence

            step {int} -- the n-th optimization step

        Returns:
            dict of data -- returned keys and values
                loss {FloatTensor []} -- loss to backword
            dict of statistics -- returned keys and values
                ppl {float} -- perplexity
                kld {float} -- KLD
                kld_term {float} -- KLD annealing coefficient
                bow_loss {float} -- bag-of-word loss
                loss {float} -- batch loss
        """
        X, Y = data["X"], data["Y"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_in = Y[:, :-1].contiguous()
        Y_out = Y[:, 1:].contiguous()

        batch_size = X.size(0)
        max_y_sent_len = Y_out.size(1)

        ## Forward
        # Get prior z
        word_encodings, sent_encodings, dial_encodings = self._encode_prior(
            inputs=X,
            input_floors=X_floor,
            output_floors=Y_floor
        )
        prior_net_input = dial_encodings
        prior_z, prior_mu, prior_var = self.prior_net(prior_net_input)
        # Get post z
        post_sent_encodings = self._encode_post(Y)
        post_net_input = torch.cat([post_sent_encodings, dial_encodings], dim=1)
        post_z, post_mu, post_var = self.post_net(post_net_input)
        # Decode
        ctx_encodings = self.ctx_fc(torch.cat([post_z, dial_encodings], dim=1))
        attn_ctx = word_encodings.view(batch_size, -1, word_encodings.size(-1))
        attn_mask = self._get_attn_mask(X.view(batch_size, -1))
        decoder_ret_dict = self._decode(
            inputs=Y_in,
            context=ctx_encodings,
            attn_ctx=attn_ctx,
            attn_mask=attn_mask
        )

        ## Loss
        loss = 0
        # Reconstruction
        word_loss = F.cross_entropy(
            decoder_ret_dict["logits"].view(-1, self.vocab_size),
            Y_out.view(-1),
            ignore_index=self.decoder.pad_token_id,
            reduction="mean"
        )
        ppl = torch.exp(word_loss)
        loss += word_loss
        # KLD
        kld_coef = self._kld_coef_term(step)
        kld_losses = gaussian_kld(post_mu, post_var, prior_mu, prior_var)
        avg_kld = kld_losses.mean()
        loss += avg_kld*kld_coef
        # BOW
        pred_bow = self.latent_to_bow(post_z)
        pred_bow = pred_bow.unsqueeze(1).repeat(1, max_y_sent_len, 1)
        bow_loss = F.cross_entropy(
            pred_bow.view(batch_size*max_y_sent_len, -1),
            Y_out.view(-1),
            ignore_index=self.pad_token_id,
            reduction="mean"
        )
        loss += bow_loss

        ## Return statistics
        ret_statistics = {}
        ret_statistics["ppl"] = ppl.item()
        ret_statistics["kld_term"] = kld_coef
        ret_statistics["kld"] = avg_kld.item()
        ret_statistics["bow_loss"] = bow_loss.item()
        ret_statistics["loss"] = loss.item()

        # return dicts
        ret_data = {
            "loss": loss 
        }
        ret_stat = {
            "ppl": ppl.item(),
            "kld_term": kld_coef,
            "kld": avg_kld.item(),
            "bow_loss": bow_loss.item(),
            "loss": loss.item()
        }

        return ret_data, ret_stat

    def evaluate_step(self, data):
        """One training step

        Arguments:
            data {dict of data} -- required keys and values:
                X {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                X_floor {LongTensor [batch_size, history_len]} -- floors of context sentences
                Y {LongTensor [batch_size, max_y_sent_len]} -- token ids of response sentence
                Y_floor {LongTensor [batch_size]} -- floor of response sentence

        Returns:
            dict of data -- returned keys and values

            dict of statistics -- returned keys and values
                ppl {float} -- perplexity
                kld {float} -- KLD
                bow_loss {float} -- bag-of-word loss
                monitor {float} -- a monitor number for learning rate scheduling
        """
        X, Y = data["X"], data["Y"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_in = Y[:, :-1].contiguous()
        Y_out = Y[:, 1:].contiguous()

        batch_size = X.size(0)
        max_y_sent_len = Y_out.size(1)

        with torch.no_grad():
            ## Forward
            # Get prior z
            word_encodings, sent_encodings, dial_encodings = self._encode_prior(
                inputs=X,
                input_floors=X_floor,
                output_floors=Y_floor
            )
            prior_net_input = dial_encodings
            prior_z, prior_mu, prior_var = self.prior_net(prior_net_input)
            # Get post z
            post_sent_encodings = self._encode_post(Y)
            post_net_input = torch.cat([post_sent_encodings, dial_encodings], dim=1)
            post_z, post_mu, post_var = self.post_net(post_net_input)
            # Decode
            ctx_encodings = self.ctx_fc(torch.cat([post_z, dial_encodings], dim=1))
            attn_ctx = word_encodings.view(batch_size, -1, word_encodings.size(-1))
            attn_mask = self._get_attn_mask(X.view(batch_size, -1))
            decoder_ret_dict = self._decode(
                inputs=Y_in,
                context=ctx_encodings,
                attn_ctx=attn_ctx,
                attn_mask=attn_mask
            )

            ## Loss
            # Reconstruction
            word_loss = F.cross_entropy(
                decoder_ret_dict["logits"].view(-1, self.vocab_size),
                Y_out.view(-1),
                ignore_index=self.decoder.pad_token_id,
                reduction="mean"
            )
            ppl = torch.exp(word_loss)
            # KLD
            kld_losses = gaussian_kld(post_mu, post_var, prior_mu, prior_var)
            avg_kld = kld_losses.mean()
            # BOW
            pred_bow = self.latent_to_bow(post_z)
            pred_bow = pred_bow.unsqueeze(1).repeat(1, max_y_sent_len, 1)
            bow_loss = F.cross_entropy(
                pred_bow.view(batch_size*max_y_sent_len, -1),
                Y_out.view(-1),
                ignore_index=self.pad_token_id,
                reduction="mean"
            )

        # return statistics
        ret_statistics = {}
        ret_statistics["ppl"] = ppl.item()
        ret_statistics["kld"] = avg_kld.item()
        ret_statistics["bow_loss"] = bow_loss.item()
        ret_statistics["monitor"] = ppl.item()

        # return dicts
        ret_data = {}
        ret_stat = {
            "ppl": ppl.item(),
            "kld": avg_kld.item(),
            "bow_loss": bow_loss.item(),
            "monitor": ppl.item()
        }

        return ret_data, ret_stat

    def test_step(self, data, **kwargs):
        """One test step

        Arguments:
            data {dict of data} -- required keys and values:
                X {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                X_floor {LongTensor [batch_size, history_len]} -- floors of context sentences
                Y_floor {LongTensor [batch_size]} -- floor of response sentence

        Returns:
            dict of data -- returned keys and values
                symbols {LongTensor [batch_size, max_decode_len]} -- token ids of response hypothesis
            dict of statistics -- returned keys and values

        """
        X = data["X"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        batch_size = X.size(0)

        with torch.no_grad():
            ## Forward
            # Get prior z
            word_encodings, sent_encodings, dial_encodings = self._encode_prior(
                inputs=X,
                input_floors=X_floor,
                output_floors=Y_floor
            )
            prior_net_input = dial_encodings
            prior_z, prior_mu, prior_var = self.prior_net(prior_net_input)
            # Decode
            ctx_encodings = self.ctx_fc(torch.cat([prior_z, dial_encodings], dim=1))
            attn_ctx = word_encodings.view(batch_size, -1, word_encodings.size(-1))
            attn_mask = self._get_attn_mask(X.view(batch_size, -1))
            decoder_ret_dict = self._sample(
                context=ctx_encodings,
                attn_ctx=attn_ctx,
                attn_mask=attn_mask
            )


        ret_data = {
            "symbols": decoder_ret_dict["symbols"]
        }
        ret_stat = {}

        return ret_data, ret_stat