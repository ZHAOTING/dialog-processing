import re
import code
import math
import json
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.modules.decoders import DecoderRNN
from model.modules.utils import init_module_weights, init_word_embedding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNNLM(nn.Module):
    def __init__(self, config, tokenizer):
        super(RNNLM, self).__init__()

        # Attributes
        # Attributes from config
        self.word_embedding_dim = config.word_embedding_dim
        self.decoder_hidden_dim = config.decoder_hidden_dim
        self.n_decoder_layers = config.n_decoder_layers
        self.decode_max_len = config.decode_max_len
        self.tie_weights = config.tie_weights
        self.rnn_type = config.rnn_type
        # Optional attributes from config
        self.gen_type = config.gen_type if hasattr(config, "gen_type") else "greedy"
        self.top_k = config.top_k if hasattr(config, "top_k") else 0
        self.top_p = config.top_p if hasattr(config, "top_p") else 0.0
        self.temp = config.temp if hasattr(config, "temp") else 1.0
        self.dropout_emb = config.dropout if hasattr(config, "dropout") else 0.0
        self.dropout_input = config.dropout if hasattr(config, "dropout") else 0.0
        self.dropout_hidden = config.dropout if hasattr(config, "dropout") else 0.0
        self.dropout_output = config.dropout if hasattr(config, "dropout") else 0.0
        self.use_pretrained_word_embedding = config.use_pretrained_word_embedding if hasattr(config, "use_pretrained_word_embedding") else False
        self.word_embedding_path = config.word_embedding_path if hasattr(config, "word_embedding_path") else None
        # Other attributes
        self.tokenizer = tokenizer
        self.word2id = tokenizer.word2id
        self.id2word = tokenizer.id2word
        self.vocab_size = len(tokenizer.word2id)
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # Components
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
        self.decoder = DecoderRNN(
            vocab_size=len(self.word2id),
            input_dim=self.word_embedding_dim,
            hidden_dim=self.decoder_hidden_dim,
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
        )

        # Initialization
        self._init_weights()

    def _init_weights(self):
        pass

    def _random_hiddens(self, batch_size):
        if self.rnn_type == "gru":
            hiddens = torch.zeros(
                batch_size,
                self.n_decoder_layers,
                self.decoder_hidden_dim
            ).to(DEVICE).transpose(0, 1).contiguous()  # (n_layers, batch_size, hidden_dim)
            nn.init.uniform_(hiddens, -1.0, 1.0)
        elif self.rnn_type == "lstm":
            hiddens = hiddens.view(
                batch_size,
                self.n_decoder_layers,
                self.decoder_hidden_dim,
                2
            ).to(DEVICE)
            nn.init.uniform_(hiddens, -1.0, 1.0)
            h = hiddens[:, :, :, 0].transpose(0, 1).contiguous()  # (n_layers, batch_size, hidden_dim)
            c = hiddens[:, :, :, 1].transpose(0, 1).contiguous()  # (n_layers, batch_size, hidden_dim)
            hiddens = (h, c)

        return hiddens

    def _decode(self, inputs):
        batch_size = inputs.size(0)
        ret_dict = self.decoder.forward(
            batch_size=batch_size,
            inputs=inputs,
            mode=DecoderRNN.MODE_TEACHER_FORCE
        )

        return ret_dict

    def _sample(self, batch_size, hiddens=None):
        ret_dict = self.decoder.forward(
            batch_size=batch_size,
            hiddens=hiddens,
            mode=DecoderRNN.MODE_FREE_RUN,
            gen_type=self.gen_type,
            temp=self.temp,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        return ret_dict

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
                'X' {LongTensor [batch_size, max_len]} -- token ids of sentences

        Returns:
            dict of data -- returned keys and values
                'loss' {FloatTensor []} -- loss to backword
            dict of statistics -- returned keys and values
                'ppl' {float} -- perplexity
                'loss' {float} -- batch loss
        """
        X = data["X"]
        X_in = X[:, :-1].contiguous()
        X_out = X[:, 1:].contiguous()

        batch_size = X.size(0)

        # Forward
        decoder_ret_dict = self._decode(
            inputs=X_in
        )

        # Calculate loss
        loss = 0
        word_loss = F.cross_entropy(
            decoder_ret_dict["logits"].view(-1, self.vocab_size),
            X_out.view(-1),
            ignore_index=self.decoder.pad_token_id,
            reduction="mean"
        )
        ppl = torch.exp(word_loss)
        loss = word_loss

        # return dicts
        ret_data = {
            "loss": loss
        }
        ret_stat = {
            "ppl": ppl.item(),
            "loss": loss.item()
        }

        return ret_data, ret_stat

    def evaluate_step(self, data):
        """One evaluation step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, max_len]} -- token ids of sentences

        Returns:
            dict of data -- returned keys and values

            dict of statistics -- returned keys and values
                'ppl' {float} -- perplexity
                'monitor' {float} -- a monitor number for learning rate scheduling
        """
        X = data["X"]
        X_in = X[:, :-1].contiguous()
        X_out = X[:, 1:].contiguous()

        batch_size = X.size(0)

        with torch.no_grad():
            decoder_ret_dict = self._decode(
                inputs=X_in
            )

            # Loss
            word_loss = F.cross_entropy(
                decoder_ret_dict["logits"].view(-1, self.vocab_size),
                X_out.view(-1),
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

    def sample_step(self, batch_size):
        """One test step

        Arguments:
            batch_size {int}

        Returns:
            dict of data -- returned keys and values
                'symbols' {LongTensor [batch_size, max_decode_len]} -- token ids of response hypothesis
            dict of statistics -- returned keys and values

        """
        with torch.no_grad():
            decoder_ret_dict = self._sample(batch_size)

        ret_data = {
            "symbols": decoder_ret_dict["symbols"]
        }
        ret_stat = {}

        return ret_data, ret_stat

    def compute_prob(self, sents):
        """Compute P(sents)

        Arguments:
            sents {List [str]} -- sentences in string form
        """
        batch_tokens = [self.tokenizer.convert_string_to_tokens(sent) for sent in sents]
        batch_token_ids = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in batch_tokens]
        batch_token_ids = self.tokenizer.convert_batch_ids_to_tensor(batch_token_ids).to(DEVICE)  # [batch_size, len]
        X_in = batch_token_ids[:, :-1]
        X_out = batch_token_ids[:, 1:]

        with torch.no_grad():
            word_ll = []
            sent_ll = []
            sent_probs = []

            batch_size = 50
            n_batches = math.ceil(X_in.size(0)/batch_size)

            for batch_idx in range(n_batches):
                begin = batch_idx * batch_size
                end = min(begin + batch_size, X_in.size(0))
                batch_X_in = X_in[begin:end]
                batch_X_out = X_out[begin:end]

                decoder_ret_dict = self._decode(
                    inputs=batch_X_in
                )
                logits = decoder_ret_dict["logits"]  # [batch_size, len-1, vocab_size]
                batch_word_ll = F.log_softmax(logits, dim=2)
                batch_gathered_word_ll = batch_word_ll.gather(2, batch_X_out.unsqueeze(2)).squeeze(2)  # [batch_size, len-1]
                batch_sent_ll = batch_gathered_word_ll.sum(1)  # [batch_size]
                batch_sent_probs = batch_sent_ll.exp()

                word_ll.append(batch_gathered_word_ll)
                sent_ll.append(batch_sent_ll)
                sent_probs.append(batch_sent_probs)

            word_ll = torch.cat(word_ll, dim=0)
            sent_ll = torch.cat(sent_ll, dim=0)
            sent_probs = torch.cat(sent_probs, dim=0)

        ret_data = {
            "word_loglikelihood": word_ll,
            "sent_loglikelihood": sent_ll,
            "sent_likelihood": sent_probs
        }

        return ret_data




