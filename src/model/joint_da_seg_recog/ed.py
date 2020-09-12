import code

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.modules.encoders import EncoderRNN
from model.modules.decoders import DecoderRNN
from model.modules.utils import init_module_weights, init_word_embedding


class EDSeqLabeler(nn.Module):
    def __init__(self, config, tokenizer, label_tokenizer):
        super(EDSeqLabeler, self).__init__()
        
        # Attributes
        # Attributes from config
        self.num_labels = len(label_tokenizer)
        self.word_embedding_dim = config.word_embedding_dim
        self.attr_embedding_dim = config.attr_embedding_dim
        self.sent_encoder_hidden_dim = config.sent_encoder_hidden_dim
        self.n_sent_encoder_layers = config.n_sent_encoder_layers
        self.decoder_hidden_dim = config.decoder_hidden_dim
        self.n_decoder_layers = config.n_decoder_layers
        self.decode_max_len = config.decode_max_len
        self.tie_weights = config.tie_weights
        self.rnn_type = config.rnn_type
        self.gen_type = config.gen_type
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.temp = config.temp
        self.word_embedding_path = config.word_embedding_path
        # Optional attributes from config
        self.dropout = config.dropout if hasattr(config, "dropout") else 0.0
        self.use_pretrained_word_embedding = config.use_pretrained_word_embedding if hasattr(config, "use_pretrained_word_embedding") else False
        # Other attributes
        self.word2id = tokenizer.word2id
        self.id2word = tokenizer.id2word
        self.label2id = label_tokenizer.word2id
        self.id2label = label_tokenizer.id2word
        self.vocab_size = len(tokenizer)
        self.label_vocab_size = len(label_tokenizer)
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_label_id = label_tokenizer.pad_token_id
        self.bos_label_id = label_tokenizer.bos_token_id
        self.eos_label_id = label_tokenizer.eos_token_id

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
            dropout_emb=self.dropout,
            dropout_input=self.dropout,
            dropout_hidden=self.dropout,
            dropout_output=self.dropout,
            bidirectional=True,
            embedding=self.word_embedding,
            rnn_type=self.rnn_type,
        )

        # Decoding components
        self.enc2dec_hidden_fc = nn.Linear(
            self.sent_encoder_hidden_dim,
            self.n_decoder_layers*self.decoder_hidden_dim if self.rnn_type == "gru"
            else self.n_decoder_layers*self.decoder_hidden_dim*2
        )
        self.label_embedding = nn.Embedding(
            self.label_vocab_size,
            self.attr_embedding_dim,
            padding_idx=self.pad_label_id,
            _weight=init_word_embedding(
                load_pretrained_word_embedding=False,
                id2word=self.id2label,
                word_embedding_dim=self.attr_embedding_dim,
                vocab_size=self.label_vocab_size,
                pad_token_id=self.pad_label_id
            ),
        )
        self.decoder = DecoderRNN(
            vocab_size=self.label_vocab_size,
            input_dim=self.attr_embedding_dim,
            hidden_dim=self.decoder_hidden_dim,
            feat_dim=self.sent_encoder_hidden_dim,
            n_layers=self.n_decoder_layers,
            bos_token_id=self.bos_label_id,
            eos_token_id=self.eos_label_id,
            pad_token_id=self.pad_label_id,
            max_len=self.decode_max_len,
            dropout_emb=self.dropout,
            dropout_input=self.dropout,
            dropout_hidden=self.dropout,
            dropout_output=self.dropout,
            embedding=self.label_embedding,
            tie_weights=self.tie_weights,
            rnn_type=self.rnn_type,
            use_attention=False
        )

        # Initialization
        self._init_weights()

    def _init_weights(self):
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

    def _encode(self, inputs):
        batch_size, history_len, max_sent_len = inputs.size()
        assert history_len == 1

        inputs = inputs.view(batch_size, max_sent_len)
        input_lens = (inputs != self.pad_token_id).sum(-1)

        word_encodings, _, sent_encodings = self.sent_encoder(inputs, input_lens)
        word_encodings = word_encodings.view(batch_size, max_sent_len, -1)
        sent_encodings = sent_encodings.view(batch_size, -1)

        return word_encodings, sent_encodings

    def _decode(self, dec_inputs, word_encodings, sent_encodings, attn_ctx=None, attn_mask=None):
        batch_size = sent_encodings.size(0)
        hiddens = self._init_dec_hiddens(sent_encodings)
        feats = word_encodings[:, 1:, :].contiguous()  # excluding <s>
        ret_dict = self.decoder.forward(
            batch_size=batch_size,
            inputs=dec_inputs,
            hiddens=hiddens,
            feats=feats,
            attn_ctx=attn_ctx,
            attn_mask=attn_mask,
            mode=DecoderRNN.MODE_TEACHER_FORCE
        )

        return ret_dict

    def _sample(self, word_encodings, sent_encodings, attn_ctx=None, attn_mask=None):
        batch_size = sent_encodings.size(0)
        hiddens = self._init_dec_hiddens(sent_encodings)
        feats = word_encodings[:, 1:, :].contiguous()  # excluding <s>
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
                'X' {LongTensor [batch_size, 1, max_sent_len]} -- token ids of sentences
                'Y' {LongTensor [batch_size, max_sent_len]} -- label ids of corresponding tokens

        Returns:
            dict of data -- returned keys and values
                'loss' {FloatTensor []} -- loss to backword
            dict of statistics -- returned keys and values
                'loss' {float} -- batch loss
        """
        X, Y = data["X"], data["Y"]
        Y_in = Y[:, :-1].contiguous()
        Y_out = Y[:, 1:].contiguous()

        batch_size = X.size(0)
        max_y_len = Y_out.size(1)

        # Forward
        word_encodings, sent_encodings = self._encode(X)
        decoder_ret_dict = self._decode(
            dec_inputs=Y_in,
            word_encodings=word_encodings,
            sent_encodings=sent_encodings
        )

        # Calculate loss
        loss = 0
        logits = decoder_ret_dict["logits"]
        label_losses = F.cross_entropy(
            logits.view(-1, self.label_vocab_size),
            Y_out.view(-1),
            ignore_index=self.pad_label_id,
            reduction="none"
        ).view(batch_size, max_y_len)
        sent_loss = label_losses.sum(1).mean(0)
        loss += sent_loss
        
        # return dicts
        ret_data = {
            "loss": loss
        }
        ret_stat = {
            "loss": loss.item()
        }

        return ret_data, ret_stat

    def evaluate_step(self, data):
        """One evaluation step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, 1, max_sent_len]} -- token ids of sentences
                'Y' {LongTensor [batch_size, 1, max_sent_len]} -- label ids of corresponding tokens

        Returns:
            dict of data -- returned keys and values

            dict of statistics -- returned keys and values
                'loss' {float} -- batch loss
        """
        X, Y = data["X"], data["Y"]
        Y_in = Y[:, :-1].contiguous()
        Y_out = Y[:, 1:].contiguous()

        batch_size = X.size(0)
        max_y_len = Y_out.size(1)

        with torch.no_grad():
            # Forward
            word_encodings, sent_encodings = self._encode(X)
            decoder_ret_dict = self._decode(
                dec_inputs=Y_in,
                word_encodings=word_encodings,
                sent_encodings=sent_encodings
            )

            # Calculate loss
            loss = 0
            logits = decoder_ret_dict["logits"]
            label_losses = F.cross_entropy(
                logits.view(-1, self.label_vocab_size),
                Y_out.view(-1),
                ignore_index=self.pad_label_id,
                reduction="none"
            ).view(batch_size, max_y_len)
            sent_loss = label_losses.sum(1).mean(0)
            loss += sent_loss
        
        # return dicts
        ret_data = {}
        ret_stat = {
            "monitor": loss.item(),
            "loss": loss.item()
        }

        return ret_data, ret_stat

    def test_step(self, data):
        """One test step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, 1, max_sent_len]} -- token ids of sentences

        Returns:
            dict of data -- returned keys and values
                'symbols' {LongTensor [batch_size, max_decode_len]} -- predicted label ids
            dict of statistics -- returned keys and values
        """
        X = data["X"]

        with torch.no_grad():
            # Forward
            word_encodings, sent_encodings = self._encode(X)
            decoder_ret_dict = self._sample(
                word_encodings=word_encodings,
                sent_encodings=sent_encodings
            )

        ret_data = {
            "symbols": decoder_ret_dict["symbols"]
        }
        ret_stat = {}

        return ret_data, ret_stat
