import code

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.joint_da_seg_recog.ed import EDSeqLabeler
from model.modules.encoders import EncoderRNN
from model.modules.decoders import DecoderRNN
from model.modules.submodules import RelFloorEmbEncoder
from model.modules.utils import init_module_weights, init_word_embedding


class AttnEDSeqLabeler(EDSeqLabeler):
    def __init__(self, config, tokenizer, label_tokenizer):
        super(AttnEDSeqLabeler, self).__init__(config, tokenizer, label_tokenizer)

        # Attributes
        # Attributes from config
        self.dial_encoder_hidden_dim = config.dial_encoder_hidden_dim
        self.n_dial_encoder_layers = config.n_dial_encoder_layers
        self.attention_type = config.attention_type

        # Encoding components
        self.dial_encoder = EncoderRNN(
            input_dim=self.sent_encoder_hidden_dim,
            hidden_dim=self.dial_encoder_hidden_dim,
            n_layers=self.n_dial_encoder_layers,
            dropout_emb=self.dropout,
            dropout_input=self.dropout,
            dropout_hidden=self.dropout,
            dropout_output=self.dropout,
            bidirectional=False,
            rnn_type=self.rnn_type,
        )

        # Decoding components
        self.enc2dec_hidden_fc = nn.Linear(
            self.dial_encoder_hidden_dim,
            self.n_decoder_layers*self.decoder_hidden_dim if self.rnn_type == "gru"
            else self.n_decoder_layers*self.decoder_hidden_dim*2
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
            use_attention=True,
            attn_dim=self.sent_encoder_hidden_dim
        )

        self.floor_encoder = RelFloorEmbEncoder(
            input_dim=self.sent_encoder_hidden_dim,
            embedding_dim=self.attr_embedding_dim
        )

    def _encode(self, inputs, input_floors):
        batch_size, history_len, max_sent_len = inputs.size()
        
        input_lens = (inputs != self.pad_token_id).sum(-1)
        dial_lens = (input_lens > 0).long().sum(1)  # equals number of non-padding sents

        flat_inputs = inputs.view(batch_size*history_len, max_sent_len)
        flat_input_lens = input_lens.view(batch_size*history_len)
        word_encodings, _, sent_encodings = self.sent_encoder(flat_inputs, flat_input_lens)
        word_encodings = word_encodings.view(batch_size, history_len, max_sent_len, -1)
        sent_encodings = sent_encodings.view(batch_size, history_len, -1)

        # fetch target-sentence-releated information
        tgt_floors = []
        tgt_word_encodings = []
        for dial_idx, dial_len in enumerate(dial_lens):
            tgt_floors.append(input_floors[dial_idx, dial_len-1])
            tgt_word_encodings.append(word_encodings[dial_idx, dial_len-1, :, :])
        tgt_floors = torch.stack(tgt_floors, 0)
        tgt_word_encodings = torch.stack(tgt_word_encodings, 0)

        src_floors = input_floors.view(-1)
        tgt_floors = tgt_floors.unsqueeze(1).repeat(1, history_len).view(-1)
        sent_encodings = sent_encodings.view(batch_size*history_len, -1)
        sent_encodings = self.floor_encoder(
            sent_encodings,
            src_floors=src_floors,
            tgt_floors=tgt_floors
        )
        sent_encodings = sent_encodings.view(batch_size, history_len, -1)

        _, _, dial_encodings = self.dial_encoder(sent_encodings, dial_lens)  # [batch_size, dialog_encoder_dim]

        return word_encodings, sent_encodings, dial_encodings, tgt_word_encodings

    # def _get_tgt_words(self, inputs, word_encodings):        
    #     input_lens = (inputs != self.pad_token_id).sum(-1)
    #     dial_lens = (input_lens > 0).long().sum(1)  # equals number of non-padding sents

    #     tgt_words = []
    #     tgt_word_encodings = []
    #     for dial_idx, dial_len in enumerate(dial_lens):
    #         tgt_words.append(inputs[dial_idx, dial_len-1, :])
    #         tgt_word_encodings.append(word_encodings[dial_idx, dial_len-1, :, :])
    #     tgt_words = torch.stack(tgt_words, 0)
    #     tgt_word_encodings = torch.stack(tgt_word_encodings, 0)

    #     return tgt_words, tgt_word_encodings

    def _get_attn_mask(self, attn_keys):
        attn_mask = (attn_keys != self.pad_token_id)
        return attn_mask

    def train_step(self, data):
        """One training step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of sentences
                'Y' {LongTensor [batch_size, max_sent_len]} -- label ids of corresponding tokens

        Returns:
            dict of data -- returned keys and values
                'loss' {FloatTensor []} -- loss to backword
            dict of statistics -- returned keys and values
                'loss' {float} -- batch loss
        """
        X, Y = data["X"], data["Y"]
        X_floor = data["X_floor"]
        Y_in = Y[:, :-1].contiguous()
        Y_out = Y[:, 1:].contiguous()

        batch_size = X.size(0)
        max_y_len = Y_out.size(1)

        # Forward
        word_encodings, sent_encodings, dial_encodings, tgt_word_encodings = self._encode(X, X_floor)
        if self.attention_type == "word":
            attn_keys = word_encodings.view(batch_size, -1, word_encodings.size(-1))
            attn_mask = self._get_attn_mask(X).view(batch_size, -1)
        elif self.attention_type == "sent":
            attn_keys = sent_encodings.view(batch_size, -1, sent_encodings.size(-1))
            attn_mask = (X != self.pad_token_id).sum(-1) > 0
        decoder_ret_dict = self._decode(
            dec_inputs=Y_in,
            word_encodings=tgt_word_encodings,
            sent_encodings=dial_encodings,
            attn_ctx=attn_keys,
            attn_mask=attn_mask
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
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of sentences
                'Y' {LongTensor [batch_size, 1, max_sent_len]} -- label ids of corresponding tokens

        Returns:
            dict of data -- returned keys and values

            dict of statistics -- returned keys and values
                'loss' {float} -- batch loss
        """
        X, Y = data["X"], data["Y"]
        X_floor = data["X_floor"]
        Y_in = Y[:, :-1].contiguous()
        Y_out = Y[:, 1:].contiguous()

        batch_size = X.size(0)
        max_y_len = Y_out.size(1)

        with torch.no_grad():
            # Forward
            word_encodings, sent_encodings, dial_encodings, tgt_word_encodings = self._encode(X, X_floor)
            if self.attention_type == "word":
                attn_keys = word_encodings.view(batch_size, -1, word_encodings.size(-1))
                attn_mask = self._get_attn_mask(X).view(batch_size, -1)
            elif self.attention_type == "sent":
                attn_keys = sent_encodings.view(batch_size, -1, sent_encodings.size(-1))
                attn_mask = (X != self.pad_token_id).sum(-1) > 0
            decoder_ret_dict = self._decode(
                dec_inputs=Y_in,
                word_encodings=tgt_word_encodings,
                sent_encodings=dial_encodings,
                attn_ctx=attn_keys,
                attn_mask=attn_mask
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
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of sentences

        Returns:
            dict of data -- returned keys and values
                'symbols' {LongTensor [batch_size, max_decode_len]} -- predicted label ids
            dict of statistics -- returned keys and values
        """
        X = data["X"]
        X_floor = data["X_floor"]
        
        batch_size = X.size(0)

        with torch.no_grad():
            # Forward
            word_encodings, sent_encodings, dial_encodings, tgt_word_encodings = self._encode(X, X_floor)
            if self.attention_type == "word":
                attn_keys = word_encodings.view(batch_size, -1, word_encodings.size(-1))
                attn_mask = self._get_attn_mask(X).view(batch_size, -1)
            elif self.attention_type == "sent":
                attn_keys = sent_encodings.view(batch_size, -1, sent_encodings.size(-1))
                attn_mask = (X != self.pad_token_id).sum(-1) > 0
            decoder_ret_dict = self._sample(
                word_encodings=tgt_word_encodings,
                sent_encodings=dial_encodings,
                attn_ctx=attn_keys,
                attn_mask=attn_mask
            )

        ret_data = {
            "symbols": decoder_ret_dict["symbols"]
        }
        ret_stat = {}

        return ret_data, ret_stat
