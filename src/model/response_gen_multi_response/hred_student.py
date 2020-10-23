import re
import code
import math
import json
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.decomposition import PCA

from model.response_gen_multi_response.hred import HRED
from model.response_gen_multi_response.gpt2 import GPT2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HREDStudent(HRED):
    def __init__(self, config, tokenizer, teacher=None):
        super(HREDStudent, self).__init__(config, tokenizer)

        self.interpolation_alpha = 0.5
        self.use_teacher_word_embedding = config.use_teacher_word_embedding if hasattr(config, "use_teacher_word_embedding") else False
        self.guidance_on = config.guidance_on if hasattr(config, "guidance_on") else "all"

        if self.use_teacher_word_embedding:
            if teacher is not None:
                self.extract_pretrained_word_embeddings(teacher)

    def extract_pretrained_word_embeddings(self, teacher):
        teacher_embedding_weights = teacher.word_embedding.weight.data.cpu().numpy()
        if self.word_embedding_dim == teacher_embedding_weights.shape[1]:
            self.word_embedding.weight = nn.Parameter(torch.FloatTensor(teacher_embedding_weights).to(DEVICE))
            print(f"Using teacher's embedding weights.")
        else:
            print(f"Student's embedding dim {self.word_embedding_dim} is smaller than teacher's {teacher_embedding_weights.shape[1]}.")
            print(f"Using PCA results as word embedding's weights.")
            pca = PCA(n_components=self.word_embedding_dim)
            PC_weights = pca.fit_transform(teacher_embedding_weights)
            self.word_embedding.weight = nn.Parameter(torch.FloatTensor(PC_weights).to(DEVICE))
        if self.tie_weights:
            self.decoder.word_classifier.weight = self.word_embedding.weight

    def train_step(self, data, teacher):
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
        max_y_len = Y_in.size(1)

        # Forward
        # -- postive
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
        logits = decoder_ret_dict["logits"]  # [batch_size, max_y_len, vocab_size]

        with torch.no_grad():
            assert isinstance(teacher, GPT2)
            # construct inputs and outputs
            data_dict = teacher._construct_input_output(
                inputs=X,
                input_floors=X_floor,
                outputs=Y,
                output_floors=Y_floor,
            )
            input_embeddings = data_dict["embeddings"]
            output_labels = data_dict["output_ids"]

            # forward
            teacher_logits, _ = teacher._forward(input_embeddings)

            # collect corresponding logits
            output_labels = output_labels.tolist()
            _teacher_logits = []
            for batch_idx in range(batch_size):
                non_padding_from = 0
                for step_idx in range(teacher_logits.size(1)):
                    if output_labels[batch_idx][step_idx] != teacher.pad_token_id:
                        non_padding_from = step_idx
                        break
                non_padding_to = non_padding_from + max_y_len

                valid_logits = teacher_logits[batch_idx][non_padding_from:non_padding_to]  # [gpt2_len, vocab_size]
                n_right_paddings = max_y_len - valid_logits.size(0)
                if n_right_paddings > 0:
                    right_padding_logits = torch.zeros(n_right_paddings, valid_logits.size(1)).to(DEVICE)
                    valid_logits = torch.cat([valid_logits, right_padding_logits], 0)  # [max_y_len, vocab_size]

                _teacher_logits.append(valid_logits)

            teacher_logits = torch.stack(_teacher_logits, dim=0)  # [batch_size, max_y_len, vocab_size]
            teacher_probs = teacher_logits.softmax(dim=2)
            padding_mask = Y_out == self.pad_token_id  # [batch_size, max_y_len]
            teacher_probs.masked_fill_(padding_mask.unsqueeze(2), 0.0)

        assert teacher_logits.size() == logits.size()
        # code.interact(local=locals())

        # Calculate loss
        loss = 0
        nll = (-1) * logits.log_softmax(dim=2)  # [batch_size, max_y_len, vocab_size]
        teacher_losses = nll * teacher_probs  # [batch_size, max_y_len, vocab_size]
        if self.guidance_on == "all":
            teacher_loss = teacher_losses.sum(2).sum(1).mean(0)
        elif self.guidance_on == "target":
            teacher_losses = teacher_losses.gather(dim=2, index=Y_out.unsqueeze(2)).squeeze(2)  # [batch_size, max_y_len]
            teacher_loss = teacher_losses.sum(1).mean(0)

        if self.interpolation_alpha > 0.0:
            lm_loss = F.cross_entropy(
                decoder_ret_dict["logits"].view(-1, self.vocab_size),
                Y_out.view(-1),
                ignore_index=self.decoder.pad_token_id,
                reduction="sum"
            )/batch_size
        else:
            lm_loss = 0

        loss = self.interpolation_alpha * lm_loss + (1-self.interpolation_alpha) * teacher_loss

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
            "loss": loss.item(),
            "teacher_loss": teacher_loss.item(),
        }
        if self.interpolation_alpha > 0.0:
            ret_stat["lm_loss"] = lm_loss.item()

        return ret_data, ret_stat
