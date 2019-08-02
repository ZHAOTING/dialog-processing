import re
import code
import math
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from ..modules.utils import init_module_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GPT2(nn.Module):
    GEN_GREEDY = "greedy"
    GEN_BEAM = "beam"
    GEN_SAMPLE = "sample"
    GEN_TOP = "top"

    def __init__(self, config, tokenizer):
        super(GPT2, self).__init__()

        ## Load pretrained gpt2 model
        from pytorch_transformers import GPT2LMHeadModel
        if config.model_size == "small":
            pretrained = GPT2LMHeadModel.from_pretrained('gpt2')
        elif config.model_size == "medium":
            pretrained = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        pretrained.resize_token_embeddings(len(tokenizer))
        self.transformer = pretrained.transformer
        self.lm_head = pretrained.lm_head

        ## Attributes
        # Attributes from config
        self.gradient_clip = config.gradient_clip
        self.decode_max_len = config.decode_max_len
        self.gen_type = config.gen_type
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.temp = config.temp
        # Other attributes
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.embedding_dim = self.transformer.config.n_embd
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id
        self.speaker1_id = tokenizer.speaker1_id
        self.speaker2_id = tokenizer.speaker2_id

        ## Embedding componenets
        self.word_embedding = self.transformer.wte
        self.position_embedding = self.transformer.wpe

        ## Optimizer
        from pytorch_transformers import AdamW
        self.optimizer = AdamW(self.parameters(), lr=0.0)

    def _construct_input_output(self, 
        inputs, input_floors, outputs, output_floors, 
        sample_mode=False):
        """Convert a list of sentences into a single sequence for each dialog
        """

        ## Use lists instead of tensors to speed up
        input_lens = (inputs != self.pad_id).sum(-1)
        dial_lens = (input_lens > 0).sum(dim=1).tolist()
        inputs = inputs.tolist()
        input_lens = input_lens.tolist()
        input_floors = input_floors.tolist()
        output_floors = output_floors.tolist()
        if not sample_mode:
            output_lens = (outputs != self.pad_id).sum(-1)
            outputs = outputs.tolist()
            output_lens = output_lens.tolist()
       
        ## build sequences
        input_token_id_seqs = []
        position_id_seqs = []
        output_token_id_seqs = []
        for dial_idx in range(len(inputs)):
            # merge sentences
            ctx_input_ids = []
            for sent_idx in range(dial_lens[dial_idx]):
                sent_len = input_lens[dial_idx][sent_idx]
                sent_token_ids = inputs[dial_idx][sent_idx][:sent_len]

                src_speaker = input_floors[dial_idx][sent_idx]
                tgt_speaker = output_floors[dial_idx]
                speaker_token_id = self.speaker1_id if src_speaker == tgt_speaker else self.speaker2_id

                ctx_input_ids += ([speaker_token_id] + sent_token_ids)

            # input token ids
            if sample_mode:
                response_input_ids = [self.speaker1_id, self.bos_id]
            else:
                response_len = output_lens[dial_idx]
                response_input_ids = [self.speaker1_id] + outputs[dial_idx][:response_len-1]
            input_token_id_seq = ctx_input_ids+response_input_ids
            input_token_id_seqs.append(input_token_id_seq)

            # output token ids
            if not sample_mode:
                ctx_output_ids = [self.pad_id]*len(ctx_input_ids)
                response_output_ids = [self.pad_id] + outputs[dial_idx][1:response_len]
                output_token_id_seq = ctx_output_ids+response_output_ids
                output_token_id_seqs.append(output_token_id_seq)

            # position ids
            position_id_seq = list(range(len(input_token_id_seq)))
            position_id_seqs.append(position_id_seq)

            # assert len(input_token_id_seq) == len(output_token_id_seq)
            # assert len(input_token_id_seq) == len(position_id_seq)

        seq_lens = [len(seq) for seq in input_token_id_seqs]
        max_seq_len = max(seq_lens)

        ## pad sequences and produce tensors
        input_token_id_seqs = [seq + [self.pad_id]*(max_seq_len-len(seq)) for seq in input_token_id_seqs]
        output_token_id_seqs = [seq + [self.pad_id]*(max_seq_len-len(seq)) for seq in output_token_id_seqs]
        position_id_seqs = [seq + [0]*(max_seq_len-len(seq)) for seq in position_id_seqs]
        input_token_id_seqs = torch.LongTensor(input_token_id_seqs).to(DEVICE)
        output_token_id_seqs = torch.LongTensor(output_token_id_seqs).to(DEVICE)
        position_id_seqs = torch.LongTensor(position_id_seqs).to(DEVICE)

        ## compute embeddings
        word_embeddings = self.word_embedding(input_token_id_seqs)
        position_embeddings = self.position_embedding(position_id_seqs)
        embeddings = word_embeddings + position_embeddings

        return {
            "embeddings": embeddings,
            "input_ids": input_token_id_seqs,
            "output_ids": output_token_id_seqs,
            "position_ids": position_id_seqs,
            "seq_lens": seq_lens
        }

    def _new_step_input_embeddings(self, new_symbol, position_id):
        ## word embeddings
        new_word_emb = self.word_embedding(new_symbol)  # [batch_size, 1, emb_dim]

        ## word position embeddings
        position_id = torch.LongTensor([position_id]).long().to(DEVICE).unsqueeze(1)  # [1, 1]
        new_pos_emb = self.position_embedding(position_id)  # [1, 1, emb_dim]
        new_embeddings = new_word_emb + new_pos_emb # [batch_size, 1, emb_dim]

        return new_embeddings

    def _step_decode(self, logits, gen_type, top_k=0, top_p=0.0, temp=1.0):
        logits = logits/temp

        if gen_type == GPT2.GEN_GREEDY:
            symbol = logits.topk(1)[1]
        elif gen_type == GPT2.GEN_SAMPLE:
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            symbol = dist.sample().unsqueeze(1)
        elif gen_type == GPT2.GEN_TOP:
            top_k = min(top_k, logits.size(-1))
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float("inf")
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # keep the first token above the threshold
                sorted_indices_to_remove[..., 0] = 0

                for batch_idx in range(logits.size(0)):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    logits[batch_idx, indices_to_remove] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            symbol = dist.sample().unsqueeze(1)

        elif gen_type == GPT2.GEN_BEAM:
            raise Exception("unsupported generation type {}".format(gen_type))

        return {
            "symbol": symbol,
        }

    def _forward(self, embeddings, past=None):
        if past is None:
            past = [None] * len(self.transformer.h)
        present = []
        input_shape = embeddings.size()[:-1]
        hidden_states = self.transformer.drop(embeddings)
        for block, layer_past in zip(self.transformer.h, past):
            hidden_states, layer_present = block(hidden_states, layer_past)
            present.append(layer_present)
        hidden_states = self.transformer.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, present

    def _sample(self,
        inputs, input_floors, output_floors,
        gen_type, top_p=0.0, top_k=0, temp=1.0):
        data_dict = self._construct_input_output(
            inputs=inputs, 
            input_floors=input_floors,
            outputs=None, 
            output_floors=output_floors,
            sample_mode=True
        )
        ctx_embeddings = data_dict["embeddings"]
        seq_lens = data_dict["seq_lens"]
        batch_size = ctx_embeddings.size(0)
        max_seq_len = max(seq_lens)
        min_seq_len = min(seq_lens)

        # input of the first step
        input_embeddings = ctx_embeddings[:, :min_seq_len, :].contiguous()
        past = None

        # decode the rest of words
        symbols = [[] for _ in range(batch_size)]
        for step in range(0, self.decode_max_len+max_seq_len-min_seq_len):
            new_word_position_id = step + min_seq_len

            # fowrad
            logits, past = self._forward(input_embeddings, past)

            # decode new words fomr last logits
            last_logits = logits[:, -1, :]
            decode_dict = self._step_decode(
                logits=last_logits,
                gen_type=gen_type,
                top_p=top_p,
                top_k=top_k,
                temp=temp
            )
            step_symbol = decode_dict["symbol"]

            # new embeddings
            input_embeddings = self._new_step_input_embeddings(
                new_symbol=step_symbol,
                position_id=new_word_position_id,
            )

            # when min_seq_len <= new_word_position_id < max_seq_len
            #   replace part of new step_embeddings with ground-truth
            #   update sequences where new_word_position_id < sequence_len
            if new_word_position_id < max_seq_len:
                for batch_idx in range(batch_size):
                    seq_len = seq_lens[batch_idx]
                    if new_word_position_id < seq_len:
                        input_embeddings.data[batch_idx] = ctx_embeddings.data[batch_idx][new_word_position_id]

            # collect output symbols for sequence where new_word_position_id >= sequence_len
            for batch_idx in range(batch_size):
                seq_len = seq_lens[batch_idx]
                if new_word_position_id >= seq_len:
                    symbols[batch_idx].append(step_symbol[batch_idx].item())

        output_lens = [len(seq) for seq in symbols]
        max_output_len = max(output_lens)
        symbols = [seq + [self.pad_id]*(max_output_len-len(seq)) for seq in symbols]
        symbols = torch.LongTensor(symbols).to(DEVICE)

        return symbols

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

    def train_step(self, data, lr):
        """One training step

        Arguments:
            data {dict of data} -- required keys and values:
                X {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                X_floor {LongTensor [batch_size, history_len]} -- floors of context sentences
                Y {LongTensor [batch_size, max_y_sent_len]} -- token ids of response sentence
                Y_floor {LongTensor [batch_size]} -- floor of response sentence

            lr {float} -- learning rate

        Returns:
            dict of statistics -- returned keys and values
                ppl {float} -- perplexity
                loss {float} -- batch loss
        """
        X, Y = data["X"], data["Y"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]

        # construct inputs and outputs
        data_dict = self._construct_input_output(
            inputs=X,
            input_floors=X_floor,
            outputs=Y,
            output_floors=Y_floor,
        )
        input_embeddings = data_dict["embeddings"]
        output_labels = data_dict["output_ids"]

        # forward
        logits, _ = self._forward(input_embeddings)

        # loss
        loss = 0
        word_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            output_labels.view(-1),
            ignore_index=self.pad_id,
            reduction="mean"
        )
        ppl = torch.exp(word_loss)
        loss += word_loss

        # return statistics
        ret_statistics = {}
        ret_statistics["ppl"] = ppl.item()
        ret_statistics["loss"] = loss.item()

        # backward
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)
        self.optimizer.step()

        return ret_statistics

    def evaluate_step(self, data):
        """One evaluation step

        Arguments:
            data {dict of data} -- required keys and values:
                X {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                X_floor {LongTensor [batch_size, history_len]} -- floors of context sentences
                Y {LongTensor [batch_size, max_y_sent_len]} -- token ids of response sentence
                Y_floor {LongTensor [batch_size]} -- floor of response sentence

        Returns:
            dict of statistics -- returned keys and values
                ppl {float} -- perplexity
                monitor {float} -- a monitor number for learning rate scheduling
        """
        X, Y = data["X"], data["Y"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]

        with torch.no_grad():
            # construct inputs and outputs
            data_dict = self._construct_input_output(
                inputs=X,
                input_floors=X_floor,
                outputs=Y,
                output_floors=Y_floor,
            )
            input_embeddings = data_dict["embeddings"]
            output_labels = data_dict["output_ids"]

            # forward
            logits, _ = self._forward(input_embeddings)

            # loss
            word_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                output_labels.view(-1),
                ignore_index=self.pad_id,
                reduction="mean"
            )
            ppl = torch.exp(word_loss)

        # return statistics
        ret_statistics = {}
        ret_statistics["ppl"] = ppl.item()
        ret_statistics["monitor"] = ppl.item()

        return ret_statistics

    def test_step(self, data):
        """One test step

        Arguments:
            data {dict of data} -- required keys and values:
                X {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of context sentences
                X_floor {LongTensor [batch_size, history_len]} -- floors of context sentences
                Y_floor {LongTensor [batch_size]} -- floor of response sentence

        Returns:
            dict of outputs -- returned keys and values
                symbols {LongTensor [batch_size, max_decode_len]} -- token ids of response hypothesis
        """
        X, Y = data["X"], data["Y"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]

        with torch.no_grad():
            symbols = self._sample(
                inputs=X, 
                input_floors=X_floor, 
                output_floors=Y_floor,
                gen_type=self.gen_type,
                top_p=self.top_p,
                top_k=self.top_k,
                temp=self.temp
            )

        return {
            "symbols": symbols
        }
