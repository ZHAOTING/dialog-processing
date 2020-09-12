import code

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.submodules import Attention, LockedDropout
from model.modules.utils import init_module_weights, init_rnn_hidden_states, embedded_dropout

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def backtrack_beam_result(batch_size, beam_size, scores, predecessors, symbols, eos_token_id):
    scores = np.array(scores)
    predecessors = np.array(predecessors)
    symbols = np.array(symbols)

    L, N = scores.shape
    assert scores.shape == predecessors.shape == symbols.shape
    assert N == batch_size*beam_size

    def backtrack_from_coordinate(i, j):
        """
        arguments
            i - step axis
            j - batch*beam axis
        """
        score = scores[i, j]
        seq = [symbols[i, j]]
        while i > 0:
            j = predecessors[i, j]
            i = i-1
            seq.append(symbols[i, j])
        seq.reverse()
        return seq, score

    batch_seqs = [[] for _ in range(batch_size)]

    # first find early-stopping sequences
    for i in range(L-1):
        for j in range(N):
            if symbols[i, j] == eos_token_id:
                seq, score = backtrack_from_coordinate(i, j)
                batch_idx = j // beam_size
                batch_seqs[batch_idx].append((seq, score))

    # then find full-length sequences
    i = L-1
    for j in range(N):
        seq, score = backtrack_from_coordinate(i, j)
        batch_idx = j // beam_size
        batch_seqs[batch_idx].append((seq, score))

    batch_seqs = [sorted(seqs, key=lambda x: x[1], reverse=True) for seqs in batch_seqs]

    return batch_seqs


class DecoderRNN(nn.Module):
    MODE_TEACHER_FORCE = "teacher forcing"
    MODE_FREE_RUN = "free running"
    GEN_GREEDY = "greedy"
    GEN_BEAM = "beam"
    GEN_SAMPLE = "sample"
    GEN_TOP = "top"
    GEN_BEAM_SAMPLE = "beam_sample"
    GEN_BEAM_MMI_ANTI_LM = "mmi_anti_lm"

    def __init__(self, vocab_size, input_dim, hidden_dim, n_layers,
                 bos_token_id, eos_token_id, pad_token_id, max_len,
                 dropout_emb=0.0, dropout_input=0.0, dropout_hidden=0.0, dropout_output=0.0,
                 use_attention=False, attn_dim=0, feat_dim=0,
                 embedding=None, tie_weights=False, rnn_type="gru"):
        super(DecoderRNN, self).__init__()

        # attributes
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.max_len = max_len
        self.dropout_emb = dropout_emb
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output
        self.use_attention = use_attention
        self.attn_dim = attn_dim
        self.feat_dim = feat_dim
        self.tie_weights = tie_weights
        self.rnn_type = rnn_type

        # input components
        if embedding is None:
            self.embedding = nn.Embedding(
                vocab_size,
                input_dim,
                padding_idx=pad_token_id
            )
        else:
            self.embedding = embedding
        if feat_dim > 0:
            self.feat_fc = nn.Linear(
                input_dim+feat_dim,
                input_dim
            )

        # rnn components
        if rnn_type == "gru":
            self.rnn_cell = nn.GRU
        elif rnn_type == "lstm":
            self.rnn_cell = nn.LSTM
        self.rnn = self.rnn_cell(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            # dropout=dropout_hidden,
        )

        # output components
        if use_attention:
            self.attention = Attention(
                query_dim=hidden_dim,
                context_dim=attn_dim,
                hidden_dim=hidden_dim
            )
        self.project_fc = nn.Linear(
            hidden_dim,
            input_dim
        )
        self.word_classifier = nn.Linear(
            input_dim,
            vocab_size
        )
        if tie_weights:
            self.word_classifier.weight = self.embedding.weight

        # variational dropout
        self.lockdrop = LockedDropout()

        # initialization
        self._init_weights()

    def _init_weights(self):
        if self.feat_dim > 0:
            init_module_weights(self.feat_fc, 0.1)
        init_module_weights(self.rnn)
        init_module_weights(self.project_fc, 0.1)
        if not self.tie_weights:
            init_module_weights(self.word_classifier, 0.1)

    def _step(self, inputs, hiddens, feats=None, attn_ctx=None, attn_mask=None):

        # dropout between words and embedding matrix
        # get word embeddings
        embedded = embedded_dropout(
            embed=self.embedding,
            words=inputs,
            dropout=self.dropout_emb if self.training else 0
        )

        # get rnn inputs
        rnn_inputs = embedded
        if self.feat_dim > 0:
            rnn_inputs = torch.cat([rnn_inputs, feats], dim=2)
            rnn_inputs = self.feat_fc(rnn_inputs)

        # dropout between rnn inputs and rnn
        rnn_inputs = self.lockdrop(rnn_inputs, self.dropout_input)

        # get rnn outputs
        outputs, new_hiddens = self.rnn(rnn_inputs, hiddens)

        # dropout between rnn outputs and subsequent layers
        outputs = self.lockdrop(outputs, self.dropout_output)

        # compute attentions
        attns = None
        if self.use_attention:
            outputs, attns = self.attention(
                query=outputs,
                context=attn_ctx,
                mask=attn_mask
            )

        # get logits
        projected = self.project_fc(outputs)
        logits = self.word_classifier(projected)

        return {
            "outputs": outputs,
            "logits": logits,
            "hiddens": new_hiddens,
            "attns": attns,
        }

    def _step_decode(self, logits, gen_type, top_k=0, top_p=0.0, temp=1.0):
        logits = logits/temp

        scores = logits
        if gen_type == DecoderRNN.GEN_GREEDY:
            symbols = logits.topk(1)[1]
        elif gen_type in (DecoderRNN.GEN_BEAM, DecoderRNN.GEN_BEAM_MMI_ANTI_LM):
            logsoftmax_scores = F.log_softmax(logits, dim=1)
            scores, symbols = logsoftmax_scores.topk(top_k)  # [batch_size, top_k], [batch_size, top_k]
        elif gen_type == DecoderRNN.GEN_SAMPLE:
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            symbols = dist.sample().unsqueeze(1)
        elif gen_type == DecoderRNN.GEN_TOP:
            top_k = min(top_k, logits.size(-1))
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float("inf")
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep the first token above the threshold
                sorted_indices_to_remove[..., 0] = 0

                for batch_idx in range(logits.size(0)):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    logits[batch_idx, indices_to_remove] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            symbols = dist.sample().unsqueeze(1)
        elif gen_type == DecoderRNN.GEN_BEAM_SAMPLE:
            softmax_scores = F.softmax(logits, dim=1)
            symbols = torch.multinomial(softmax_scores, top_k)
            scores = torch.gather(softmax_scores, 1, symbols).log()
        else:
            raise Exception("unsupported generation type {}".format(gen_type))

        return {
            "scores": scores,
            "symbols": symbols,
        }

    def forward(self, batch_size, inputs=None, hiddens=None, feats=None,
                attn_ctx=None, attn_mask=None,
                mode="teacher forcing", gen_type="greedy",
                top_k=1, top_p=1.0, temp=1.0,
                mmi_args={"lm": None, "lambda": 0, "gamma": 0, "tokenizer": None}):

        ret_dict = {
            "outputs": None,
            "logits": None,
            "hiddens": None,
            "attn": None
        }

        # unrolling over steps
        # use automatic unrolling in standard teacher forcing
        if mode == DecoderRNN.MODE_TEACHER_FORCE:
            step_ret_dict = self._step(
                inputs=inputs,
                hiddens=hiddens,
                feats=feats,
                attn_ctx=attn_ctx,
                attn_mask=attn_mask
            )
            ret_dict["outputs"] = step_ret_dict["outputs"]
            ret_dict["logits"] = step_ret_dict["logits"]
            ret_dict["hiddens"] = step_ret_dict["hiddens"]
            ret_dict["attns"] = step_ret_dict["attns"]
        # manual unrolling in free running mode
        elif mode == DecoderRNN.MODE_FREE_RUN:
            if feats is None:
                n_unrolling_steps = self.max_len
            else:
                n_unrolling_steps = feats.size(1)

            # bos input for the first step
            bos_input = torch.LongTensor([self.bos_token_id]).to(DEVICE)
            bos_input.requires_grad_(False)
            step_input = bos_input.expand(batch_size, 1)

            # hidden for the first step
            step_hidden = hiddens

            def beamize_data(data, k, batch_dim=0):
                if batch_dim != 0:
                    data = data.transpose(0, batch_dim).contiguous()

                batch_size = data.size()[0]
                data_size = data.size()[1:]
                num_dims = data.dim()

                data = data.unsqueeze(1) \
                    .repeat((1,) + (k,) + (1,)*(num_dims-1)) \
                    .view((batch_size*k,) + data_size)

                if batch_dim != 0:
                    data = data.transpose(0, batch_dim).contiguous()
                return data

            if gen_type in (DecoderRNN.GEN_BEAM, DecoderRNN.GEN_BEAM_SAMPLE, DecoderRNN.GEN_BEAM_MMI_ANTI_LM):
                # Replicate inputs for each beam.
                step_input = beamize_data(step_input, top_k)
                step_hidden = beamize_data(step_hidden, top_k, batch_dim=1)
                attn_ctx = beamize_data(attn_ctx, top_k)
                attn_mask = beamize_data(attn_mask, top_k)
                feats = None if feats is None else beamize_data(feats, top_k)

                # Initialize the scores.
                #  For each beam, put only the first entry's score as 0.0 and ignore other entries.
                cur_scores = torch.full((batch_size, top_k), -float("inf")).to(DEVICE)  # [batch_size, top_k]
                cur_scores[:, 0] = 0.0
                # Initialize decisions backtracking.
                beam_scores = []
                beam_predecessors = []
                beam_emmited_symbols = []
                # partial sequences
                partial_seqs = step_input.tolist()
            else:
                # lists for collecting step products
                step_output_lst = []
                step_logit_lst = []
                step_attn_lst = []
                step_symbol_lst = []

            # unrolling
            for step_idx in range(n_unrolling_steps):
                # some inputs to the current step
                if feats is None:
                    step_feat = None
                else:
                    step_feat = feats[:, step_idx, :].unsqueeze(1)
                # take a step
                step_ret_dict = self._step(
                    inputs=step_input,
                    hiddens=step_hidden,
                    feats=step_feat,
                    attn_ctx=attn_ctx,
                    attn_mask=attn_mask
                )
                # get inputs to next step
                decode_dict = self._step_decode(
                    logits=step_ret_dict["logits"].squeeze(1),
                    gen_type=gen_type,
                    top_k=top_k,
                    top_p=top_p,
                    temp=temp
                )

                if gen_type not in (DecoderRNN.GEN_BEAM, DecoderRNN.GEN_BEAM_SAMPLE, DecoderRNN.GEN_BEAM_MMI_ANTI_LM):
                    step_input = decode_dict["symbols"]
                    step_hidden = step_ret_dict["hiddens"]

                    # collect step productions
                    step_output_lst.append(step_ret_dict["outputs"])
                    step_logit_lst.append(step_ret_dict["logits"])
                    step_attn_lst.append(step_ret_dict["attns"])
                    step_symbol_lst.append(step_input)
                else:
                    step_scores = decode_dict["scores"]  # [batch_size*top_k, top_k]
                    step_symbols = decode_dict["symbols"]  # [batch_size*top_k, top_k]

                    # get topk outputs from candidates and update total scores
                    cur_scores = cur_scores.view(batch_size*top_k, 1)

                    if gen_type == DecoderRNN.GEN_BEAM_MMI_ANTI_LM:
                        # code.interact(local=locals())
                        step_symbol_lst = step_symbols.tolist()
                        new_partial_seqs = []
                        for partial_seq_idx in range(len(partial_seqs)):
                            partial_seq = partial_seqs[partial_seq_idx]
                            for symbol in step_symbol_lst[partial_seq_idx]:
                                new_partial_seqs.append(partial_seq + [symbol])
                        lm_seqs = [mmi_args["tokenizer"].convert_ids_to_tokens(ids) for ids in new_partial_seqs]
                        lm_seqs = [mmi_args["tokenizer"].convert_tokens_to_string(tokens) for tokens in lm_seqs]
                        lm_outputs = mmi_args["lm"].compute_prob(lm_seqs)  # [batch_size*top_k*top_k]
                        lm_word_ll = lm_outputs["word_loglikelihood"]
                        length_mask = torch.LongTensor(list(range(lm_word_ll.size(1)))).to(DEVICE).unsqueeze(0)
                        length_mask = ((length_mask + 1) <= mmi_args["gamma"])
                        length_penalty = length_mask.float().log()
                        masked_lm_word_ll = lm_word_ll + length_penalty
                        masked_lm_sent_ll = masked_lm_word_ll.sum(1)
                        U_t = masked_lm_sent_ll.exp().view(batch_size*top_k, top_k)
                        P_t_given_s = cur_scores.exp()

                        mmi_scores = P_t_given_s - mmi_args["lambda"]*U_t + mmi_args["gamma"]*(step_idx+1)

                        new_score_candidates = mmi_scores.view(batch_size, top_k*top_k)
                        _, cand_idcs = new_score_candidates.topk(top_k, dim=1)  # [batch_size, top_k]
                        new_cur_scores = []
                        step_scores_flat = step_scores.view(batch_size, top_k*top_k)
                        for batch_idx in range(len(cand_idcs)):
                            for cand_idx in cand_idcs[batch_idx]:
                                pred = cand_idx//top_k + batch_idx*top_k
                                new_cur_scores.append(cur_scores[pred] + step_scores_flat[batch_idx][cand_idx])
                        cur_scores = torch.FloatTensor(new_cur_scores).to(DEVICE)  # [batch_size*top_k]

                    else:
                        # -- length penalty
                        new_score_candidates = (cur_scores*(step_idx+1) + step_scores)/(step_idx+2)  # [batch_size*top_k, top_k]
                        new_score_candidates = new_score_candidates.view(batch_size, top_k*top_k)
                        cur_scores, cand_idcs = new_score_candidates.topk(top_k, dim=1)  # both [batch_size, top_k]
                        cur_scores = cur_scores.view(batch_size*top_k)

                    # according to outputs' indices in candidates, find predecessors and to-emit symbols
                    cand_idcs = cand_idcs.tolist()
                    to_emit_symbol_candidates = step_symbols.view(batch_size, top_k*top_k).tolist()
                    step_predecessors = []
                    step_emitted_symbols = []
                    for batch_idx in range(len(cand_idcs)):
                        for cand_idx in cand_idcs[batch_idx]:
                            pred = cand_idx//top_k + batch_idx*top_k
                            emit = to_emit_symbol_candidates[batch_idx][cand_idx]
                            step_predecessors.append(pred)
                            step_emitted_symbols.append(emit)
                    beam_emmited_symbols.append(step_emitted_symbols)
                    beam_predecessors.append(step_predecessors)
                    beam_scores.append(cur_scores.tolist())

                    # complete partial sequences
                    new_partial_seqs = []
                    for step_e, step_pred in zip(step_emitted_symbols, step_predecessors):
                        pred_partial_seq = partial_seqs[step_pred]
                        new_partial_seq = pred_partial_seq + [step_e]
                        new_partial_seqs.append(new_partial_seq)
                    partial_seqs = new_partial_seqs

                    # put early-stopping sentences' scores as negative infinite to avoid continuning generation in the next step
                    eos_token_masks = (torch.LongTensor(step_emitted_symbols).to(DEVICE) == self.eos_token_id)
                    cur_scores = cur_scores.masked_fill(eos_token_masks, -float("inf"))

                    # construct new token ids
                    step_emitted_symbols = torch.LongTensor(step_emitted_symbols).view(batch_size*top_k, 1).to(DEVICE)
                    step_input = step_emitted_symbols  # [batch_size*top_k, 1]

                    # construct new hiddens
                    #  find past hidden states (similar to predecessors) given candidate indices
                    step_hidden = step_ret_dict["hiddens"]
                    step_hidden = step_hidden.transpose(0, 1).contiguous()
                    new_step_hidden = []
                    for batch_idx in range(len(cand_idcs)):
                        for cand_idx in cand_idcs[batch_idx]:
                            pred = cand_idx//top_k + batch_idx*top_k
                            new_step_hidden.append(step_hidden[pred])
                    new_step_hidden = torch.stack(new_step_hidden)
                    step_hidden = new_step_hidden
                    step_hidden = step_hidden.transpose(0, 1).contiguous()

            # organize step products
            if gen_type not in (DecoderRNN.GEN_BEAM, DecoderRNN.GEN_BEAM_SAMPLE, DecoderRNN.GEN_BEAM_MMI_ANTI_LM):
                outputs = torch.cat(step_output_lst, dim=1)
                logits = torch.cat(step_logit_lst, dim=1)
                if self.use_attention:
                    attns = torch.cat(step_attn_lst, dim=1)
                else:
                    attns = None
                symbols = torch.cat(step_symbol_lst, dim=1)
                ret_dict["outputs"] = outputs
                ret_dict["logits"] = logits
                ret_dict["hiddens"] = step_hidden
                ret_dict["attns"] = attns
                ret_dict["symbols"] = symbols
            else:
                batch_seqs = backtrack_beam_result(
                    batch_size=batch_size,
                    beam_size=top_k,
                    scores=beam_scores,
                    predecessors=beam_predecessors,
                    symbols=beam_emmited_symbols,
                    eos_token_id=self.eos_token_id
                )

                batch_best_seqs = [seq_score_pairs[0][0] for seq_score_pairs in batch_seqs]
                seq_lens = [len(seq) for seq in batch_best_seqs]
                max_seq_len = max(seq_lens)
                symbols = [seq + [self.pad_token_id]*(max_seq_len-len(seq)) for seq in batch_best_seqs]
                symbols = torch.LongTensor(symbols).to(DEVICE)  # [batch_size, decode_max_len]
                ret_dict["symbols"] = symbols
                ret_dict["beam_hypotheses"] = batch_seqs

        return ret_dict

    def init_hidden_states(self, batch_size, init_type):
        return init_rnn_hidden_states(
            batch_size=batch_size,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            bidirectional=False,
            rnn_type=self.rnn_type,
            init_type=init_type
        )

    def tie_weights(self):
        self.word_classifier.weight = self.embedding.weight