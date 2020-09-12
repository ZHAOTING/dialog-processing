import code
import json
from collections import Counter

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity as cosine
import sklearn as sk

from .sif_embedding import SIF_embedding, compute_pc, get_weighted_average


class ClassificationMetrics:
    def __init__(self, classes):
        self.classes = classes

    def classification_report(self, true_labels, pred_labels, return_dict=False):
        return sk.metrics.classification_report(true_labels, pred_labels, target_names=self.classes, labels=list(range(0, len(self.classes))), output_dict=return_dict)

    def classification_metrics(self, true_labels, pred_labels):
        precision_macro = sk.metrics.precision_score(true_labels, pred_labels, average="macro")
        precision_micro = sk.metrics.precision_score(true_labels, pred_labels, average="micro")
        precision_weighted = sk.metrics.precision_score(true_labels, pred_labels, average="weighted")
        recall_macro = sk.metrics.recall_score(true_labels, pred_labels, average="macro")
        recall_micro = sk.metrics.recall_score(true_labels, pred_labels, average="micro")
        recall_weighted = sk.metrics.recall_score(true_labels, pred_labels, average="weighted")
        f1_macro = sk.metrics.f1_score(true_labels, pred_labels, average="macro")
        f1_micro = sk.metrics.f1_score(true_labels, pred_labels, average="micro")
        f1_weighted = sk.metrics.f1_score(true_labels, pred_labels, average="weighted")
        accuracy = sk.metrics.accuracy_score(true_labels, pred_labels)

        return {
            "precision_macro": precision_macro,
            "precision_micro": precision_micro,
            "precision_weighted": precision_weighted,
            "recall_macro": recall_macro,
            "recall_micro": recall_micro,
            "recall_weighted": recall_weighted,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "accuracy": accuracy,
        }

class DAMetrics:
    def __init__(self):
        pass

    def instance_metrics(self, ref_labels, hyp_labels):
        segment_records = []
        n_segment_tokens, n_segment_seg_errors, n_segment_joint_errors = 0, 0, 0
        for ref, hyp in zip(ref_labels, hyp_labels):
            n_segment_tokens += 1
            if hyp[0] != ref[0]:
                n_segment_seg_errors += 1
            if hyp != ref:
                n_segment_joint_errors += 1
            if ref.startswith("E"):
                segment_records.append((n_segment_tokens, n_segment_seg_errors, n_segment_joint_errors))
                n_segment_tokens, n_segment_seg_errors, n_segment_joint_errors = 0, 0, 0
        
        n_segments = len(segment_records)
        n_tokens = 0
        n_wrong_seg_segments = 0
        n_wrong_seg_tokens = 0
        n_wrong_joint_segments = 0
        n_wrong_joint_tokens = 0
        for (n_segment_tokens, n_segment_seg_errors, n_segment_joint_errors) in segment_records:
            n_tokens += n_segment_tokens
            if n_segment_seg_errors > 0:
                n_wrong_seg_segments += 1
                n_wrong_seg_tokens += n_segment_tokens
            if n_segment_joint_errors > 0:
                n_wrong_joint_segments += 1
                n_wrong_joint_tokens += n_segment_tokens

        DSER = n_wrong_seg_segments / n_segments
        strict_seg_err = n_wrong_seg_tokens / n_tokens
        DER = n_wrong_joint_segments / n_segments
        strict_joint_err = n_wrong_joint_tokens / n_tokens

        return {
            "DSER": DSER,
            "strict segmentation error": strict_seg_err,
            "DER": DER,
            "strict joint error": strict_joint_err
        }

    def batch_metrics(self, refs, hyps):
        score_lists = {
            "DSER": [],
            "strict segmentation error": [],
            "DER": [],
            "strict joint error": []
        }
        for ref_labels, hyp_labels in zip(refs, hyps):
            instance_metrics = self.instance_metrics(ref_labels, hyp_labels)
            for k, v in instance_metrics.items():
                score_lists[k].append(v)

        flattened_refs = [label for ref in refs for label in ref]
        flattened_hyps = [label for hyp in hyps for label in hyp]
        macro_f1 = sk.metrics.f1_score(flattened_refs, flattened_hyps, average="macro")
        micro_f1 = sk.metrics.f1_score(flattened_refs, flattened_hyps, average="micro")

        return {
            "DSER": np.mean(score_lists["DSER"]),
            "strict segmentation error": np.mean(score_lists["strict segmentation error"]),
            "DER": np.mean(score_lists["DER"]),
            "strict joint error": np.mean(score_lists["strict joint error"]),
            "Macro F1": macro_f1,
            "Micro F1": micro_f1,
        }


class SentenceMetrics:
    def __init__(self, word_embedding_path, tokenizer):
        self.tokenizer = tokenizer
        self.word2id = tokenizer.word2id
        self.id2word = tokenizer.id2word

        if hasattr(tokenizer, "word2prob"):
            self.id2prob = {}
            for word, prob in tokenizer.word2prob.items():
                if word in self.word2id:
                    self.id2prob[self.word2id[word]] = prob
        else:
            self.id2prob = None

        try:
            with open(word_embedding_path) as f:
                self.word2vec = json.load(f)
        except FileNotFoundError:
            raise Exception(f"Pretrained word embeddings are needed at {word_embedding_path} to calculate embedding-based metrics.")

        self.emb_mat = []
        emb_dim = len(list(self.word2vec.values())[0])
        for word_id in range(len(self.word2id)):
            word = self.id2word[word_id]
            if word in self.word2vec:
                self.emb_mat.append(self.word2vec[word])
            else:
                self.emb_mat.append([0.]*emb_dim)
        self.emb_mat = np.array(self.emb_mat)

    def _sent2tokens(self, sent):
        return self.tokenizer.convert_string_to_tokens(sent)

    def _tokens2ids(self, tokens):
        return [self.word2id[token] for token in tokens if token in self.word2id]

    def _tokens2emb(self, tokens):
        embs = [self.word2vec[token] for token in tokens if token in self.word2vec]
        if len(embs) == 0:
            embs = [[0.]*self.emb_mat.shape[1]]
        return embs

    def _ids2probs(self, ids):
        probs = []
        for word_id in ids:
            if word_id in self.id2prob:
                probs.append(self.id2prob[word_id])
            else:
                probs.append(10000)
        return probs

    def _cosine_similarity(self, hyps, refs):
        sims = np.sum(hyps * refs, axis=1) / (np.sqrt((np.sum(hyps * hyps, axis=1) * np.sum(refs * refs, axis=1))) + 1e-10)
        return sims

    def _embedding_metric(self, hyps_emb, refs_emb, method='average'):
        if method == 'average':
            hyps_avg_emb = [np.mean(hyp, axis=0) for hyp in hyps_emb]
            refs_avg_emb = [np.mean(ref, axis=0) for ref in refs_emb]
            sims = self._cosine_similarity(np.array(hyps_avg_emb), np.array(refs_avg_emb))
            return sims.tolist()
        elif method == "multi_ref_average":
            hyps_avg_emb = [np.mean(hyp, axis=0) for hyp in hyps_emb]
            mrefs_avg_emb = [[np.mean(ref, axis=0) for ref in refs] for refs in refs_emb]
            msims = []
            for hyp_avg_emb, mref_avg_emb in zip(hyps_avg_emb, mrefs_avg_emb):
                msim = self._cosine_similarity(np.array([hyp_avg_emb]*len(mref_avg_emb)), np.array(mref_avg_emb))
                msims.append(msim.tolist())
            return msims
        elif method == 'extrema':
            hyps_ext_emb = []
            refs_ext_emb = []
            for hyp, ref in zip(hyps_emb, refs_emb):
                h_max = np.max(hyp, axis=0)
                h_min = np.min(hyp, axis=0)
                h_plus = np.absolute(h_min) <= h_max
                h = h_max * h_plus + h_min * np.logical_not(h_plus)
                hyps_ext_emb.append(h)

                r_max = np.max(ref, axis=0)
                r_min = np.min(ref, axis=0)
                r_plus = np.absolute(r_min) <= r_max
                r = r_max * r_plus + r_min * np.logical_not(r_plus)
                refs_ext_emb.append(r)
            sims = self._cosine_similarity(np.array(hyps_ext_emb), np.array(refs_ext_emb))
            return sims.tolist()
        elif method == "multi_ref_extrema":
            hyps_ext_emb = []
            mrefs_ext_emb = []
            for hyp, mref in zip(hyps_emb, refs_emb):
                h_max = np.max(hyp, axis=0)
                h_min = np.min(hyp, axis=0)
                h_plus = np.absolute(h_min) <= h_max
                h = h_max * h_plus + h_min * np.logical_not(h_plus)
                hyps_ext_emb.append(h)

                mref_ext_emb = []
                for ref in mref:
                    r_max = np.max(ref, axis=0)
                    r_min = np.min(ref, axis=0)
                    r_plus = np.absolute(r_min) <= r_max
                    r = r_max * r_plus + r_min * np.logical_not(r_plus)
                    mref_ext_emb.append(r)
                mrefs_ext_emb.append(mref_ext_emb)
            msims = []
            for hyp_ext_emb, mref_ext_emb in zip(hyps_ext_emb, mrefs_ext_emb):
                msim = self._cosine_similarity(np.array([hyp_ext_emb]*len(mref_ext_emb)), np.array(mref_ext_emb))
                msims.append(msim.tolist())
            return msims
        elif method == 'greedy':
            sims = []
            for hyp, ref in zip(hyps_emb, refs_emb):
                hyp = np.array(hyp)
                ref = np.array(ref).T
                sim = (np.matmul(hyp, ref) / (np.sqrt(np.matmul(np.sum(hyp * hyp, axis=1, keepdims=True), np.sum(ref * ref, axis=0, keepdims=True)))+1e-10))
                sim = np.max(sim, axis=0).mean()
                sims.append(sim)
            return sims
        elif method == "multi_ref_greedy":
            msims = []
            for hyp, mref in zip(hyps_emb, refs_emb):
                hyp = np.array(hyp)
                msim = []
                for ref in mref:
                    ref = np.array(ref).T
                    sim = (np.matmul(hyp, ref) / (np.sqrt(np.matmul(np.sum(hyp * hyp, axis=1, keepdims=True), np.sum(ref * ref, axis=0, keepdims=True)))+1e-10))
                    sim = np.max(sim, axis=0).mean()
                    msim.append(sim)
                msims.append(msim)
            return msims
        else:
            raise NotImplementedError

    def batch_sim_bow(self, hyps, refs):
        """Calculate Average/Extrema/Greedy embedding similarities in a batch

        Arguments:
            hyps {list of str} -- list of hypothesis strings
            refs {list of str} -- list of reference strings

        Returns:
            {list of float} Average similarities
            {list of float} Extrema similarities
            {list of float} Greedy similarities
        """
        assert len(hyps) == len(refs)
        hyps_tokens = [self._sent2tokens(hyp) for hyp in hyps]
        refs_tokens = [self._sent2tokens(ref) for ref in refs]

        hyps_emb = [self._tokens2emb(tokens) for tokens in hyps_tokens]
        refs_emb = [self._tokens2emb(tokens) for tokens in refs_tokens]

        emb_avg_scores = self._embedding_metric(hyps_emb, refs_emb, "average")
        emb_ext_scores = self._embedding_metric(hyps_emb, refs_emb, "extrema")
        emb_greedy_scores = self._embedding_metric(hyps_emb, refs_emb, "greedy")

        return emb_avg_scores, emb_ext_scores, emb_greedy_scores

    def batch_multi_ref_sim_bow(self, hyps, mrefs):
        """Calculate multi-referenced Average/Extrema/Greedy embedding similarities in a batch

        Arguments:
            hyps {list of str} -- list of hypothesis strings
            mrefs {list of list of str} -- list of multiple reference strings

        Returns:
            {list of float} Average similarities
            {list of float} Extrema similarities
            {list of float} Greedy similarities
        """
        assert len(hyps) == len(mrefs)
        hyps_tokens = [self._sent2tokens(hyp) for hyp in hyps]
        mrefs_tokens = [[self._sent2tokens(ref) for ref in mref] for mref in mrefs]

        hyps_emb = [self._tokens2emb(tokens) for tokens in hyps_tokens]
        mrefs_emb = [[self._tokens2emb(tokens) for tokens in mref_tokens] for mref_tokens in mrefs_tokens]

        emb_avg_scores = self._embedding_metric(hyps_emb, mrefs_emb, "multi_ref_average")
        emb_ext_scores = self._embedding_metric(hyps_emb, mrefs_emb, "multi_ref_extrema")
        emb_greedy_scores = self._embedding_metric(hyps_emb, mrefs_emb, "multi_ref_greedy")

        return emb_avg_scores, emb_ext_scores, emb_greedy_scores

    def batch_bleu(self, hyps, refs, n=2):
        """Calculate BLEU-n scores in a batch

        Arguments:
            hyps {list of str} -- list of hypothesis strings
            refs {list of str} -- list of reference strings
            n {int} -- n for BLEU-n (default: 2)

        Returns:
            {list of float} list of BLEU scores
        """
        assert len(hyps) == len(refs)
        hyps_token = [self._sent2tokens(hyp) for hyp in hyps]
        refs_token = [self._sent2tokens(ref) for ref in refs]

        weights = [1./n]*n
        scores = []
        for hyp_tokens, ref_tokens in zip(hyps_token, refs_token):
            if len(hyp_tokens) == 0:
                score = 0.0
            else:
                try:
                    score = sentence_bleu(
                        [ref_tokens],
                        hyp_tokens,
                        weights=weights,
                        smoothing_function=SmoothingFunction().method1
                    )
                except e:
                    raise Exception(f"BLEU score error: {e}")
            scores.append(score)
        return scores

    def batch_multi_ref_bleu(self, hyps, mrefs, n=2):
        """Calculate multiple-referenced BLEU-n scores in a batch

        Arguments:
            hyps {list of str} -- list of hypothesis strings
            mrefs {list of list of str} -- list of multiple reference strings
            n {int} -- n for BLEU-n (default: 2)

        Returns:
            {list of float} list of BLEU scores
        """
        assert len(hyps) == len(mrefs)
        hyps_token = [self._sent2tokens(hyp) for hyp in hyps]
        mrefs_tokens = [[self._sent2tokens(ref) for ref in mref] for mref in mrefs]

        weights = [1./n]*n
        scores = []
        for hyp_tokens, mref_tokens in zip(hyps_token, mrefs_tokens):
            if len(hyp_tokens) == 0:
                score = 0.0
            else:
                try:
                    score = sentence_bleu(
                        mref_tokens,
                        hyp_tokens,
                        weights=weights,
                        smoothing_function=SmoothingFunction().method1
                    )
                except e:
                    raise Exception(f"BLEU score error: {e}")
            scores.append(score)
        return scores

    def batch_div_distinct(self, sents):
        """
        distinct-1 distinct-2 metrics for diversity measure proposed
        by Li et al. "A Diversity-Promoting Objective Function for Neural Conversation Models"
        we counted numbers of distinct unigrams and bigrams in the generated responses
        and divide the numbers by total number of unigrams and bigrams.
        The two metrics measure how informative and diverse the generated responses are.
        High numbers and high ratios mean that there is much content in the generated responses,
        and high numbers further indicate that the generated responses are long

        Arguments:
            hyps {list of str} -- list of hypothesis strings

        Returns:
            {float} intra distinct 1 ratio
            {float} intra distinct 2 ratio
            {float} inter distinct 1 ratio
            {float} inter distinct 2 ratio
            {int} intra distinct 1 types
            {int} intra distinct 2 types
            {int} inter distinct 1 types
            {int} inter distinct 2 types
        """
        tokens_lst = [self._sent2tokens(sent) for sent in sents]
        seq_lens = [len(tokens) for tokens in tokens_lst]
        max_seq_len = max(seq_lens)
        seqs = np.array([seq+[0]*(max_seq_len-len(seq)) for seq in tokens_lst])

        batch_size = seqs.shape[0]
        intra_dist1, intra_dist2 = np.zeros(batch_size), np.zeros(batch_size)
        intra_unigram_types, intra_bigram_types = np.zeros(batch_size), np.zeros(batch_size)

        n_unigrams, n_bigrams, n_unigrams_total, n_bigrams_total = 0., 0., 0., 0.
        unigrams_all, bigrams_all = Counter(), Counter()
        for b in range(batch_size):
            unigrams = Counter([tuple(seqs[b, i:i+1]) for i in range(seq_lens[b])])
            bigrams = Counter([tuple(seqs[b, i:i+2]) for i in range(seq_lens[b]-1)])
            intra_dist1[b] = (len(unigrams.items())+1e-12)/(seq_lens[b]+1e-5)
            intra_dist2[b] = (len(bigrams.items())+1e-12)/(max(0, seq_lens[b]-1)+1e-5)
            intra_unigram_types[b] = len(unigrams.items())
            intra_bigram_types[b] = len(bigrams.items())

            unigrams_all.update([tuple(seqs[b, i:i+1]) for i in range(seq_lens[b])])
            bigrams_all.update([tuple(seqs[b, i:i+2]) for i in range(seq_lens[b]-1)])
            n_unigrams_total += seq_lens[b]
            n_bigrams_total += max(0, seq_lens[b]-1)
        intra_unigram_types = np.mean(intra_unigram_types)
        intra_bigram_types = np.mean(intra_bigram_types)
        intra_dist1 = np.mean(intra_dist1)
        intra_dist2 = np.mean(intra_dist2)
        inter_unigram_types = len(unigrams_all.items())+1e-12
        inter_bigram_types = len(bigrams_all.items())+1e-12
        inter_dist1 = inter_unigram_types/(n_unigrams_total+1e-5)
        inter_dist2 = inter_bigram_types/(n_bigrams_total+1e-5)
        return intra_dist1, intra_dist2, inter_dist1, inter_dist2, \
            intra_unigram_types, intra_bigram_types, \
            int(inter_unigram_types), int(inter_bigram_types)

    def batch_coverage(self, hyps, refs):
        """Calculate average word coverage rate in a batch

        Arguments:
            hyps {list of str} -- list of hypothesis strings
            refs {list of str} -- list of reference strings

        Returns:
            {float} average word coverage rate
        """
        hyp_tokens_lst = [self._sent2tokens(sent) for sent in hyps]
        ref_tokens_lst = [self._sent2tokens(sent) for sent in refs]
        hyp_words_sets = [set(tokens) for tokens in hyp_tokens_lst]
        ref_words_sets = [set(tokens) for tokens in ref_tokens_lst]
        coverage_rates = []
        for hyp_words_set, ref_words_set in zip(hyp_words_sets, ref_words_sets):
            if len(ref_words_set) == 0:
                continue
            coverage_set = ref_words_set.intersection(hyp_words_set)
            coverage_rate = 1.0*len(coverage_set)/len(ref_words_set)
            coverage_rates.append(coverage_rate)

        return np.mean(coverage_rates)

    def compute_pc_for_sif_embedding(self, hyps):
        """Compute principle component for SIF embedding from hypotheses

        Arguments:
            hyps {list of str} -- list of hypothesis strings

        Returns:
            principle component
        """
        hyp_tokens_lst = [self._sent2tokens(sent) for sent in hyps]
        hyp_ids = [self._tokens2ids(tokens) for tokens in hyp_tokens_lst]
        hyp_word_probs = [self._ids2probs(ids) for ids in hyp_ids]

        # compute principle component using references
        pc_sent_lens = [len(sent) for sent in hyp_ids]
        max_pc_sent_len = max(pc_sent_lens)
        padded_pc_sent_ids = [word_ids + [0]*(max_pc_sent_len-len(word_ids)) for word_ids in hyp_ids]
        padded_pc_sent_word_probs = [word_probs + [0.0]*(max_pc_sent_len-len(word_probs)) for word_probs in hyp_word_probs]
        padded_pc_sent_ids = np.array(padded_pc_sent_ids)
        padded_pc_sent_word_probs = np.array(padded_pc_sent_word_probs)
        compute_pc_input = get_weighted_average(self.emb_mat, padded_pc_sent_ids, padded_pc_sent_word_probs)
        pc = compute_pc(compute_pc_input)

        return pc

    def batch_sif_emb_sim(self, hyps, refs, pc=None):
        """Calculate SIF embedding similarities in a batch

        Arguments:
            hyps {list of str} -- list of hypothesis strings
            refs {list of str} -- list of reference strings

        Returns:
            {list of float} similarities
        """
        hyp_tokens_lst = [self._sent2tokens(sent) for sent in hyps]
        ref_tokens_lst = [self._sent2tokens(sent) for sent in refs]

        hyp_ids = [self._tokens2ids(tokens) for tokens in hyp_tokens_lst]
        ref_ids = [self._tokens2ids(tokens) for tokens in ref_tokens_lst]
        hyp_word_probs = [self._ids2probs(ids) for ids in hyp_ids]
        ref_word_probs = [self._ids2probs(ids) for ids in ref_ids]

        concat_ids = hyp_ids+ref_ids
        concat_word_probs = hyp_word_probs+ref_word_probs

        # pad and make np array
        sent_lens = [len(sent) for sent in concat_ids]
        max_len = max(sent_lens)
        concat_ids = [word_ids + [0]*(max_len-len(word_ids)) for word_ids in concat_ids]
        concat_word_probs = [word_probs + [0.0]*(max_len-len(word_probs)) for word_probs in concat_word_probs]
        concat_ids = np.array(concat_ids)
        concat_word_probs = np.array(concat_word_probs)

        # compute principle component using references
        if pc is None:
            pc_sent_lens = [len(sent) for sent in hyp_ids]
            max_pc_sent_len = max(pc_sent_lens)
            padded_pc_sent_ids = [word_ids + [0]*(max_pc_sent_len-len(word_ids)) for word_ids in hyp_ids]
            padded_pc_sent_word_probs = [word_probs + [0.0]*(max_pc_sent_len-len(word_probs)) for word_probs in hyp_word_probs]
            padded_pc_sent_ids = np.array(padded_pc_sent_ids)
            padded_pc_sent_word_probs = np.array(padded_pc_sent_word_probs)
            compute_pc_input = get_weighted_average(self.emb_mat, padded_pc_sent_ids, padded_pc_sent_word_probs)
            pc = compute_pc(compute_pc_input)

        sif_embs = SIF_embedding(self.emb_mat, concat_ids, concat_word_probs, pc=pc)
        n_sents = len(hyp_ids)
        hyp_embs = sif_embs[:n_sents]
        ref_embs = sif_embs[-n_sents:]

        similarities = self._cosine_similarity(hyp_embs, ref_embs)
        return similarities
