import code
import json
from collections import Counter

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity as cosine
import sklearn as sk
from scipy import stats

from .sif_embedding import SIF_embedding, compute_pc, get_weighted_average

def classification_metrics(true_labels, pred_labels, classes, verbose=True):
    display_str = "\tClassification report: \n"
    if verbose:
        display_str += "{}\n".format(sk.metrics.classification_report(true_labels, pred_labels, target_names=classes, labels=list(range(0,len(classes)))))
    display_str += "\tF1 macro        : {:.1f}\n".format(100*sk.metrics.f1_score(true_labels, pred_labels, average="macro"))
    display_str += "\tF1 micro        : {:.1f}\n".format(100*sk.metrics.f1_score(true_labels, pred_labels, average="micro"))
    display_str += "\tF1 weighted     : {:.1f}\n".format(100*sk.metrics.f1_score(true_labels, pred_labels, average="weighted"))
    display_str += "\tAccruracy       : {:.1f}".format(100*sk.metrics.accuracy_score(true_labels, pred_labels))

    return display_str

def calculateContingency(data_A, data_B):
    n = len(data_A)
    assert len(data_B) == n
    ABrr = 0
    ABrw = 0
    ABwr = 0
    ABww = 0
    for i in range(0,n):
        if(data_A[i]==1 and data_B[i]==1):
            ABrr = ABrr+1
        if (data_A[i] == 1 and data_B[i] == 0):
            ABrw = ABrw + 1
        if (data_A[i] == 0 and data_B[i] == 1):
            ABwr = ABwr + 1
        else:
            ABww = ABww + 1
    return np.array([[ABrr, ABrw], [ABwr, ABww]])

def mcNemar(table):
    statistic = float(np.abs(table[0][1]-table[1][0]))**2/(table[1][0]+table[0][1])
    pval = 1-stats.chi2.cdf(statistic,1)
    return pval

class ClassificationMetrics:
    def __init__(self, classes):
        self.classes = classes
        pass

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

class SentenceMetrics:
    def __init__(self, word_embedding_path, tokenizer):
        self.tokenizer = tokenizer
        self.word2id = tokenizer.word2id
        self.id2word = tokenizer.id2word
        self.id2prob = {}
        for word, prob in tokenizer.word2prob.items():
            if word in self.word2id:
                self.id2prob[self.word2id[word]] = prob

        with open(word_embedding_path) as f:
            self.word2vec = json.load(f)

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
        return self.tokenizer.convert_sent_to_tokens(sent)

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
            hyps = [np.mean(hyp, axis=0) for hyp in hyps_emb]
            refs = [np.mean(ref, axis=0) for ref in refs_emb]
            return self._cosine_similarity(np.array(hyps), np.array(refs))
        elif method == 'extrema':
            hyps = []
            refs = []
            for hyp, ref in zip(hyps_emb, refs_emb):
                h_max = np.max(hyp, axis=0)
                h_min = np.min(hyp, axis=0)
                h_plus = np.absolute(h_min) <= h_max
                h = h_max * h_plus + h_min * np.logical_not(h_plus)
                hyps.append(h)

                r_max = np.max(ref, axis=0)
                r_min = np.min(ref, axis=0)
                r_plus = np.absolute(r_min) <= r_max
                r = r_max * r_plus + r_min * np.logical_not(r_plus)
                refs.append(r)

            return self._cosine_similarity(np.array(hyps), np.array(refs))
        elif method == 'greedy':
            sim_list = []
            for hyp, ref in zip(hyps_emb, refs_emb):
                hyp = np.array(hyp)
                ref = np.array(ref).T
                sim = (np.matmul(hyp, ref)
                    / (np.sqrt(np.matmul(np.sum(hyp * hyp, axis=1, keepdims=True), np.sum(ref * ref, axis=0, keepdims=True)))+1e-10)
                )
                sim = np.max(sim, axis=0)
                sim_list.append(np.mean(sim))

            return np.array(sim_list)
        else:
            raise NotImplementedError

    def batch_sim_bow(self, hyps, refs):
        hyps_tokens = [self._sent2tokens(hyp) for hyp in hyps]
        refs_tokens = [self._sent2tokens(ref) for ref in refs]

        hyps_emb = [self._tokens2emb(tokens) for tokens in hyps_tokens]
        refs_emb = [self._tokens2emb(tokens) for tokens in refs_tokens]

        emb_avg_scores = self._embedding_metric(hyps_emb, refs_emb, "average").tolist()
        emb_ext_scores = self._embedding_metric(hyps_emb, refs_emb, "extrema").tolist()
        emb_greedy_scores = self._embedding_metric(hyps_emb, refs_emb, "greedy").tolist()

        return emb_avg_scores, emb_ext_scores, emb_greedy_scores

    def batch_bleu(self, hyps, refs, n=2):
        """
        :param refs - a list of reference sentence str
        :param hyps - a list of hypothese sentence str
        """
        hyps_token = [self._sent2tokens(hyp) for hyp in hyps]
        refs_token = [self._sent2tokens(ref) for ref in refs]

        weights = [1./n]*n
        scores = []
        for hyp_tokens, ref_tokens in zip(hyps_token, refs_token):
            if len(hyp_tokens) == 0:
                score = 0.0
            else:
                try:
                    score = sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=SmoothingFunction().method1)
                except:
                    code.interact(local=locals())
                    raise Exception("BLEU score error")
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

        :param sents - a list of sentence strs
        """
        tokens_lst = [self._sent2tokens(sent) for sent in sents]
        seq_lens = [len(tokens) for tokens in tokens_lst]
        max_seq_len = max(seq_lens)
        seqs = np.array([seq+[0]*(max_seq_len-len(seq)) for seq in tokens_lst])

        batch_size = seqs.shape[0]
        intra_dist1, intra_dist2 = np.zeros(batch_size), np.zeros(batch_size)
        intra_unigram_types, intra_bigram_types = np.zeros(batch_size), np.zeros(batch_size)

        n_unigrams, n_bigrams, n_unigrams_total , n_bigrams_total = 0. ,0., 0., 0.
        unigrams_all, bigrams_all = Counter(), Counter()
        for b in range(batch_size):
            unigrams= Counter([tuple(seqs[b,i:i+1]) for i in range(seq_lens[b])])
            bigrams = Counter([tuple(seqs[b,i:i+2]) for i in range(seq_lens[b]-1)])
            intra_dist1[b]=(len(unigrams.items())+1e-12)/(seq_lens[b]+1e-5)
            intra_dist2[b]=(len(bigrams.items())+1e-12)/(max(0, seq_lens[b]-1)+1e-5)
            intra_unigram_types[b] = len(unigrams.items())
            intra_bigram_types[b] = len(bigrams.items())

            unigrams_all.update([tuple(seqs[b,i:i+1]) for i in range(seq_lens[b])])
            bigrams_all.update([tuple(seqs[b,i:i+2]) for i in range(seq_lens[b]-1)])
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

    def batch_coverage(self, preds, refs):
        """
        average word coverage rate of pairwise inputs

        :param preds - a list of predicted sentence strs
        :param refs - a list of reference sentence strs
        """
        pred_tokens_lst = [self._sent2tokens(sent) for sent in preds]
        ref_tokens_lst = [self._sent2tokens(sent) for sent in refs]
        pred_words_sets = [set(tokens) for tokens in pred_tokens_lst]
        ref_words_sets = [set(tokens) for tokens in ref_tokens_lst]
        coverage_rates = []
        for pred_words_set, ref_words_set in zip(pred_words_sets, ref_words_sets):
            if len(ref_words_set) == 0:
                continue
            coverage_set = ref_words_set.intersection(pred_words_set)
            coverage_rate = 1.0*len(coverage_set)/len(ref_words_set)
            coverage_rates.append(coverage_rate)

        return np.mean(coverage_rates)

    def compute_pc_for_sif_embedding(self, preds):
        """
        pc for sif embedding from pred uttrs

        :param preds - a list of predicted sentence strs
        """
        pred_tokens_lst = [self._sent2tokens(sent) for sent in preds]
        pred_ids = [self._tokens2ids(tokens) for tokens in pred_tokens_lst]
        pred_word_probs = [self._ids2probs(ids) for ids in pred_ids]

        # compute principle component using references
        pc_sent_lens = [len(sent) for sent in pred_ids]
        max_pc_sent_len = max(pc_sent_lens)
        padded_pc_sent_ids = [word_ids + [0]*(max_pc_sent_len-len(word_ids)) for word_ids in pred_ids]
        padded_pc_sent_word_probs = [word_probs + [0.0]*(max_pc_sent_len-len(word_probs)) for word_probs in pred_word_probs]
        padded_pc_sent_ids = np.array(padded_pc_sent_ids)
        padded_pc_sent_word_probs = np.array(padded_pc_sent_word_probs)
        compute_pc_input = get_weighted_average(self.emb_mat, padded_pc_sent_ids, padded_pc_sent_word_probs)
        pc = compute_pc(compute_pc_input)

        return pc

    def batch_sif_emb_sim(self, preds, refs, pc=None):
        """
        average sif embedding similarity of pairwise inputs

        :param preds - a list of predicted sentence strs
        :param refs - a list of reference sentence strs
        """
        pred_tokens_lst = [self._sent2tokens(sent) for sent in preds]
        ref_tokens_lst = [self._sent2tokens(sent) for sent in refs]

        pred_ids = [self._tokens2ids(tokens) for tokens in pred_tokens_lst]
        ref_ids = [self._tokens2ids(tokens) for tokens in ref_tokens_lst]
        pred_word_probs = [self._ids2probs(ids) for ids in pred_ids]
        ref_word_probs = [self._ids2probs(ids) for ids in ref_ids]

        concat_ids = pred_ids+ref_ids
        concat_word_probs = pred_word_probs+ref_word_probs

        # pad and make np array
        sent_lens = [len(sent) for sent in concat_ids]
        max_len = max(sent_lens)
        concat_ids = [word_ids + [0]*(max_len-len(word_ids)) for word_ids in concat_ids]
        concat_word_probs = [word_probs + [0.0]*(max_len-len(word_probs)) for word_probs in concat_word_probs]
        concat_ids = np.array(concat_ids)
        concat_word_probs = np.array(concat_word_probs)

        # compute principle component using references
        if pc is None:
            pc_sent_lens = [len(sent) for sent in pred_ids]
            max_pc_sent_len = max(pc_sent_lens)
            padded_pc_sent_ids = [word_ids + [0]*(max_pc_sent_len-len(word_ids)) for word_ids in pred_ids]
            padded_pc_sent_word_probs = [word_probs + [0.0]*(max_pc_sent_len-len(word_probs)) for word_probs in pred_word_probs]
            padded_pc_sent_ids = np.array(padded_pc_sent_ids)
            padded_pc_sent_word_probs = np.array(padded_pc_sent_word_probs)
            compute_pc_input = get_weighted_average(self.emb_mat, padded_pc_sent_ids, padded_pc_sent_word_probs)
            pc = compute_pc(compute_pc_input)

        sif_embs = SIF_embedding(self.emb_mat, concat_ids, concat_word_probs, pc=pc)
        n_sents = len(pred_ids)
        pred_embs = sif_embs[:n_sents]
        ref_embs = sif_embs[-n_sents:]

        similarities = self._cosine_similarity(pred_embs, ref_embs)
        return similarities
