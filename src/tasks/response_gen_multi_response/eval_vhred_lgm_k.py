import json
import random
import code
import time
import sys
import math
import argparse
import os

from tqdm import tqdm
import torch
import numpy as np

from model.response_eval.roberta import Roberta
from model.response_gen_multi_response.vhred import VHRED
from utils.metrics import SentenceMetrics
from utils.helpers import StatisticsReporter
from utils.config import ConfigFromDict
from tokenization.roberta_tokenizer import ModRobertaTokenizer
from tokenization.whitespace_tokenizer import WhiteSpaceTokenizer
from tasks.response_gen_multi_response.data_source import DataSource

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str2bool(v):
    return v.lower() in ('true', '1', "True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model - architecture
    parser.add_argument("--model", type=str, default="vhred")
    parser.add_argument("--rnn_type", type=str, default="gru", help="[gru, lstm]")
    parser.add_argument("--floor_encoder", type=str, default="rel", help="floor encoder type in [none, rel, abs]")
    parser.add_argument("--use_attention", type=str2bool, default=True, help="use attention for decoder")
    parser.add_argument("--tie_weights", type=str2bool, default=True, help="tie weights for decoder")
    parser.add_argument("--tokenizer", type=str, default="ws", help="[ws]")

    # model - numbers
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--history_len", type=int, default=5, help="number of history sentences")
    parser.add_argument("--word_embedding_dim", type=int, default=200)
    parser.add_argument("--attr_embedding_dim", type=int, default=30)
    parser.add_argument("--sent_encoder_hidden_dim", type=int, default=500)
    parser.add_argument("--n_sent_encoder_layers", type=int, default=1)
    parser.add_argument("--dial_encoder_hidden_dim", type=int, default=500)
    parser.add_argument("--n_dial_encoder_layers", type=int, default=1)
    parser.add_argument("--decoder_hidden_dim", type=int, default=500)
    parser.add_argument("--n_decoder_layers", type=int, default=1)
    # -- variational model
    parser.add_argument("--latent_dim", type=int, default=200)
    parser.add_argument("--n_components", type=int, default=1)
    parser.add_argument("--gaussian_mix_type", type=str, default="lgm")

    # other
    parser.add_argument("--seed", type=int, default=42, help="random initialization seed")
    parser.add_argument("--max_uttr_len", type=int, default=40, help="max utterance length for trauncation")
    parser.add_argument("--eval_batch_size", type=int, default=100, help="batch size for evaluation")
    parser.add_argument("--evaluator_batch_size", type=int, default=100, help="batch size for evaluator")

    # inference
    parser.add_argument("--n_sample_times", type=int, default=1)
    parser.add_argument("--decode_max_len", type=int, default=40, help="max utterance length for decoding")
    parser.add_argument("--gen_type", type=str, default="greedy", help="[greedy, sample, top]")
    parser.add_argument("--temp", type=float, default=1.0, help="temperature for decoding")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)

    # management
    parser.add_argument("--model_path", help="path to model")
    parser.add_argument("--evaluator_model_path", type=str, default="../data/dd/model/response_eval/roberta_large.floor_none.seed_42.2020-04-01-12:28:50.semi.supervised_by_overall.model.pt", help="path to evaluator model")
    parser.add_argument("--corpus", type=str, default="dd", help="[dd]")
    parser.add_argument("--enable_log", type=str2bool, default=False)
    parser.add_argument("--save_samples", type=str2bool, default=False)
    config = parser.parse_args()

    # load corpus config
    if config.corpus == "dd":
        from corpora.dd.config import Config
    corpus_config = Config(task="response_gen_multi_response")

    # merge parse args with corpus config
    # priority: parse args > corpus config
    corpus_config_dict = {}
    for k, v in corpus_config.__dict__.items():
        if not k.startswith("__") and k not in config.__dict__:
            corpus_config_dict[k] = v
    config.__dict__.update(corpus_config_dict)

    # define logger
    MODEL_NAME = config.model_path[config.model_path.rfind("/")+1:]
    # MODEL_NAME = ""
    GEN_METHOD = f"{config.gen_type}_temp{config.temp}_k{config.top_k}_p{config.top_p}"
    MODEL_NAME = f"{MODEL_NAME}.{GEN_METHOD}"

    def mlog(s):
        if config.enable_log:
            with open(f"../log/{config.corpus}/{config.task}/{MODEL_NAME}.eval_vhred_mog_k.log", "a+", encoding="utf-8") as log_f:
                log_f.write(s+"\n")
        print(s)

    # set random seeds
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # tokenizers
    special_token_dict = {
        "speaker1_token": "<speaker1>",
        "speaker2_token": "<speaker2>"
    }
    if config.tokenizer == "ws":
        tokenizer = WhiteSpaceTokenizer(
            word_count_path=config.word_count_path,
            vocab_size=config.vocab_size,
            special_token_dict=special_token_dict
        )
    eval_tokenizer = WhiteSpaceTokenizer(config.word_count_path, 100000)

    # data loaders
    intrinsic_stat_reporter = StatisticsReporter()
    with open(config.dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    mlog("----- Loading test data -----")
    test_data_source = DataSource(
        data=dataset["test"],
        config=config,
        tokenizer=tokenizer
    )
    mlog(str(test_data_source.statistics))

    # metrics calculator
    metrics = SentenceMetrics(config.eval_word_embedding_path, eval_tokenizer)
    evaluator_config = ConfigFromDict({
        "model_path": config.evaluator_model_path,
        "model_size": "large",
    })
    evaluator_tokenizer = ModRobertaTokenizer(
        model_size=evaluator_config.model_size,
        special_token_dict=special_token_dict
    )
    evaluator = Roberta(evaluator_config, evaluator_tokenizer)
    if torch.cuda.is_available():
        evaluator = evaluator.cuda()
    evaluator.load_model(evaluator_config.model_path)
    mlog(f"Loaded pretrained evaluator at {evaluator_config.model_path}")
    evaluator.eval()

    # build model
    if config.model == "vhred":
        Model = VHRED
    model = Model(config, tokenizer)

    # model adaption
    if torch.cuda.is_available():
        mlog("----- Using GPU -----")
        model = model.cuda()
    if config.model_path:
        model.load_model(config.model_path)
        mlog("----- Model loaded -----")
        mlog("model path: {}".format(config.model_path))

    # log hyper parameters
    start_time = time.time()
    mlog("----- Hyper-parameters -----")
    for k, v in sorted(dict(config.__dict__).items()):
        mlog("{}: {}".format(k, v))

    # here we go
    # Test
    model.eval()
    k2mhyps = {}
    for assign_k in range(config.n_components):
        mlog(f"***** Assignated component {assign_k} *****")
        ctxs = []
        refs = []
        hyps = []
        mhyps = []
        pies = []
        test_data_source.epoch_init(shuffle=False)
        for _ in tqdm(range(math.ceil(len(test_data_source)/config.eval_batch_size))):
            batch_data = test_data_source.next(config.eval_batch_size)
            if batch_data is None:
                break

            batch_ctxs = batch_data["X"].tolist()
            batch_floors = batch_data["X_floor"].tolist()
            for idx in range(len(batch_ctxs)):
                ctx_seq = batch_ctxs[idx]
                ctx_floors = batch_floors[idx]
                ctx_lst = []
                for uttr_idx in range(len(ctx_seq)):
                    ctx = ctx_seq[uttr_idx]
                    if ctx[0] == tokenizer.pad_token_id:
                        continue
                    ctx = tokenizer.convert_ids_to_tokens(
                        ids=ctx,
                        trim_bos=True,
                        trim_from_eos=True,
                        trim_pad=True
                    )
                    ctx = tokenizer.convert_tokens_to_string(ctx)
                    ctx_floor = ctx_floors[uttr_idx]
                    ctx_floor = "A" if ctx_floor == 1 else "B"
                    ctx_lst.append((ctx, ctx_floor))
                ctxs.append(ctx_lst)

            batch_refs = batch_data["Y"].tolist()
            batch_floors = batch_data["Y_floor"].tolist()
            for idx in range(len(batch_refs)):
                ref = batch_refs[idx]
                ref = tokenizer.convert_ids_to_tokens(
                    ids=ref,
                    trim_bos=True,
                    trim_from_eos=True,
                    trim_pad=True
                )
                ref = tokenizer.convert_tokens_to_string(ref)
                ref_floor = "A" if batch_floors[idx] == 1 else "B"
                refs.append((ref, ref_floor))

            batch_mhyps = [[] for _ in range(len(batch_refs))]
            for sample_idx in range(config.n_sample_times):
                ret_data, ret_stat = model.test_step(batch_data, assign_k=assign_k)

                batch_hyps = ret_data["symbols"].tolist()
                for idx in range(len(batch_hyps)):
                    hyp = batch_hyps[idx]
                    hyp = tokenizer.convert_ids_to_tokens(
                        ids=hyp,
                        trim_bos=True,
                        trim_from_eos=True,
                        trim_pad=True,
                    )
                    hyp = tokenizer.convert_tokens_to_string(hyp)
                    hyp_floor = "A" if batch_floors[idx] == 1 else "B"
                    batch_mhyps[idx].append((hyp, hyp_floor))
            mhyps += batch_mhyps
            hyps += [mhyp[0] for mhyp in batch_mhyps]

            ret_data, ret_stat = model.test_step(batch_data)
            pies.append(ret_data["pi"])

            ret_data, ret_stat = model.evaluate_step(batch_data, assign_k=assign_k)
            intrinsic_stat_reporter.update_data(ret_stat)

        ref_texts = [text for (text, floor) in refs]
        hyp_texts = [text for (text, floor) in hyps]
        mhyp_texts = [[text for (text, floor) in mhyp] for mhyp in mhyps]
        k2mhyps[assign_k] = mhyp_texts
        assert len(ref_texts) == len(hyp_texts)

        # Intrinsic evaluation
        if assign_k == 0:
            pies = torch.cat(pies, dim=0)
            avg_pi = pies.mean(dim=0)
            avg_pi_std = avg_pi.std()
            mlog(f"avg pi std: {avg_pi_std}")
            mlog(f"avg pi:")
            for idx, pi in enumerate(avg_pi.tolist()):
                mlog(f"\t{idx}: {pi}")

        log_s = f"\n<Tst> - {time.time()-start_time:.3f}s - "
        log_s += intrinsic_stat_reporter.to_string()
        mlog(log_s)

        # Extrinsic evaluation metrics
        # BLEU
        bleu_scores = metrics.batch_bleu(hyp_texts, ref_texts)
        bleu = np.mean(bleu_scores)
        # Embedding similarities
        avg_emb_sims, ext_emb_sims, greedy_emb_sims = metrics.batch_sim_bow(hyp_texts, ref_texts)
        avg_emb_sim = np.mean(avg_emb_sims)
        ext_emb_sim = np.mean(ext_emb_sims)
        greedy_emb_sim = np.mean(greedy_emb_sims)
        # SIF embedding similarity
        sif_emb_sims = metrics.batch_sif_emb_sim(hyp_texts, ref_texts)
        sif_emb_sim = np.mean(sif_emb_sims)
        # Distinct n-grams
        intra_dist1, intra_dist2, inter_dist1, inter_dist2, \
            intra_types1, intra_types2, inter_types1, inter_types2 \
            = metrics.batch_div_distinct(hyp_texts)
        # Average sentence length
        hyp_tokens_lst = [eval_tokenizer.convert_string_to_tokens(sent) for sent in hyp_texts]
        hyp_lens = [len(tokens) for tokens in hyp_tokens_lst]
        avg_len = np.mean(hyp_lens)
        # Output
        log_s = \
            f"\n<Tst> - {time.time()-start_time:.3f}s - \n"\
            f"\tbleu:          {bleu:.5g}\n"\
            f"\tbow extrema:   {ext_emb_sim:.5g}\n"\
            f"\tbow avg:       {avg_emb_sim:.5g}\n"\
            f"\tbow greedy:    {greedy_emb_sim:.5g}\n"\
            f"\tSIF emb sim:   {sif_emb_sim:.5g}\n"\
            f"\tintra dist 1:  {intra_dist1:.5g}\n"\
            f"\tintra dist 2:  {intra_dist2:.5g}\n"\
            f"\tinter dist 1:  {inter_dist1:.5g}\n"\
            f"\tinter dist 2:  {inter_dist2:.5g}\n"\
            f"\tintra types 1: {intra_types1:.5g}\n"\
            f"\tintra types 2: {intra_types2:.5g}\n"\
            f"\tinter types 1: {inter_types1}\n"\
            f"\tinter types 2: {inter_types2}\n"\
            f"\tavg length:    {avg_len:.5g}"
        mlog(log_s)
        # Evaluator
        scores = []
        idx = 0
        n_batches = math.ceil(len(ctxs)/config.evaluator_batch_size)
        for batch_idx in tqdm(range(n_batches)):
            batch_ctx = ctxs[batch_idx*config.evaluator_batch_size:(batch_idx+1)*config.evaluator_batch_size]
            batch_hyp = hyps[batch_idx*config.evaluator_batch_size:(batch_idx+1)*config.evaluator_batch_size]
            batch_scores = evaluator.predict(batch_ctx, batch_hyp)
            scores += batch_scores
        evaluator_score = np.mean(scores)
        log_s = f"\n<Tst> - {time.time()-start_time:.3f}s - evaluator score: {evaluator_score:.3f}\n"
        mlog(log_s)

    if config.save_samples:
        eval_outputs = {
            "meta_information": {
                "corpus": config.corpus,
                "model_name": MODEL_NAME
            },
            "samples": []
        }
        pies = torch.cat(pies, dim=0)
        for idx, (ctx, ref, pie) in enumerate(zip(ctxs, refs, pies)):
            sample_json = {
                "sample_idx": idx,
                "context": [],
                "reference": None,
                "pie": pie.tolist(),
                "k2multi_hypotheses": {}
            }
            for uttr, floor in ctx:
                sample_json["context"].append((floor, uttr))
            ref_uttr, ref_floor = ref
            sample_json["reference"] = (ref_floor, ref_uttr)

            for assign_k in range(config.n_components):
                mhyp = k2mhyps[assign_k][idx]
                sample_json["k2multi_hypotheses"][assign_k] = mhyp

            eval_outputs["samples"].append(sample_json)

        if not os.path.exists(f"../log/{config.corpus}/{config.task}/"):
            os.makedirs(f"../log/{config.corpus}/{config.task}/")

        with open(f"../log/{config.corpus}/{config.task}/{MODEL_NAME}.eval_vhred_mog_k_samples.json", "w+", encoding="utf-8") as f:
            json.dump(eval_outputs, f)
