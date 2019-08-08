from __future__ import absolute_import
import json
import random
import code
import time
import sys
import math
import argparse
import os

import torch
import numpy as np

from model.response_gen.s2s import S2S
from model.response_gen.hred import HRED
from model.response_gen.hred_sep_uttr_enc import HREDSepUttrEnc
from model.response_gen.vhred import VHRED
from model.response_gen.vhcr import VHCR
from model.response_gen.gpt2 import GPT2
from utils.helpers import metric_is_improving, StatisticsReporter
from utils.metrics import SentenceMetrics
from tokenization.basic_tokenizer import BasicTokenizer
from tokenization.gpt2_tokenizer import ModGPT2Tokenizer
from .data_source import DataSource

def str2bool(v):
    return v.lower() in ('true', '1', "True")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model - architecture
    parser.add_argument("--model", type=str, default="hred", help="[s2s, hred, hred_sep_uttr_enc, vhred, vhcr, gpt2]")
    parser.add_argument("--model_size", type=str, default=None, help="[small, medium], model size for GPT2")
    parser.add_argument("--rnn_type", type=str, default="lstm", help="[gru, lstm]")
    parser.add_argument("--floor_encoder", type=str, default="none", help="floor encoder type in [none, rel, abs]")
    parser.add_argument("--use_attention", type=str2bool, default=False, help="use attention for decoder")
    parser.add_argument("--tie_weights", type=str2bool, default=True, help="tie weights for decoder")
    parser.add_argument("--tokenizer", type=str, default="basic", help="[basic, gpt2]")

    # model - numbers
    parser.add_argument("--history_len", type=int, default=5, help="number of history sentences")
    parser.add_argument("--word_embedding_dim", type=int, default=200)
    parser.add_argument("--attr_embedding_dim", type=int, default=30)
    parser.add_argument("--sent_encoder_hidden_dim", type=int, default=500)
    parser.add_argument("--n_sent_encoder_layers", type=int, default=2)
    parser.add_argument("--dial_encoder_hidden_dim", type=int, default=500)
    parser.add_argument("--n_dial_encoder_layers", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--decoder_hidden_dim", type=int, default=500)
    parser.add_argument("--n_decoder_layers", type=int, default=2)

    # training
    parser.add_argument("--seed", type=int, default=42, help="random initialization seed")
    parser.add_argument("--max_uttr_len", type=int, default=40, help="max utterance length for trauncation")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout probability")
    parser.add_argument("--l2_penalty", type=float, default=0.0, help="l2 penalty")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument("--init_lr", type=float, default=0.001, help="init learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="init learning rate")
    parser.add_argument("--lr_decay_rate", type=float, default=0.5)
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs for training")
    parser.add_argument("--use_pretrained_word_embedding", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=30, help="batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=60, help="batch size for evaluation")

    # inference
    parser.add_argument("--decode_max_len", type=int, default=40, help="max utterance length for decoding")
    parser.add_argument("--gen_type", type=str, default="greedy", help="[greedy, sample, top]")
    parser.add_argument("--temp", type=float, default=1.0, help="temperature for decoding")
    parser.add_argument("--top_k", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)

    # management
    parser.add_argument("--model_path", help="path to model")
    parser.add_argument("--corpus", type=str, default="dd", help="[dd, cornellmovie]")
    parser.add_argument("--enable_log", type=str2bool, default=False)
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--check_loss_after_n_step", type=int, default=100)
    parser.add_argument("--validate_after_n_step", type=int, default=1000)
    parser.add_argument("--sample_after_n_step", type=int, default=1000)
    config = parser.parse_args()

    # load corpus config
    if config.corpus == "dd":
        from corpora.dd.config import Config
    elif config.corpus == "cornellmovie":
        from corpora.cornellmovie.config import Config
    corpus_config = Config(task="response_gen")

    ## merge parse args with corpus config
    # priority: parse args > corpus config
    corpus_config_dict = {}
    for k, v in corpus_config.__dict__.items():
        if not k.startswith("__") and k not in config.__dict__:
            corpus_config_dict[k] = v
    config.__dict__.update(corpus_config_dict)

    # set random seeds
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # tokenizers
    if config.tokenizer == "basic":
        tokenizer = BasicTokenizer(config.word_count_path, config.vocab_size)
    elif config.tokenizer == "gpt2":
        tokenizer = ModGPT2Tokenizer()

    # data loaders & number reporters
    trn_reporter = StatisticsReporter()
    dev_reporter = StatisticsReporter()
    train_data_source = DataSource(
        config=config,
        tokenizer=tokenizer,
        dataset="train"
    )
    dev_data_source = DataSource(
        config=config,
        tokenizer=tokenizer,
        dataset="dev"
    )
    test_data_source = DataSource(
        config=config,
        tokenizer=tokenizer,
        dataset="test"
    )

    # metrics calculator
    eval_tokenizer = BasicTokenizer(config.word_count_path, config.vocab_size)
    metrics = SentenceMetrics(config.eval_word_embedding_path, eval_tokenizer)

    # build model
    if config.model == "s2s":
        Model = S2S
    elif config.model == "hred":
        Model = HRED
    elif config.model == "hred_sep_uttr_enc":
        Model = HREDSepUttrEnc
    elif config.model == "vhred":
        Model = VHRED
    elif config.model == "vhcr":
        Model = VHCR
    elif config.model == "gpt2":
        Model = GPT2
    model = Model(config, tokenizer)

    # model adaption
    if torch.cuda.is_available():
        print("----- Using GPU -----")
        model = model.cuda()
    if config.model_path:
        model.load_model(config.model_path)
        print("----- Model loaded -----")
        print("model path: {}".format(config.model_path))
    print(str(model))

    # define logger
    MODEL_NAME = Model.__name__
    if config.use_attention:
        MODEL_NAME += "_attn"
    if config.model_size:
        MODEL_NAME += "_{}".format(config.model_size)
    LOG_FILE_NAME = "{}.floor_{}.seed_{}.{}".format(
        MODEL_NAME,
        config.floor_encoder,
        config.seed,
        time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    )
    def mlog(s):
        if config.enable_log:
            if not os.path.exists(f"../log/{config.corpus}/{config.task}"):
                os.makedirs(f"../log/{config.corpus}/{config.task}")

            with open(f"../log/{config.corpus}/{config.task}/{LOG_FILE_NAME}.log", "a+", encoding="utf-8") as log_f:
                log_f.write(s+"\n")
        print(s)

    # log hyper parameters
    start_time = time.time()
    mlog("----- Hyper-parameters -----")
    for k, v in sorted(dict(config.__dict__).items()):
        mlog("{}: {}".format(k, v))

    # here we go
    n_step = 0
    loss_history = []
    lr = config.init_lr
    for epoch in range(1, config.n_epochs+1):
        if lr <= config.min_lr:
            break

        # Train
        n_batch = 0
        train_data_source.epoch_init()
        while True:
            batch_data = train_data_source.next(config.batch_size)
            if batch_data is None:
                break

            model.train()
            if config.model in ["vhred", "vhcr"]:
                ret_statistics = model.train_step(batch_data, lr=lr, step=n_step)
            else:
                ret_statistics = model.train_step(batch_data, lr=lr)
            trn_reporter.update_data(ret_statistics)
            n_step += 1
            n_batch += 1

            # Session result output
            if n_step > 0 and n_step%config.check_loss_after_n_step == 0:
                log_s = f"{time.time()-start_time:.2f}s Epoch {epoch} batch {n_batch} - "
                log_s += trn_reporter.to_string()
                mlog(log_s)
                trn_reporter.clear()

            # Sampling from test dataset
            if n_step > 0 and n_step % config.sample_after_n_step == 0:
                model.eval()

                log_s = "<Test> - Samples:"
                mlog(log_s)
                test_data_source.epoch_init()
                for sample_idx in range(5):
                    batch_data = test_data_source.next(1)

                    ret_dict = model.test_step(batch_data)

                    log_s = "context:\n"
                    context = batch_data["X"].tolist()[0]
                    context_floors = batch_data["X_floor"].tolist()[0]
                    for uttr, floor in zip(context, context_floors):
                        if uttr[0] == tokenizer.pad_id:
                            continue
                        uttr = tokenizer.convert_ids_to_tokens(
                            ids=uttr,
                            trim_bos=True,
                            trim_from_eos=True,
                        )
                        floor = "A" if floor == 1 else "B"
                        log_s += "  {}: {}\n".format(
                            floor,
                            tokenizer.convert_tokens_to_sent(uttr)
                        )
                    mlog(log_s)

                    log_s = "ref text:\n"
                    floor = batch_data["Y_floor"][0].item()
                    floor = "A" if floor == 1 else "B"
                    uttr = batch_data["Y"][0].tolist()
                    uttr = tokenizer.convert_ids_to_tokens(
                        ids=uttr,
                        trim_bos=True,
                        trim_from_eos=True,
                    )
                    log_s += "  {}: {}\n".format(
                        floor,
                        tokenizer.convert_tokens_to_sent(uttr)
                    )
                    mlog(log_s)

                    log_s = "hyp text:\n"
                    hyp = ret_dict["symbols"][0].tolist()
                    hyp = tokenizer.convert_ids_to_tokens(
                        ids=hyp,
                        trim_bos=True,
                        trim_from_eos=True,
                        trim_pad=True,
                    )
                    log_s += "  {}\n".format(
                        tokenizer.convert_tokens_to_sent(hyp)
                    )
                    log_s += "="*30
                    mlog(log_s)

            # Evaluation on dev dataset
            if n_step > 0 and n_step%config.validate_after_n_step == 0:
                model.eval()

                log_s = f"<Dev> learning rate: {lr}\n"
                mlog(log_s)

                dev_data_source.epoch_init(shuffle=False)
                while True:
                    batch_data = dev_data_source.next(config.eval_batch_size)
                    if batch_data is None:
                        break

                    ret_statistics = model.evaluate_step(batch_data)
                    dev_reporter.update_data(ret_statistics)

                log_s = f"\n<Dev> - {time.time()-start_time:.3f}s - "
                log_s += dev_reporter.to_string()
                mlog(log_s)

                # Save model if it has better monitor measurement
                if config.save_model:
                    if not os.path.exists(f"../data/{config.corpus}/model/{config.task}"):
                        os.makedirs(f"../data/{config.corpus}/model/{config.task}")
                    
                    torch.save(model.state_dict(), f"../data/{config.corpus}/model/{config.task}/{LOG_FILE_NAME}.model.pt")
                    mlog(f"model saved to data/{config.corpus}/model/{config.task}/{LOG_FILE_NAME}.model.pt")
                    
                    if torch.cuda.is_available(): 
                        model = model.cuda()

                loss_history.append(dev_reporter.get_value("monitor"))
                dev_reporter.clear()

                # Learning rate decay every epoch
                if not metric_is_improving(loss_history):
                    lr = lr*config.lr_decay_rate

        # Evaluation on test dataset
        model.eval()
        test_data_source.epoch_init(shuffle=False)
        hyps = []
        refs = []
        while True:
            batch_data = test_data_source.next(config.eval_batch_size)
            if batch_data is None:
                break

            ret_dict = model.test_step(batch_data)

            batch_refs = batch_data["Y"].tolist()
            batch_floors = batch_data["Y_floor"].tolist()
            for idx in range(len(batch_refs)):
                ref = batch_refs[idx]
                ref = tokenizer.convert_ids_to_tokens(
                    ids=ref,
                    trim_bos=True,
                    trim_from_eos=True
                )
                ref = tokenizer.convert_tokens_to_sent(ref)
                ref_floor = "A" if batch_floors[idx] == 1 else "B"
                refs.append((ref, ref_floor))

            batch_hyps = ret_dict["symbols"].tolist()
            for idx in range(len(batch_hyps)):
                hyp = batch_hyps[idx]
                hyp = tokenizer.convert_ids_to_tokens(
                    ids=hyp,
                    trim_bos=True,
                    trim_from_eos=True,
                    trim_pad=True
                )
                hyp = tokenizer.convert_tokens_to_sent(hyp)
                hyp_floor = "A" if batch_floors[idx] == 1 else "B"
                hyps.append((hyp, hyp_floor))

        ref_texts = [text for (text, floor) in refs]
        hyp_texts = [text for (text, floor) in hyps]
        assert len(ref_texts) == len(hyp_texts)

        ## Evaluation metrics
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
        hyp_tokens_lst = [eval_tokenizer.convert_sent_to_tokens(sent) for sent in hyp_texts]
        hyp_lens = [len(tokens) for tokens in hyp_tokens_lst]
        avg_len = np.mean(hyp_lens)
        # Output
        log_s = \
            f"\n<Tst> - {time.time()-start_time:.3f}s -"\
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
