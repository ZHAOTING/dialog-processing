import json
import random
import code
import time
import sys
import math
import argparse
import os

import numpy as np
import torch
import torch.optim as optim

from model.response_gen_multi_response.hred import HRED
from model.response_gen_multi_response.vhred import VHRED
from model.response_gen_multi_response.vhred_multi_avg import VHREDMultiAvg
from model.response_gen_multi_response.mhred import Mechanism_HRED
from model.response_gen_multi_response.hred_cvar import HRED_CVaR
from model.response_gen_multi_response.hred_student import HREDStudent
from model.response_gen_multi_response.gpt2 import GPT2
from utils.helpers import StatisticsReporter
from utils.metrics import SentenceMetrics
from utils.config import ConfigFromDict
from tokenization.whitespace_tokenizer import WhiteSpaceTokenizer
from tokenization.gpt2_tokenizer import ModGPT2Tokenizer
from tasks.response_gen_multi_response.data_source import DataSource


def str2bool(v):
    return v.lower() in ('true', '1', "True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model - architecture
    parser.add_argument("--model", type=str, default="hred", help="[hred, vhred, mhred, hred_cvarvhred_multi_avg, hred_student, gpt2]")
    parser.add_argument("--rnn_type", type=str, default="gru", help="[gru, lstm]")
    parser.add_argument("--floor_encoder", type=str, default="rel", help="floor encoder type in [none, rel, abs]")
    parser.add_argument("--use_attention", type=str2bool, default=True, help="use attention for decoder")
    parser.add_argument("--tie_weights", type=str2bool, default=True, help="tie weights for decoder")
    parser.add_argument("--tokenizer", type=str, default="ws", help="[ws]")
    # -- gpt2
    parser.add_argument("--model_size", type=str, help="[small, medium, distilgpt2], model size for GPT2")

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
    parser.add_argument("--gaussian_mix_type", type=str, default="gmm")
    parser.add_argument("--n_attn_layers", type=int, default=1)
    # -- mechanism model
    parser.add_argument("--n_mechanisms", type=int, default=5)

    # training
    parser.add_argument("--seed", type=int, default=42, help="random initialization seed")
    parser.add_argument("--max_uttr_len", type=int, default=40, help="max utterance length for trauncation")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout probability")
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs for training")
    parser.add_argument("--use_pretrained_word_embedding", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=30, help="batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=60, help="batch size for evaluation")
    parser.add_argument("--shuffle_data", type=str2bool, default=True)
    # -- distillation
    parser.add_argument("--use_teacher_word_embedding", type=str2bool, default=True)
    parser.add_argument("--guidance_on", type=str, default="all")
    # -- multiple reference training
    parser.add_argument("--use_ground_truth", type=str2bool, default=True)
    parser.add_argument("--use_flattened_hyps", type=str2bool, default=False)
    parser.add_argument("--use_nested_hyps", type=str2bool, default=False)
    parser.add_argument("--n_hyps", type=int, default=0, help="number of hypothesis responses to use for training")
    parser.add_argument("--n_nested_hyps", type=int, default=1, help="number of nested hypothesis responses to use for training")
    # -- variational model
    parser.add_argument("--n_step_annealing", type=int, default=40000, help="number of steps for KLD annealing in variational models")
    parser.add_argument("--use_bow_loss", type=str2bool, default=True, help="use BoW loss in variational models")

    # optimizer
    parser.add_argument("--l2_penalty", type=float, default=0.0, help="l2 penalty")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument("--init_lr", type=float, default=1e-3, help="init learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="minimum learning rate for early stopping")
    parser.add_argument("--lr_decay_rate", type=float, default=0.75)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="gradient clipping")

    # inference
    parser.add_argument("--decode_max_len", type=int, default=40, help="max utterance length for decoding")
    parser.add_argument("--gen_type", type=str, default="greedy", help="[greedy, sample, top]")
    parser.add_argument("--temp", type=float, default=1.0, help="temperature for decoding")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)

    # management
    parser.add_argument("--model_path", help="path to model")
    parser.add_argument("--corpus", type=str, default="dd", help="[dd]")
    parser.add_argument("--enable_log", type=str2bool, default=False)
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--check_loss_after_n_step", type=int, default=100)
    parser.add_argument("--validate_after_n_step", type=int, default=1000)
    parser.add_argument("--sample_after_n_step", type=int, default=1000)
    parser.add_argument("--filename_note", type=str, help="take a note in saved files' names")
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
    MODEL_NAME = config.model
    if config.use_attention:
        MODEL_NAME += "_attn"
    LOG_FILE_NAME = "{}.floor_{}.seed_{}.{}".format(
        MODEL_NAME,
        config.floor_encoder,
        config.seed,
        time.strftime("%Y%m%d-%H%M%S", time.localtime())
    )
    if config.filename_note:
        LOG_FILE_NAME += f".{config.filename_note}"

    def mlog(s):
        if config.enable_log:
            if not os.path.exists(f"../log/{config.corpus}/{config.task}"):
                os.makedirs(f"../log/{config.corpus}/{config.task}")

            with open(f"../log/{config.corpus}/{config.task}/{LOG_FILE_NAME}.log", "a+", encoding="utf-8") as log_f:
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
    elif config.tokenizer == "gpt2":
        tokenizer = ModGPT2Tokenizer(
            model_size=config.model_size,
            special_token_dict=special_token_dict
        )

    # data loaders & number reporters
    trn_reporter = StatisticsReporter()
    dev_reporter = StatisticsReporter()
    with open(f"{config.dataset_path}.aggregate", encoding="utf-8") as f:
        dataset = json.load(f)
    mlog("----- Loading training data -----")
    train_data_source = DataSource(
        data=dataset["train"],
        config=config,
        tokenizer=tokenizer,
        hyp_source_model_names=[
            "gpt2_medium.floor_rel.seed_42.20200407-135521.model.pt.top_temp1.0_k0_p0.95",
        ],
        use_ground_truth=config.use_ground_truth,
        use_flattened_hyps=config.use_flattened_hyps,
        use_nested_hyps=config.use_nested_hyps,
        n_nested_hyps=config.n_nested_hyps,
        deduplicate_hyps=(config.model == "hred_weight")
    )
    mlog(str(train_data_source.statistics))
    mlog("----- Loading dev data -----")
    dev_data_source = DataSource(
        data=dataset["dev"],
        config=config,
        tokenizer=tokenizer,
        use_ground_truth=True,
        use_nested_hyps=config.use_nested_hyps,
        n_nested_hyps=1,
    )
    mlog(str(dev_data_source.statistics))
    mlog("----- Loading test data -----")
    test_data_source = DataSource(
        data=dataset["test"],
        config=config,
        tokenizer=tokenizer
    )
    mlog(str(test_data_source.statistics))
    mlog("----- Loading test sample data -----")
    test_sample_data_source = DataSource(
        data=dataset["test"],
        config=config,
        tokenizer=tokenizer
    )
    mlog(str(test_sample_data_source.statistics))
    del dataset

    # metrics calculator
    eval_tokenizer = WhiteSpaceTokenizer(config.word_count_path, config.vocab_size)
    metrics = SentenceMetrics(config.eval_word_embedding_path, eval_tokenizer)

    # build teacher model
    if config.model in ["hred_student"]:
        teacher_config = ConfigFromDict({
            "model_size": "small",
            # "model_size": "medium",
            "floor_encoder": "rel",
            "decode_max_len": config.decode_max_len,
            "model_path": "../data/dd/model/response_gen/gpt2_small.floor_none.seed_42.20200407-133531.model.pt",
            # "model_path": "../data/dd/model/response_gen/gpt2_medium.floor_rel.seed_42.20200407-135521.model.pt"
        })
        teacher_tokenizer = ModGPT2Tokenizer(
            model_size="small",
            # model_size="medium",
            special_token_dict=special_token_dict
        )
        teacher_model = GPT2(teacher_config, teacher_tokenizer)
        if torch.cuda.is_available():
            teacher_model = teacher_model.cuda()
        teacher_model.load_model(teacher_config.model_path)
        mlog("----- Teacher Model loaded -----")
        mlog("model path: {}".format(teacher_config.model_path))
        teacher_model.eval()

    # build model
    if config.model == "hred":
        Model = HRED
    elif config.model == "vhred":
        Model = VHRED
    elif config.model == "vhred_multi_avg":
        Model = VHREDMultiAvg
    elif config.model == "gpt2":
        Model = GPT2
    elif config.model == "mhred":
        Model = Mechanism_HRED
    elif config.model == "hred_cvar":
        Model = HRED_CVaR
    elif config.model == "hred_student":
        Model = HREDStudent
    if config.model in ["hred_student"]:
        model = Model(config, tokenizer, teacher_model)
    else:
        model = Model(config, tokenizer)

    # model adaption
    if torch.cuda.is_available():
        mlog("----- Using GPU -----")
        model = model.cuda()
    if config.model_path:
        model.load_model(config.model_path)
        mlog("----- Model loaded -----")
        mlog("model path: {}".format(config.model_path))

    # Build optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.init_lr,
        weight_decay=config.l2_penalty
    )

    # Build lr scheduler
    if config.lr_decay_rate < 1.0:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=config.lr_decay_rate,
            patience=config.patience,
        )

    # log hyper parameters
    start_time = time.time()
    mlog("----- Hyper-parameters -----")
    for k, v in sorted(dict(config.__dict__).items()):
        mlog("{}: {}".format(k, v))

    # here we go
    n_step = 0
    for epoch in range(1, config.n_epochs+1):
        if config.lr_decay_rate < 1.0:
            lr = list(lr_scheduler.optimizer.param_groups)[0]["lr"]
            if lr <= config.min_lr:
                break

        # Train
        n_batch = 0
        train_data_source.epoch_init(shuffle=config.shuffle_data)
        while True:
            model.train()

            # batch data
            batch_data = train_data_source.next(config.batch_size)
            if batch_data is None:
                break

            # forward
            if config.model in ["hred_student"]:
                ret_data, ret_stat = model.train_step(batch_data, teacher_model)
            elif "vhred" in config.model:
                ret_data, ret_stat = model.train_step(batch_data, step=n_step)
            else:
                ret_data, ret_stat = model.train_step(batch_data)
            trn_reporter.update_data(ret_stat)

            # backward
            loss = ret_data["loss"]
            loss.backward()

            # update
            if config.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip
                )
            optimizer.step()
            optimizer.zero_grad()

            # check loss
            if n_step > 0 and n_step % config.check_loss_after_n_step == 0:
                log_s = f"{time.time()-start_time:.2f}s Epoch {epoch} batch {n_batch} - "
                log_s += trn_reporter.to_string()
                trn_reporter.clear()
                mlog(log_s)

            # Sampling from test dataset
            if n_step > 0 and n_step % config.sample_after_n_step == 0:
                model.eval()

                log_s = "<Test> - Samples:"
                mlog(log_s)
                test_sample_data_source.epoch_init(shuffle=True)
                for sample_idx in range(5):
                    batch_data = test_sample_data_source.next(1)

                    log_s = "context:\n"
                    context = batch_data["X"].tolist()[0]
                    context_floors = batch_data["X_floor"].tolist()[0]
                    for uttr, floor in zip(context, context_floors):
                        if uttr[0] == tokenizer.pad_token_id:
                            continue
                        uttr = tokenizer.convert_ids_to_tokens(
                            ids=uttr,
                            trim_bos=True,
                            trim_from_eos=True,
                            trim_pad=True
                        )
                        floor = "A" if floor == 1 else "B"
                        log_s += "  {}: {}\n".format(
                            floor,
                            tokenizer.convert_tokens_to_string(uttr)
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
                        trim_pad=True
                    )
                    log_s += "  {}: {}\n".format(
                        floor,
                        tokenizer.convert_tokens_to_string(uttr)
                    )
                    mlog(log_s)

                    log_s = "hyp text:\n"
                    for hyp_idx in range(1):
                        ret_data, ret_stat = model.test_step(batch_data)
                        hyp = ret_data["symbols"][0].tolist()
                        hyp = tokenizer.convert_ids_to_tokens(
                            ids=hyp,
                            trim_bos=True,
                            trim_from_eos=True,
                            trim_pad=True,
                        )
                        log_s += "  {}: {}\n".format(
                            hyp_idx,
                            tokenizer.convert_tokens_to_string(hyp)
                        )
                    log_s += "="*30
                    mlog(log_s)

            # Evaluation on dev dataset
            if n_step > 0 and n_step % config.validate_after_n_step == 0:
                model.eval()

                if config.lr_decay_rate < 1.0:
                    lr = list(lr_scheduler.optimizer.param_groups)[0]["lr"]
                    log_s = f"<Dev> learning rate: {lr}\n"
                    mlog(log_s)

                dev_data_source.epoch_init(shuffle=False)
                while True:
                    batch_data = dev_data_source.next(config.eval_batch_size)
                    if batch_data is None:
                        break

                    ret_data, ret_stat = model.evaluate_step(batch_data)
                    dev_reporter.update_data(ret_stat)

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

                # Decay learning rate
                if config.lr_decay_rate < 1.0:
                    lr_scheduler.step(dev_reporter.get_value("monitor"))
                dev_reporter.clear()

            # finished a step
            n_step += 1
            n_batch += 1

        # Evaluation on test dataset
        model.eval()
        test_data_source.epoch_init(shuffle=False)
        hyps = []
        refs = []
        while True:
            batch_data = test_data_source.next(config.eval_batch_size)
            if batch_data is None:
                break

            ret_data, ret_stat = model.test_step(batch_data)

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

            batch_hyps = ret_data["symbols"].tolist()
            for idx in range(len(batch_hyps)):
                hyp = batch_hyps[idx]
                hyp = tokenizer.convert_ids_to_tokens(
                    ids=hyp,
                    trim_bos=True,
                    trim_from_eos=True,
                    trim_pad=True
                )
                hyp = tokenizer.convert_tokens_to_string(hyp)
                hyp_floor = "A" if batch_floors[idx] == 1 else "B"
                hyps.append((hyp, hyp_floor))

        ref_texts = [text for (text, floor) in refs]
        hyp_texts = [text for (text, floor) in hyps]
        assert len(ref_texts) == len(hyp_texts)

        # Evaluation metrics
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
