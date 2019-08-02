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

from model.da_recog.hre import HRE
from model.da_recog.hre_sep_uttr_enc import HRESepUttrEnc
from utils.helpers import metric_is_improving, StatisticsReporter
from utils.metrics import ClassificationMetrics
from tokenization.basic_tokenizer import BasicTokenizer
from .data_source import DataSource

def str2bool(v):
    return v.lower() in ('true', '1', "True")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model - architecture
    parser.add_argument("--model", type=str, default="hre", help="[hre, hre_sep_uttr_enc]")
    parser.add_argument("--rnn_type", type=str, default="lstm", help="[gru, lstm]")
    parser.add_argument("--floor_encoder", type=str, default="none", help="floor encoder type in [none, rel, abs]")

    # model - numbers
    parser.add_argument("--history_len", type=int, default=5, help="number of history sentences")
    parser.add_argument("--word_embedding_dim", type=int, default=200)
    parser.add_argument("--attr_embedding_dim", type=int, default=30)
    parser.add_argument("--sent_encoder_hidden_dim", type=int, default=500)
    parser.add_argument("--n_sent_encoder_layers", type=int, default=2)
    parser.add_argument("--dial_encoder_hidden_dim", type=int, default=500)
    parser.add_argument("--n_dial_encoder_layers", type=int, default=2)

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

    # management
    parser.add_argument("--model_path", help="path to model")
    parser.add_argument("--corpus", type=str, default="swda", help="[swda]")
    parser.add_argument("--enable_log", type=str2bool, default=False)
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--check_loss_after_n_step", type=int, default=100)
    parser.add_argument("--validate_after_n_step", type=int, default=1000)
    config = parser.parse_args()

    # load corpus config
    if config.corpus == "swda":
        from corpora.swda.config import Config
    corpus_config = Config(task="da_recog")

    ## merge parse args with corpus config
    # priority: parse args > corpus config
    corpus_config_dict = {}
    for k, v in corpus_config.__dict__.items():
        if not k.startswith("__") and k not in config.__dict__:
            corpus_config_dict[k] = v
    config.__dict__.update(corpus_config_dict)
    config.task = "response_gen"

    # set random seeds
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # tokenizers
    tokenizer = BasicTokenizer(config.word_count_path, config.vocab_size)

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
    metrics = ClassificationMetrics(config.dialog_acts)

    # build model
    if config.model == "hre":
        Model = HRE
    elif config.model == "hre_sep_uttr_enc":
        Model = HRESepUttrEnc
    model = Model(config, tokenizer)

    # model adaption
    if torch.cuda.is_available():
        print("----- Using GPU -----")
        model = model.cuda()
    if config.model_path:
        model.load_model(config.model_path)
        print("----- Model loaded -----")
        print(f"model path: {config.model_path}")
    print(str(model))

    # define logger
    MODEL_NAME = Model.__name__
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
            ret_statistics = model.train_step(batch_data, lr=lr)
            trn_reporter.update_data(ret_statistics)
            n_step += 1
            n_batch += 1

            # Check loss
            if n_step > 0 and n_step%config.check_loss_after_n_step == 0:
                log_s = f"{time.time()-start_time:.2f}s Epoch {epoch} batch {n_batch} - "
                log_s += trn_reporter.to_string()
                mlog(log_s)
                trn_reporter.clear()

            # Evaluate on dev dataset
            if n_step > 0 and n_step%config.validate_after_n_step == 0:
                model.eval()

                log_s = f"<Dev> learning rate: {lr}\n"
                mlog(log_s)

                pred_labels, true_labels = [], []
                dev_data_source.epoch_init(shuffle=False)
                while True:
                    batch_data = dev_data_source.next(config.eval_batch_size)
                    if batch_data is None:
                        break

                    ret_outputs, ret_statistics = model.evaluate_step(batch_data)
                    dev_reporter.update_data(ret_statistics)
                    pred_labels += ret_outputs["labels"].tolist()
                    true_labels += batch_data["Y_da"].tolist()

                log_s = f"\n<Dev> - {time.time()-start_time:.3f}s - "
                log_s += dev_reporter.to_string()
                mlog(log_s)
                log_s = "\tClassification report:\n"
                log_s += metrics.classification_report(true_labels, pred_labels)
                log_s += "\n"
                metrics_results = metrics.classification_metrics(true_labels, pred_labels)
                log_s += \
                    f"\tF1 macro:       {100*metrics_results['f1_macro']:.2f}\n" \
                    f"\tF1 micro:       {100*metrics_results['f1_micro']:.2f}\n" \
                    f"\tF1 weighted:    {100*metrics_results['f1_weighted']:.2f}\n" \
                    f"\tAccuracy:       {100*metrics_results['accuracy']:.2f}\n"
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

                # Decay learning rate
                if not metric_is_improving(loss_history):
                    lr = lr*config.lr_decay_rate
        
        # Evaluate on test dataset every epoch
        model.eval()
        pred_labels, true_labels = [], []
        test_data_source.epoch_init(shuffle=False)
        while True:
            batch_data = test_data_source.next(config.eval_batch_size)
            if batch_data is None:
                break

            ret_outputs, ret_statistics = model.evaluate_step(batch_data)
            pred_labels += ret_outputs["labels"].tolist()
            true_labels += batch_data["Y_da"].tolist()

        log_s = f"\n<Test> - {time.time()-start_time:.3f}s -"
        mlog(log_s)
        log_s = "\tClassification report:\n"
        log_s += metrics.classification_report(true_labels, pred_labels)
        log_s += "\n"
        metrics_results = metrics.classification_metrics(true_labels, pred_labels)
        log_s += \
            f"\tF1 macro:       {100*metrics_results['f1_macro']:.2f}\n" \
            f"\tF1 micro:       {100*metrics_results['f1_micro']:.2f}\n" \
            f"\tF1 weighted:    {100*metrics_results['f1_weighted']:.2f}\n" \
            f"\tAccuracy:       {100*metrics_results['accuracy']:.2f}\n"
        mlog(log_s)
