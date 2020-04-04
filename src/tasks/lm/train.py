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

from model.lm.rnnlm import RNNLM
from utils.helpers import StatisticsReporter
from tokenization.whitespace_tokenizer import WhiteSpaceTokenizer
from tasks.lm.data_source import DataSource


def str2bool(v):
    return v.lower() in ('true', '1', "True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model - architecture
    parser.add_argument("--model", type=str, default="rnnlm", help="[rnnlm]")
    parser.add_argument("--rnn_type", type=str, default="gru", help="[gru, lstm]")
    parser.add_argument("--tie_weights", type=str2bool, default=True, help="tie weights for decoder")
    parser.add_argument("--tokenizer", type=str, default="ws", help="[ws, gpt2]")

    # model - numbers
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--word_embedding_dim", type=int, default=200)
    parser.add_argument("--decoder_hidden_dim", type=int, default=500)
    parser.add_argument("--n_decoder_layers", type=int, default=2)

    # training
    parser.add_argument("--seed", type=int, default=42, help="random initialization seed")
    parser.add_argument("--max_uttr_len", type=int, default=40, help="max utterance length for trauncation")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout probability")
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs for training")
    parser.add_argument("--use_pretrained_word_embedding", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=50, help="batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=100, help="batch size for evaluation")

    # optimizer
    parser.add_argument("--l2_penalty", type=float, default=0.0, help="l2 penalty")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument("--init_lr", type=float, default=1e-3, help="init learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="minimum learning rate for early stopping")
    parser.add_argument("--lr_decay_rate", type=float, default=0.5)
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="gradient clipping")

    # inference
    parser.add_argument("--decode_max_len", type=int, default=40, help="max utterance length for decoding")
    parser.add_argument("--gen_type", type=str, default="greedy", help="[greedy, sample, top]")
    parser.add_argument("--temp", type=float, default=1.0, help="temperature for decoding")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)

    # management
    parser.add_argument("--model_path", help="path to model")
    parser.add_argument("--corpus", type=str, default="dd", help="[dd, cornellmovie, personachat]")
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
    elif config.corpus == "cornellmovie":
        from corpora.cornellmovie.config import Config
    elif config.corpus == "personachat":
        from corpora.personachat.config import Config
    corpus_config = Config(task="lm")

    # merge parse args with corpus config
    # priority: parse args > corpus config
    corpus_config_dict = {}
    for k, v in corpus_config.__dict__.items():
        if not k.startswith("__") and k not in config.__dict__:
            corpus_config_dict[k] = v
    config.__dict__.update(corpus_config_dict)

    # define logger
    MODEL_NAME = config.model
    LOG_FILE_NAME = "{}.seed_{}.{}".format(
        MODEL_NAME,
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
    }
    if config.tokenizer == "ws":
        tokenizer = WhiteSpaceTokenizer(
            word_count_path=config.word_count_path,
            vocab_size=config.vocab_size,
            special_token_dict=special_token_dict
        )

    # data loaders & number reporters
    trn_reporter = StatisticsReporter()
    dev_reporter = StatisticsReporter()
    with open(config.dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    mlog("----- Loading training data -----")
    train_data_source = DataSource(
        data=dataset["train"],
        config=config,
        tokenizer=tokenizer
    )
    mlog(str(train_data_source.statistics))
    mlog("----- Loading dev data -----")
    dev_data_source = DataSource(
        data=dataset["dev"],
        config=config,
        tokenizer=tokenizer
    )
    mlog(str(dev_data_source.statistics))
    del dataset

    # build model
    if config.model == "rnnlm":
        Model = RNNLM
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
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=config.lr_decay_rate,
        patience=2,
    )

    # log hyper parameters
    start_time = time.time()
    mlog("----- Hyper-parameters -----")
    for k, v in sorted(dict(config.__dict__).items()):
        mlog("{}: {}".format(k, v))

    # here we go
    n_step = 0
    for epoch in range(1, config.n_epochs+1):
        lr = list(lr_scheduler.optimizer.param_groups)[0]["lr"]
        if lr <= config.min_lr:
            break

        # Train
        n_batch = 0
        train_data_source.epoch_init(shuffle=True)
        while True:
            batch_data = train_data_source.next(config.batch_size)
            if batch_data is None:
                break

            # forward
            model.train()
            ret_data, ret_stat = model.train_step(batch_data)

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
            trn_reporter.update_data(ret_stat)

            # check loss
            if n_step > 0 and n_step % config.check_loss_after_n_step == 0:
                log_s = f"{time.time()-start_time:.2f}s Epoch {epoch} batch {n_batch} - "
                log_s += trn_reporter.to_string()
                mlog(log_s)
                trn_reporter.clear()

            # Sampling
            if n_step > 0 and n_step % config.sample_after_n_step == 0:
                model.eval()

                log_s = "<Test> - Samples:"
                mlog(log_s)
                for sample_idx in range(5):
                    ret_data, ret_stat = model.sample_step(batch_size=1)

                    log_s = "hyp text:\n"
                    hyp = ret_data["symbols"][0].tolist()
                    hyp = tokenizer.convert_ids_to_tokens(
                        ids=hyp,
                        trim_bos=True,
                        trim_from_eos=True,
                        trim_pad=True,
                    )
                    log_s += "  {}\n".format(
                        tokenizer.convert_tokens_to_string(hyp)
                    )
                    log_s += "="*30
                    mlog(log_s)

            # Evaluation on dev dataset
            if n_step > 0 and n_step % config.validate_after_n_step == 0:
                model.eval()

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
                lr_scheduler.step(dev_reporter.get_value("monitor"))
                dev_reporter.clear()

            # finished a step
            n_step += 1
            n_batch += 1
