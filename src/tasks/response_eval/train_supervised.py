import json
import random
import code
import time
import argparse
import os
import collections

import numpy as np
import torch
import torch.optim as optim

from model.response_eval.adem import ADEM
from model.response_eval.ruber import RUBER
from model.response_eval.roberta import Roberta
from utils.helpers import StatisticsReporter
from utils.statistics import CorrelationMetrics
from tokenization.whitespace_tokenizer import WhiteSpaceTokenizer
from tokenization.roberta_tokenizer import ModRobertaTokenizer
from tasks.response_eval.data_source_supervised import DataSourceSupervised


def str2bool(v):
    return v.lower() in ('true', '1', "True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model - architecture
    parser.add_argument("--model", type=str, default="adem", help="[adem, ruber, roberta]")
    parser.add_argument("--model_size", type=str, default=None, help="[base, large, large-mnli] for Roberta")
    parser.add_argument("--rnn_type", type=str, default="gru", help="[gru, lstm]")
    parser.add_argument("--floor_encoder", type=str, default="none", help="floor encoder type in [none, rel, abs]")
    parser.add_argument("--use_attention", type=str2bool, default=False, help="use attention for decoder")
    parser.add_argument("--tie_weights", type=str2bool, default=True, help="tie weights for decoder")
    parser.add_argument("--tokenizer", type=str, default="ws", help="[ws, gpt2, bert, roberta]")
    parser.add_argument("--human_score_names", nargs='+', type=str, default=["overall"])
    parser.add_argument("--target_score_name", type=str, default="overall")
    parser.add_argument("--metric_type", type=str, default="hybrid", help="[hybrid, ref, unref] for RUBER/ADEM")

    # model - numbers
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--history_len", type=int, default=5, help="number of history sentences")
    parser.add_argument("--word_embedding_dim", type=int, default=200)
    parser.add_argument("--attr_embedding_dim", type=int, default=30)
    parser.add_argument("--sent_encoder_hidden_dim", type=int, default=500)
    parser.add_argument("--n_sent_encoder_layers", type=int, default=2)
    parser.add_argument("--dial_encoder_hidden_dim", type=int, default=500)
    parser.add_argument("--n_dial_encoder_layers", type=int, default=2)

    # training
    parser.add_argument("--seed", type=int, default=42, help="random initialization seed")
    parser.add_argument("--max_uttr_len", type=int, default=40, help="max utterance length for truncation")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs for training")
    parser.add_argument("--use_pretrained_word_embedding", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=30, help="batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=60, help="batch size for evaluation")
    parser.add_argument("--data_scale", type=float, default=1.0, help="percentage of training data to be used")

    # optimizer
    parser.add_argument("--l2_penalty", type=float, default=0.0, help="l2 penalty")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument("--init_lr", type=float, default=1e-3, help="init learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="minimum learning rate for early stopping")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1)
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="gradient clipping")

    # management
    parser.add_argument("--model_path", help="path to model")
    parser.add_argument("--corpus", type=str, default="dd", help="[dd, personachat]")
    parser.add_argument("--enable_log", type=str2bool, default=False)
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--check_loss_after_n_step", type=int, default=100)
    parser.add_argument("--validate_after_n_step", type=int, default=100)
    parser.add_argument("--filename_note", type=str, help="take a note in saved files' names")
    config = parser.parse_args()

    # load corpus config
    if config.corpus == "dd":
        from corpora.dd.config import Config
    elif config.corpus == "personachat":
        from corpora.personachat.config import Config
    corpus_config = Config(task="response_eval")

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
    if config.model_size:
        MODEL_NAME += "_{}".format(config.model_size)
    LOG_FILE_NAME = "{}.floor_{}.seed_{}.{}.{}supervised_by_{}".format(
        MODEL_NAME,
        config.floor_encoder,
        config.seed,
        time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),
        "semi." if config.model_path else "",
        config.target_score_name,
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
    elif config.tokenizer == "roberta":
        tokenizer = ModRobertaTokenizer(
            model_size=config.model_size,
            special_token_dict=special_token_dict
        )

    # data loaders & number reporters
    trn_reporter = StatisticsReporter()
    dev_reporter = StatisticsReporter()
    with open(config.dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    mlog("----- Loading training data -----")
    training_data = dataset["train"]
    scaled_n_training_data = int(len(training_data)*config.data_scale)
    scaled_training_data = training_data[:scaled_n_training_data]
    train_data_source = DataSourceSupervised(
        data=scaled_training_data,
        config=config,
        tokenizer=tokenizer
    )
    mlog(str(train_data_source.statistics))
    mlog("----- Loading dev data -----")
    dev_data_source = DataSourceSupervised(
        data=dataset["dev"],
        config=config,
        tokenizer=tokenizer
    )
    mlog(str(dev_data_source.statistics))
    mlog("----- Loading test data -----")
    test_data_source = DataSourceSupervised(
        data=dataset["test"],
        config=config,
        tokenizer=tokenizer
    )
    mlog(str(test_data_source.statistics))
    del dataset

    # metrics calculator
    metrics = CorrelationMetrics()

    # build model
    if config.model == "adem":
        Model = ADEM
    elif config.model == "ruber":
        Model = RUBER
    elif config.model == "roberta":
        Model = Roberta
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
    if config.model in ["roberta"]:
        config.l2_penalty = 0.01  # follow the original papers
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
    if config.model == "adem":
        model.estimate_pca_parameters(train_data_source)
        model.estimate_scaling_constants(train_data_source)

    n_step = 0
    loss_history = []
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
            ret_data, ret_stat = model.supervised_train_step(batch_data)

            # backward
            loss = ret_data["loss"]
            loss.backward()
            if config.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip
                )
            optimizer.step()
            optimizer.zero_grad()

            # update
            trn_reporter.update_data(ret_stat)

            # Session result output
            if n_step > 0 and n_step % config.check_loss_after_n_step == 0:
                log_s = f"{time.time()-start_time:.2f}s Epoch {epoch} batch {n_batch} - "
                log_s += trn_reporter.to_string()
                mlog(log_s)
                trn_reporter.clear()

            # Evaluation on dev dataset
            if n_step > 0 and n_step % config.validate_after_n_step == 0:
                model.eval()

                log_s = f"<Dev> learning rate: {lr}\n"
                mlog(log_s)

                dev_data_source.epoch_init(shuffle=False)
                score_pairs = collections.defaultdict(lambda: {"hyps": [], "refs": []})
                while True:
                    batch_data = dev_data_source.next(config.eval_batch_size)
                    if batch_data is None:
                        break

                    ret_data, ret_stat = model.supervised_test_step(batch_data)
                    hyp_scores = ret_data["scores"]
                    ref_scores = batch_data["Y_score"]

                    for score_idx, score_name in enumerate(config.human_score_names):
                        score_pairs[score_name]["hyps"] += hyp_scores.tolist()
                        score_pairs[score_name]["refs"] += ref_scores[:, score_idx].tolist()

                score_name = "overall"
                data1 = score_pairs[score_name]["hyps"]
                data2 = score_pairs[score_name]["refs"]
                pearson_r, pearson_p = metrics.pearson_cor(data1, data2)
                spearman_r, spearman_p = metrics.spearman_cor(data1, data2)

                log_s = f"\n<Dev> - {time.time()-start_time:.3f}s - "
                log_s += f"pearson {pearson_r:.2g} ({pearson_p:.5g}) "
                log_s += f"spearman {spearman_r:.2g} ({spearman_p:.5g})"
                mlog(log_s)

                # Decay learning rate
                lr_scheduler.step(pearson_r+spearman_r)
                dev_reporter.clear()

            # finished a step
            n_step += 1
            n_batch += 1

        # Evaluation on test dataset
        model.eval()

        test_data_source.epoch_init(shuffle=False)
        score_pairs = collections.defaultdict(lambda: {"hyps": [], "refs": []})
        while True:
            batch_data = test_data_source.next(config.eval_batch_size)
            if batch_data is None:
                break

            ret_data, ret_stat = model.supervised_test_step(batch_data)
            hyp_scores = ret_data["scores"]
            ref_scores = batch_data["Y_score"]

            for score_idx, score_name in enumerate(config.human_score_names):
                score_pairs[score_name]["hyps"] += hyp_scores.tolist()
                score_pairs[score_name]["refs"] += ref_scores[:, score_idx].tolist()

        log_s = "<Test>:\n"
        for score_name in config.human_score_names:
            data1 = score_pairs[score_name]["hyps"]
            data2 = score_pairs[score_name]["refs"]
            pearson_r, pearson_p = metrics.pearson_cor(data1, data2)
            spearman_r, spearman_p = metrics.spearman_cor(data1, data2)

            if score_name == config.target_score_name:
                log_s += f"{score_name} [TARGET]:\n"
            else:
                log_s += f"{score_name}:\n"
            log_s += f"  pearson r: {pearson_r:.2f} (p = {pearson_p:.2g})\n"
            log_s += f"  spearman r: {spearman_r:.2f}, (p = {spearman_p:.2g})\n"
        mlog(log_s)

    # Save model
    if config.save_model:
        if not os.path.exists(f"../data/{config.corpus}/model/{config.task}"):
            os.makedirs(f"../data/{config.corpus}/model/{config.task}")

        torch.save(model.state_dict(), f"../data/{config.corpus}/model/{config.task}/{LOG_FILE_NAME}.model.pt")
        mlog(f"model saved to data/{config.corpus}/model/{config.task}/{LOG_FILE_NAME}.model.pt")
