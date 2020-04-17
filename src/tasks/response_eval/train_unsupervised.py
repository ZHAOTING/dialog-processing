import random
import code
import time
import argparse
import os
import collections
import json
import copy

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
from tasks.response_eval.data_source_unsupervised import DataSourceUnsupervised


def str2bool(v):
    return v.lower() in ('true', '1', "True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model - architecture
    parser.add_argument("--model", type=str, default="adem", help="[adem, ruber, roberta]")
    parser.add_argument("--model_size", type=str, default=None, help="[base, large, large-mnli] for Roberta")
    parser.add_argument("--rnn_type", type=str, default="gru", help="[gru, lstm]")
    parser.add_argument("--floor_encoder", type=str, default="none", help="floor encoder type in [none, rel, abs]")
    parser.add_argument("--tie_weights", type=str2bool, default=True, help="tie weights for decoder")
    parser.add_argument("--tokenizer", type=str, default="ws", help="[ws, roberta]")
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
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs for training")
    parser.add_argument("--use_pretrained_word_embedding", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=30, help="batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=60, help="batch size for evaluation")

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
    parser.add_argument("--validate_after_n_step", type=int, default=1000)
    parser.add_argument("--sample_after_n_step", type=int, default=1000)
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

    # use annotated partial corpus for supervised test
    supervised_config = copy.deepcopy(config)
    # use unannotated full corpus for unsupervised training
    gen_corpus_config = Config(task="response_gen")
    config.dataset_path = gen_corpus_config.dataset_path

    # define logger
    MODEL_NAME = config.model
    if config.model_size:
        MODEL_NAME += "_{}".format(config.model_size)
    LOG_FILE_NAME = "{}.floor_{}.seed_{}.{}".format(
        MODEL_NAME,
        config.floor_encoder,
        config.seed,
        time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    )
    if config.filename_note:
        LOG_FILE_NAME += f".{config.filename_note}"

    def mlog(s):
        if config.enable_log:
            if not os.path.exists(f"../log/{config.corpus}/{config.task}"):
                os.makedirs(f"../log/{config.corpus}/{config.task}")

            with open(f"../log/{config.corpus}/{config.task}/{LOG_FILE_NAME}.unsupervised.log", "a+", encoding="utf-8") as log_f:
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
    mlog("----- Loading unsupervised training data -----")
    train_data_source = DataSourceUnsupervised(
        data=dataset["train"],
        config=config,
        tokenizer=tokenizer
    )
    mlog(str(train_data_source.statistics))
    mlog("----- Loading unsupervised dev data -----")
    dev_data_source = DataSourceUnsupervised(
        data=dataset["dev"],
        config=config,
        tokenizer=tokenizer
    )
    mlog(str(dev_data_source.statistics))
    mlog("----- Loading unsupervised test data -----")
    test_data_source = DataSourceUnsupervised(
        data=dataset["test"],
        config=config,
        tokenizer=tokenizer
    )
    mlog(str(test_data_source.statistics))
    del dataset

    with open(supervised_config.dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    mlog("----- Loading supervised test data -----")
    scoring_test_data_source = DataSourceSupervised(
        data=dataset["test"],
        config=supervised_config,
        tokenizer=tokenizer,
    )
    mlog(str(scoring_test_data_source.statistics))
    mlog("----- Loading supervised all data -----")
    scoring_all_data_source = DataSourceSupervised(
        data=dataset["train"]+dataset["dev"]+dataset["test"],
        config=supervised_config,
        tokenizer=tokenizer,
    )
    mlog(str(scoring_all_data_source.statistics))
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
        print("----- Using GPU -----")
        model = model.cuda()
    if config.model_path:
        model.load_model(config.model_path)
        print("----- Model loaded -----")
        print("model path: {}".format(config.model_path))

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
            if config.model in ["ruber"]:
                batch_data = train_data_source.next(config.batch_size, return_paired_Y=True)
            else:
                batch_data = train_data_source.next(config.batch_size)
            if batch_data is None:
                break

            # forward
            model.train()
            ret_data, ret_stat = model.unsupervised_train_step(batch_data)

            # update
            loss = ret_data["loss"]
            loss.backward()
            if config.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip
                )
            optimizer.step()
            optimizer.zero_grad()
            trn_reporter.update_data(ret_stat)

            # Session result output
            if n_step > 0 and n_step % config.check_loss_after_n_step == 0:
                log_s = f"{time.time()-start_time:.2f}s Epoch {epoch} batch {n_batch} - "
                log_s += trn_reporter.to_string()
                mlog(log_s)
                trn_reporter.clear()

            # Sampling from test dataset
            if n_step > 0 and n_step % config.sample_after_n_step == 0:
                model.eval()

                log_s = "<Test> - Samples for is-next prediction:"
                mlog(log_s)
                test_data_source.epoch_init(shuffle=True)
                for sample_idx in range(5):
                    batch_data = test_data_source.next(1)

                    ret_data, ret_stat = model.unsupervised_test_step(batch_data)

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
                            trim_pad=True,
                        )
                        floor = "A" if floor == 1 else "B"
                        log_s += "  {}: {}\n".format(
                            floor,
                            tokenizer.convert_tokens_to_string(uttr)
                        )
                    mlog(log_s)

                    log_s = "response:\n"
                    floor = batch_data["Y_floor"][0].item()
                    floor = "A" if floor == 1 else "B"
                    uttr = batch_data["Y"][0].tolist()
                    uttr = tokenizer.convert_ids_to_tokens(
                        ids=uttr,
                        trim_bos=True,
                        trim_from_eos=True,
                        trim_pad=True,
                    )
                    log_s += "  {}: {}\n".format(
                        floor,
                        tokenizer.convert_tokens_to_string(uttr)
                    )
                    mlog(log_s)

                    is_next_label = (batch_data["Y_is_next"].long()).item()
                    is_next_pred = ret_data["probs"].item()
                    log_s = f"True label: {is_next_label}\n"
                    log_s += f"Prediction: {is_next_pred:.4f}\n"
                    log_s += "="*30
                    mlog(log_s)

            # Evaluation on dev dataset
            if n_step > 0 and n_step % config.validate_after_n_step == 0:
                model.eval()

                log_s = f"<Dev> learning rate: {lr}\n"
                mlog(log_s)

                # unsupervised development
                dev_data_source.epoch_init(shuffle=False)
                ref_labels, hyp_labels = [], []
                while True:
                    batch_data = dev_data_source.next(config.eval_batch_size)
                    if batch_data is None:
                        break

                    ret_data, ret_stat = model.unsupervised_evaluate_step(batch_data)
                    dev_reporter.update_data(ret_stat)

                    ref_labels += batch_data["Y_is_next"].long().tolist()
                    hyp_labels += torch.round(ret_data["probs"]).long().tolist()

                ref_labels = np.array(ref_labels)
                hyp_labels = np.array(hyp_labels)
                accuracy = (ref_labels == hyp_labels).sum()/len(ref_labels)

                log_s = f"\n<Is-Next Dev> - {time.time()-start_time:.3f}s - accuracy: {accuracy:.3f} - "
                log_s += dev_reporter.to_string()
                mlog(log_s)

                # supervised
                scoring_all_data_source.epoch_init(shuffle=False)
                hyp_scores, ref_scores = [], []
                while True:
                    batch_data = scoring_all_data_source.next(config.eval_batch_size)
                    if batch_data is None:
                        break

                    ret_data, ret_stat = model.unsupervised_test_step(batch_data)
                    hyp_score = (ret_data["probs"]*4+1).tolist()  # scaling from [0, 1] to [1, 5]
                    hyp_scores += hyp_score
                    ref_scores += batch_data["Y_score"].tolist()

                log_s = f"\n<Scoring Dev> - {time.time()-start_time:.3f}s - [Supervised All Data] - Correlation between prediction and\n"
                for score_idx, score_name in enumerate(config.human_score_names):
                    if score_name == "fact":
                        continue
                    data1 = hyp_scores
                    data2 = np.array(ref_scores)[:, score_idx].tolist()
                    pearson_r, pearson_p = metrics.pearson_cor(data1, data2)
                    spearman_r, spearman_p = metrics.spearman_cor(data1, data2)

                    log_s += f"{score_name} scores:\n"
                    log_s += f"  pearson r: {pearson_r:.2f} (p = {pearson_p:.2g})\n"
                    log_s += f"  spearman r: {spearman_r:.2f}, (p = {spearman_p:.2g})\n"
                mlog(log_s)

                # Save model if it has better monitor measurement
                if config.save_model:
                    if not os.path.exists(f"../data/{config.corpus}/model/{config.task}"):
                        os.makedirs(f"../data/{config.corpus}/model/{config.task}")

                    torch.save(model.state_dict(), f"../data/{config.corpus}/model/{config.task}/{LOG_FILE_NAME}.unsupervised.model.pt")
                    mlog(f"model saved to data/{config.corpus}/model/{config.task}/{LOG_FILE_NAME}.unsupervised.model.pt")

                    if torch.cuda.is_available():
                        model = model.cuda()

                # Decay learning rate
                lr_scheduler.step(dev_reporter.get_value("monitor"))
                dev_reporter.clear()

            # finished a step
            n_step += 1
            n_batch += 1

        # Evaluation on test dataset
        model.eval()

        # accuracy on unsupervised test data
        test_data_source.epoch_init(shuffle=False)
        hyp_labels, ref_labels = [], []
        while True:
            batch_data = test_data_source.next(config.eval_batch_size)
            if batch_data is None:
                break

            ret_data, ret_stat = model.unsupervised_test_step(batch_data)
            ref_labels += batch_data["Y_is_next"].long().tolist()
            hyp_labels += torch.round(ret_data["probs"]).long().tolist()

        ref_labels = np.array(ref_labels)
        hyp_labels = np.array(hyp_labels)
        accuracy = (ref_labels == hyp_labels).sum()/len(ref_labels)

        log_s = f"<Is-Next Test> - [Unsupervised Test Data] accuracy: {accuracy:.3f}\n"
        mlog(log_s)

        # correlation with scores on supervised test dataset
        scoring_test_data_source.epoch_init(shuffle=False)
        hyp_scores, ref_scores = [], []
        while True:
            batch_data = scoring_test_data_source.next(config.eval_batch_size)
            if batch_data is None:
                break

            ret_data, ret_stat = model.unsupervised_test_step(batch_data)
            hyp_score = (ret_data["probs"]*4+1).tolist()  # scaling from [0, 1] to [1, 5]
            hyp_scores += hyp_score
            ref_scores += batch_data["Y_score"].tolist()

        log_s = "<Scoring Test> - [Supervised Test Data] Correlation between prediction and\n"
        for score_idx, score_name in enumerate(config.human_score_names):
            if score_name == "fact":
                continue
            data1 = hyp_scores
            data2 = np.array(ref_scores)[:, score_idx].tolist()
            pearson_r, pearson_p = metrics.pearson_cor(data1, data2)
            spearman_r, spearman_p = metrics.spearman_cor(data1, data2)

            log_s += f"{score_name} scores:\n"
            log_s += f"  pearson r: {pearson_r:.2f} (p = {pearson_p:.2g})\n"
            log_s += f"  spearman r: {spearman_r:.2f}, (p = {spearman_p:.2g})\n"
        mlog(log_s)

        # correlation with scores on supervised all dataset
        scoring_all_data_source.epoch_init(shuffle=False)
        hyp_scores, ref_scores = [], []
        while True:
            batch_data = scoring_all_data_source.next(config.eval_batch_size)
            if batch_data is None:
                break

            ret_data, ret_stat = model.unsupervised_test_step(batch_data)
            hyp_score = (ret_data["probs"]*4+1).tolist()  # scaling from [0, 1] to [1, 5]
            hyp_scores += hyp_score
            ref_scores += batch_data["Y_score"].tolist()

        log_s = "<Scoring Test> - [Supervised All Data] Correlation between prediction and\n"
        for score_idx, score_name in enumerate(config.human_score_names):
            if score_name == "fact":
                continue
            data1 = hyp_scores
            data2 = np.array(ref_scores)[:, score_idx].tolist()
            pearson_r, pearson_p = metrics.pearson_cor(data1, data2)
            spearman_r, spearman_p = metrics.spearman_cor(data1, data2)

            log_s += f"{score_name} scores:\n"
            log_s += f"  pearson r: {pearson_r:.2f} (p = {pearson_p:.2g})\n"
            log_s += f"  spearman r: {spearman_r:.2f}, (p = {spearman_p:.2g})\n"
        mlog(log_s)

