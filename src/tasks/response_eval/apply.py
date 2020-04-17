import random
import code
import time
import argparse
import os
import json
import collections

import torch
import numpy as np
import pandas as pd

from model.response_eval.adem import ADEM
from model.response_eval.ruber import RUBER
from model.response_eval.roberta import Roberta
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
    parser.add_argument("--tokenizer", type=str, default="ws", help="[ws, roberta]")
    parser.add_argument("--human_score_names", nargs='+', type=str, default=["grammar", "relevance", "overall"])
    parser.add_argument("--target_score_name", type=str, default="overall")
    parser.add_argument("--metric_type", type=str, default="hybrid", help="metric type for RUBER/ADEM")

    # model - numbers
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--history_len", type=int, default=5, help="number of history sentences")
    parser.add_argument("--word_embedding_dim", type=int, default=200)
    parser.add_argument("--attr_embedding_dim", type=int, default=30)
    parser.add_argument("--sent_encoder_hidden_dim", type=int, default=500)
    parser.add_argument("--n_sent_encoder_layers", type=int, default=2)
    parser.add_argument("--dial_encoder_hidden_dim", type=int, default=500)
    parser.add_argument("--n_dial_encoder_layers", type=int, default=2)

    # other
    parser.add_argument("--seed", type=int, default=42, help="random initialization seed")
    parser.add_argument("--max_uttr_len", type=int, default=40, help="max utterance length for truncation")
    parser.add_argument("--eval_batch_size", type=int, default=60, help="batch size for evaluation")

    # management
    parser.add_argument("--data_scale", type=float, default=1.0, help="percentage of training data to be used")
    parser.add_argument("--adem_pretrained_model_path", type=str, help="path to pretrained model used for training ADEM")
    parser.add_argument("--model_path", type=str, help="path to model")
    parser.add_argument("--output_file_path", type=str, help="path to save outputs")
    parser.add_argument("--output_model_name", type=str, required=True, help="model name displayed in saved outputs")
    parser.add_argument("--corpus", type=str, default="dd", help="[dd, personachat]")
    config = parser.parse_args()

    # Load corpus config
    if config.corpus == "dd":
        from corpora.dd.config import Config
    elif config.corpus == "personachat":
        from corpora.personachat.config import Config
    corpus_config = Config(task="response_eval")

    # Merge parse args with corpus config
    # priority: parse args > corpus config
    corpus_config_dict = {}
    for k, v in corpus_config.__dict__.items():
        if not k.startswith("__") and k not in config.__dict__:
            corpus_config_dict[k] = v
    config.__dict__.update(corpus_config_dict)

    # Set random seeds
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Tokenizers
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

    # Data loaders
    with open(config.dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    data_source = DataSourceSupervised(
        data=dataset["test"],
        config=config,
        tokenizer=tokenizer,
    )
    print(data_source.statistics)

    # metrics calculator
    metrics = CorrelationMetrics()

    # Build model
    if config.model == "adem":
        Model = ADEM
    elif config.model == "ruber":
        Model = RUBER
    elif config.model == "roberta":
        Model = Roberta
    model = Model(config, tokenizer)

    # Model adaption
    if torch.cuda.is_available():
        model = model.cuda()
    if config.model == "adem":
        # NOTE: ADEM's PCA parameters should be the same as in the phase of supervised training
        model.load_model(config.adem_pretrained_model_path)
        train_data_source = DataSourceSupervised(
            config=config,
            tokenizer=tokenizer,
            dataset="train"
        )
        model.estimate_pca_parameters(train_data_source)
        model.estimate_scaling_constants(train_data_source)
    if config.model_path:
        model.load_model(config.model_path)

    # JSON for storing outputs
    if config.output_file_path:
        if os.path.exists(config.output_file_path):
            with open(config.output_file_path, encoding="utf-8") as f:
                result_dict = json.load(f)
        else:
            result_dict = collections.defaultdict(dict)

    # Evaluation on test dataset
    model.eval()

    data_source.epoch_init(shuffle=False)
    score_df = pd.DataFrame()
    while True:
        batch_data = data_source.next(config.eval_batch_size)
        if batch_data is None:
            break

        ret_data, ret_stat = model.supervised_test_step(batch_data)

        hyp_scores = ret_data["scores"].tolist()
        ref_scores = batch_data["Y_score"].tolist()
        response_metas = batch_data["Y_meta_dict"]

        for batch_idx in range(len(hyp_scores)):
            hyp_score = hyp_scores[batch_idx]
            response_meta = response_metas[batch_idx]
            model_name = response_meta["model_name"]
            if model_name in ["ground-truth", "negative-sample"]:
                decode_method = "human"
            else:
                model_name, decode_method = model_name.split(" ")
            item_dict = {
                "model": model_name,
                "decode_method": decode_method,
                "model_decode_method": f"{model_name} {decode_method}",
                "hyp_score": hyp_score,
            }
            for score_idx, score_name in enumerate(config.human_score_names):
                ref_score = ref_scores[batch_idx][score_idx]
                item_dict.update({
                    score_name: ref_score
                })
            score_df = score_df.append(item_dict, ignore_index=True)

        if config.output_file_path:
            Y_meta_dict = batch_data["Y_meta_dict"]
            for hyp_idx, y_meta_dict in enumerate(Y_meta_dict):
                pass
                hyp_score = ret_data["scores"][hyp_idx].item()
                response_id = y_meta_dict["response_id"]

                result_dict[str(response_id)][config.output_model_name] = hyp_score

    log_s = "Correlations of each system:\n"
    for sys_name in score_df["model"].unique().tolist():
        sys_score_df = score_df.loc[score_df["model"] == sys_name]
        log_s += f"{sys_name} ({len(sys_score_df)} examples)\n"
        for score_name in config.human_score_names:
            data1 = sys_score_df["hyp_score"].tolist()
            data2 = sys_score_df[score_name].tolist()
            pearson_r, pearson_p = metrics.pearson_cor(data1, data2)
            spearman_r, spearman_p = metrics.spearman_cor(data1, data2)

            log_s += f"\t{score_name}:\n"
            log_s += f"\t  pearson r: {pearson_r:.2f} (p = {pearson_p:.2g})\n"
            log_s += f"\t  spearman r: {spearman_r:.2f}, (p = {spearman_p:.2g})\n"

            hyp_mean, hyp_std = np.mean(data1), np.std(data1)
            ref_mean, ref_std = np.mean(data2), np.std(data2)
            log_s += f"\t  hyp mean: {hyp_mean:.2f}, hyp std: {hyp_std:.2f}\n"
            log_s += f"\t  ref mean: {ref_mean:.2f}, ref std: {ref_std:.2f}\n"

    print(log_s)

    log_s = "Correlations of all systems excluding ground-truth:\n"
    non_gt_score_df = score_df.loc[score_df["model"] != "ground-truth"]
    for score_name in config.human_score_names:
        data1 = non_gt_score_df["hyp_score"].tolist()
        data2 = non_gt_score_df[score_name].tolist()
        pearson_r, pearson_p = metrics.pearson_cor(data1, data2)
        spearman_r, spearman_p = metrics.spearman_cor(data1, data2)

        log_s += f"{score_name}:\n"
        log_s += f"  pearson r: {pearson_r:.2f} (p = {pearson_p:.2g})\n"
        log_s += f"  spearman r: {spearman_r:.2f}, (p = {spearman_p:.2g})\n"

        hyp_mean, hyp_std = np.mean(data1), np.std(data1)
        ref_mean, ref_std = np.mean(data2), np.std(data2)
        log_s += f"  hyp mean: {hyp_mean:.2f}, hyp std: {hyp_std:.2f}\n"
        log_s += f"  ref mean: {ref_mean:.2f}, ref std: {ref_std:.2f}\n"
    print(log_s)

    log_s = "Correlations of all systems:\n"
    for score_name in config.human_score_names:
        data1 = score_df["hyp_score"].tolist()
        data2 = score_df[score_name].tolist()
        pearson_r, pearson_p = metrics.pearson_cor(data1, data2)
        spearman_r, spearman_p = metrics.spearman_cor(data1, data2)

        log_s += f"{score_name}:\n"
        log_s += f"  pearson r: {pearson_r:.2f} (p = {pearson_p:.2g})\n"
        log_s += f"  spearman r: {spearman_r:.2f}, (p = {spearman_p:.2g})\n"

        hyp_mean, hyp_std = np.mean(data1), np.std(data1)
        ref_mean, ref_std = np.mean(data2), np.std(data2)
        log_s += f"  hyp mean: {hyp_mean:.2f}, hyp std: {hyp_std:.2f}\n"
        log_s += f"  ref mean: {ref_mean:.2f}, ref std: {ref_std:.2f}\n"
    print(log_s)

    if config.output_file_path:
        with open(config.output_file_path, "w+", encoding="utf-8") as f:
            json.dump(result_dict, f)
