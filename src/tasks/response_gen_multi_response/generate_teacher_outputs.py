import json
import random
import code
import time
import sys
import math
import argparse
import os
from collections import defaultdict

from tqdm import tqdm
import torch
import numpy as np

from model.response_gen_multi_response.gpt2 import GPT2
from tokenization.gpt2_tokenizer import ModGPT2Tokenizer
from tasks.response_gen_multi_response.data_source import DataSource


def str2bool(v):
    return v.lower() in ('true', '1', "True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model - architecture
    parser.add_argument("--model", type=str, default="gpt2", help="[gpt2]")
    parser.add_argument("--model_size", type=str, default="medium", help="[small, medium, distilgpt2], model size for GPT2")
    parser.add_argument("--floor_encoder", type=str, default="rel", help="floor encoder type in [none, rel, abs]")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="[gpt2]")

    # model - numbers
    parser.add_argument("--history_len", type=int, default=5, help="number of history sentences")

    # inference
    parser.add_argument("--seed", type=int, help="(optional) random initialization seed")
    parser.add_argument("--max_uttr_len", type=int, default=40, help="max utterance length for trauncation")
    parser.add_argument("--eval_batch_size", type=int, default=50, help="batch size for evaluation")
    parser.add_argument("--decode_max_len", type=int, default=20, help="max utterance length for decoding")
    parser.add_argument("--gen_type", type=str, default="top", help="[greedy, sample, top, mmi_antilm]")
    parser.add_argument("--temp", type=float, default=1.0, help="temperature for decoding")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--n_samples", type=int, default=5)

    # management
    parser.add_argument("--model_path", help="path to model")
    parser.add_argument("--output_path", required=True, help="path to augmented dataset")
    parser.add_argument("--corpus", type=str, default="dd", help="[dd]")
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
    GEN_METHOD = f"{config.gen_type}_temp{config.temp}_k{config.top_k}_p{config.top_p}"
    MODEL_NAME = f"{MODEL_NAME}.{GEN_METHOD}"

    # set random seeds
    if config.seed:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)

    # tokenizers
    special_token_dict = {
        "speaker1_token": "<speaker1>",
        "speaker2_token": "<speaker2>"
    }
    if config.tokenizer == "gpt2":
        tokenizer = ModGPT2Tokenizer(
            model_size=config.model_size,
            special_token_dict=special_token_dict
        )

    # data loaders
    with open(config.dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    print("----- Loading train data -----")
    train_data_source = DataSource(
        data=dataset["train"],
        config=config,
        tokenizer=tokenizer,
    )
    print(str(train_data_source.statistics))

    # build model
    if config.model == "gpt2":
        Model = GPT2
    model = Model(config, tokenizer)
    print(f"number of parameters: {sum([param.nelement() for param in model.parameters()])}")

    # model adaption
    if torch.cuda.is_available():
        print("----- Using GPU -----")
        model = model.cuda()
    if config.model_path:
        model.load_model(config.model_path)
        print("----- Model loaded -----")
        print("model path: {}".format(config.model_path))

    # log hyper parameters
    start_time = time.time()
    print("----- Hyper-parameters -----")
    for k, v in sorted(dict(config.__dict__).items()):
        print("{}: {}".format(k, v))

    # load clean dataset
    with open(config.dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    # create hyps incrementally
    model.eval()
    for sample_idx in range(config.n_samples):
        train_data_source.epoch_init(shuffle=False)
        id2hyps = defaultdict(list)
        for batch_idx in tqdm(range(math.ceil(len(train_data_source)/config.eval_batch_size)), desc=f"iteration {sample_idx+1}/{config.n_samples}"):
            batch_data = train_data_source.next(config.eval_batch_size)
            if batch_data is None:
                break

            ret_data, ret_stat = model.test_step(batch_data)

            batch_hyps = ret_data["symbols"].tolist()
            batch_hyp_word_logprobs = ret_data["symbol_logprobs"]
            batch_y_ids = batch_data["Y_id"]
            for idx in range(len(batch_hyps)):
                hyp = batch_hyps[idx]
                hyp_word_logprobs = batch_hyp_word_logprobs[idx]
                hyp_tokens = tokenizer.convert_ids_to_tokens(
                    ids=hyp,
                    trim_bos=True,
                    trim_from_eos=True,
                    trim_pad=True,
                )
                hyp_text = tokenizer.convert_tokens_to_string(hyp_tokens)
                hyp_len = len(hyp_tokens)
                hyp_word_logprobs = hyp_word_logprobs[:hyp_len+1]
                hyp_sent_logprob = hyp_word_logprobs.sum()
                hyp_sent_prob = hyp_sent_logprob.exp().item()

                hyp_id = batch_y_ids[idx]
                id2hyps[hyp_id].append((hyp_text, hyp_sent_prob))

            # break


        for sess in dataset["train"]:
            for uttr in sess["utterances"]:
                sent_id = uttr["utterance_meta"]["sentence_id"]
                if sent_id in id2hyps.keys():
                    hyps = id2hyps[sent_id]
                    if "model2hypothesis" not in uttr["utterance_meta"]:
                        uttr["utterance_meta"]["model2hypothesis"] = {}
                    if MODEL_NAME not in uttr["utterance_meta"]["model2hypothesis"]:
                        model_id = len(uttr["utterance_meta"]["model2hypothesis"])
                        uttr["utterance_meta"]["model2hypothesis"][MODEL_NAME] = []
                    else:
                        hyp0_sent_id = uttr["utterance_meta"]["model2hypothesis"][MODEL_NAME][0]["utterance_meta"]["sentence_id"]
                        model_id_from = hyp0_sent_id.find("model") + len("model")
                        model_id_to = hyp0_sent_id.find("hyp") - 1
                        model_id = hyp0_sent_id[model_id_from:model_id_to]

                    for hyp, hyp_prob in hyps:
                        hyp_id = len(uttr["utterance_meta"]["model2hypothesis"][MODEL_NAME])
                        hyp_dict = {
                            "text": hyp,
                            "utterance_meta": {
                                "sentence_id": f"{sent_id}_model{model_id}_hyp{hyp_id}",
                                "sentence_prob": hyp_prob
                            }
                        }
                        uttr["utterance_meta"]["model2hypothesis"][MODEL_NAME].append(hyp_dict)

        with open(config.output_path, "w+", encoding="utf-8") as f:
            json.dump(dataset, f)

    # code.interact(local=locals())
