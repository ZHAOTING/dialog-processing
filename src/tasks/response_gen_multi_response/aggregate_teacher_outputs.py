import json
import random
import code
import time
import sys
import math
import argparse
import os
from collections import defaultdict


def str2bool(v):
    return v.lower() in ('true', '1', "True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # management
    parser.add_argument("--hyp_file_paths", nargs="+", required=True, help="path to hypotheses datasets")
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

    # load hyps
    id2model2hyps = defaultdict(lambda: defaultdict(lambda: list))
    for file_path in config.hyp_file_paths:
        print(f"Loading {file_path}")
        with open(file_path, encoding="utf-8") as f:
            hyp_dataset = json.load(f)

        for sess in hyp_dataset["train"]:
            for uttr in sess["utterances"]:
                sent_id = uttr["utterance_meta"]["sentence_id"]
                if "model2hypothesis" in uttr["utterance_meta"]:
                    _model2hyps = id2model2hyps[sent_id]
                    model2hyps = uttr["utterance_meta"]["model2hypothesis"]

                    for model_name, hyps in model2hyps.items():
                        for hyp in hyps:

                            if model_name not in _model2hyps:
                                _model2hyps[model_name] = []
                            model_id = len(model2hyps) - 1

                            hyp_id = len(_model2hyps[model_name])

                            hyp["utterance_meta"]["sentence_id"] = f"{sent_id}_model{model_id}_hyp{hyp_id}"
                            _model2hyps[model_name].append(hyp)

    # load clean dataset
    print("Writing...")
    with open(f"{config.dataset_path}.original", encoding="utf-8") as f:
        dataset = json.load(f)
    for sess in dataset["train"]:
        for uttr in sess["utterances"]:
            sent_id = uttr["utterance_meta"]["sentence_id"]
            if sent_id in id2model2hyps:
                uttr["utterance_meta"]["model2hypothesis"] = id2model2hyps[sent_id]
    with open(f"{config.dataset_path}.aggregate", "w+", encoding="utf-8") as f:
        json.dump(dataset, f)

    # code.interact(local=locals())
