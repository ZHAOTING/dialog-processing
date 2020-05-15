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
from tqdm import tqdm

from model.response_eval.roberta import Roberta
from utils.metrics import SentenceMetrics
from utils.helpers import StatisticsReporter
from utils.config import ConfigFromDict
from tokenization.roberta_tokenizer import ModRobertaTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str2bool(v):
    return v.lower() in ('true', '1', "True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model - architecture
    parser.add_argument("--model_size", type=str, default="large", help="evaluator's model size")
    parser.add_argument("--model_path", type=str, required=True, help="path to evaluator model")
    parser.add_argument("--eval_batch_size", type=int, default=30, help="batch size")
    config = parser.parse_args()

    # tokenizers
    special_token_dict = {
        "speaker1_token": "<speaker1>",
        "speaker2_token": "<speaker2>"
    }
    tokenizer = ModRobertaTokenizer(
        model_size=config.model_size,
        special_token_dict=special_token_dict
    )
    evaluator = Roberta(config, tokenizer)
    if torch.cuda.is_available():
        evaluator = evaluator.cuda()
    evaluator.load_model(config.model_path)
    print(f"Loaded pretrained evaluator at {config.model_path}")
    evaluator.eval()

    # get input data
    ctxs = [
        [
            ["hello there .", "A"],
            ["hi .", "B"],
        ],
        [
            ["it has been a terrible year .", "B"],
            ["any plan after graduation ?", "A"],
        ],
        [
            ["hi, how much is the guitar ?", "A"],
            ["it only takes five thousand bucks .", "B"],
            ["that is a lot !", "A"]
        ]
    ]
    hyps = [
        ["how is it going ?", "A"],
        ["not really .", "B"],
        ["that is funny .", "B"]
    ]

    # compute scores
    scores = []
    idx = 0
    n_batches = math.ceil(len(ctxs)/config.eval_batch_size)
    for batch_idx in tqdm(range(n_batches)):
        batch_ctx = ctxs[batch_idx*config.eval_batch_size:(batch_idx+1)*config.eval_batch_size]
        batch_hyp = hyps[batch_idx*config.eval_batch_size:(batch_idx+1)*config.eval_batch_size]
        batch_scores = evaluator.predict(batch_ctx, batch_hyp)
        scores += batch_scores
    print(scores)
    print(f"avg score: {np.mean(scores)}")


