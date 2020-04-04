import code
import argparse
import json
import os
import random
import collections

import numpy as np

from .config import Config

config = Config(task="response_eval")


def extract_dialogs_from_amt_results(amt_results):
    dialogs = []
    response_id = 0
    for dialog_id, _dialog in amt_results.items():
        _context = _dialog["context"]
        _responses = _dialog["responses"]
        _reference = _dialog["reference"]
        ref_floor, ref_text = _reference

        context = [
            {
                "floor": uttr[0],
                "text": uttr[1],
            } for uttr in _context
        ]
        for model_name, _response in _responses.items():
            if "scores" not in _response:
                continue

            dialog = {
                "utterances": [],
                "dialog_meta": {
                    "dialog_id": dialog_id
                }
            }

            scores = extract_scores_from_worker_results(_response["scores"])
            response = {
                "floor": ref_floor,
                "text": _response["uttr"],
                "utterance_meta": {
                    "human_scores": scores,
                    "reference_text": ref_text,
                    "model_name": model_name,
                    "response_id": response_id,
                }
            }
            response_id += 1

            for uttr in context:
                dialog["utterances"].append(uttr)
            dialog["utterances"].append(response)
            dialogs.append(dialog)
    return dialogs


def extract_scores_from_worker_results(worker_results):
    score_dict = {}
    for score_name in config.human_score_names:
        scores = []
        for worker_scores in worker_results.values():
            if worker_scores[f"{score_name}_is_outlier"]:
                continue
            score = worker_scores[score_name]
            scores.append(score)
        assert len(scores) > 0
        averaged_score = np.mean(scores)
        score_dict[score_name] = averaged_score
    return score_dict


def train_dev_test_split(dialogs):
    n_dial = len(dialogs)

    random.shuffle(dialogs)

    dataset = {
        "train": dialogs[:int(n_dial*0.8)],
        "dev": dialogs[int(n_dial*0.8):int(n_dial*0.9)],
        "test": dialogs[int(n_dial*0.9):]
    }

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mturk_data_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(config.task_data_dir):
        os.makedirs(config.task_data_dir)

    random.seed(args.seed)

    with open(args.mturk_data_path, encoding="utf-8") as f:
        mturk_data = json.load(f)

    print("Extracting dialogs from AMT results...")
    dialogs = extract_dialogs_from_amt_results(mturk_data)
    print(f"Got {len(dialogs)} dialogs.")
    print("Splitting dataset...")
    dataset = train_dev_test_split(dialogs)

    print("Saving dataset to file...")
    with open(config.dataset_path, "w+", encoding="utf-8") as f:
        json.dump(dataset, f)

