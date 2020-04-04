import code
import argparse
import json
import os
import random
import re
import collections
from multiprocessing import Pool
from urllib.request import urlretrieve

from tqdm import tqdm
import spacy

from .config import Config
from utils.helpers import standardize_english_text

config = Config(task="response_gen")
raw_data_dir = config.raw_data_dir
raw_data_file_path = f"{config.raw_data_dir}/personachat.json"


def clean_personachat_text(text):
    text = standardize_english_text(text)
    text = re.sub(r"(\w)n ' (t\W)", r"\1 n'\2", text)
    text = re.sub(r" ' (m|s|re|ve|d|ll)(\W)", r" '\1\2", text)
    return text


def download_data():
    """Download and unpack dialogs"""

    if not os.path.exists(raw_data_file_path):
        print(f'Downloading {config.download_url} to {raw_data_file_path}')
        urlretrieve(config.download_url, raw_data_file_path)
        print(f'Successfully downloaded {raw_data_file_path}')


def build_session(raw_dialog_json):
    personalities = raw_dialog_json["personality"]
    full_history = raw_dialog_json["utterances"][-1]["history"]

    session = {
        "utterances": [],
        "dialog_meta": {
            "speaker2_personality": personalities,
        }
    }
    for uttr in raw_dialog_json["utterances"]:
        speaker1_uttr_text = uttr["history"][-1]
        
        session["utterances"].append({
            "floor": "A",
            "text": speaker1_uttr_text,
            "utterance_meta": {}
        })

        candidates = uttr["candidates"]
        speaker2_uttr_text = candidates[-1]

        session["utterances"].append({
            "floor": "B",
            "text": speaker2_uttr_text,
            "utterance_meta": {
                "candidate_texts": candidates,
            }
        })

    return session


def train_dev_test_split(train_sessions, valid_sessions):
    n_train_sessions = len(train_sessions)

    dataset = {
        "train": train_sessions[:n_train_sessions-1000],
        "dev": train_sessions[n_train_sessions-1000:],
        "test": valid_sessions
    }

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    if not os.path.exists(config.raw_data_dir):
        os.makedirs(config.raw_data_dir)
    if not os.path.exists(config.task_data_dir):
        os.makedirs(config.task_data_dir)

    print("Preparing Personachat data.")
    download_data()
    print('Personachat data prepared!')

    print("Processing data...")
    with open(raw_data_file_path, encoding="utf-8") as f:
        raw_data_json = json.load(f)
    print("Processing training data...")
    train_dialogs = raw_data_json["train"]
    build_session(train_dialogs[0])
    with Pool(args.n_workers) as pool:
        train_sessions = list(
            tqdm(
                pool.imap(build_session, train_dialogs),
                total=len(train_dialogs)
            )
        )
    print("Processing valid data...")
    valid_dialogs = raw_data_json["valid"]
    with Pool(args.n_workers) as pool:
        valid_sessions = list(
            tqdm(
                pool.imap(build_session, valid_dialogs),
                total=len(valid_dialogs)
            )
        )
    dataset = train_dev_test_split(train_sessions, valid_sessions)

    print(f"Writing dataset json file to {config.dataset_path}...")
    with open(config.dataset_path, "w+", encoding="utf-8") as f:
        json.dump(dataset, f)
    print("Dataset built.")

    print(f"Building word count file...")
    word_count = collections.defaultdict(int)
    for sess in dataset["train"]:
        for uttr in sess["utterances"]:
            for token in uttr["text"].split(" "):
                word_count[token] += 1
    ordered_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    with open(config.word_count_path, "w+") as f:
        for word, count in ordered_word_count:
            f.write("{}\t{}\n".format(word, count))
