import code
import argparse
import json
import os
import random
import re
import collections
from multiprocessing import Pool
from urllib.request import urlretrieve
from zipfile import ZipFile

from tqdm import tqdm
import spacy

from .config import Config
from utils.helpers import standardize_english_text

gen_config = Config(task="response_gen")
config = Config(task="lm")
raw_data_dir = config.raw_data_dir
zipfile_path = f"{config.raw_data_dir}/dailydialog.zip"
extracted_dir = f"{config.raw_data_dir}/dailydialog"


if __name__ == "__main__":

    if not os.path.exists(gen_config.dataset_path):
        raise Exception(f"Must build response generation data set first.")

    if not os.path.exists(config.task_data_dir):
        os.makedirs(config.task_data_dir)

    print(f"Loading response generation data set...")
    with open(gen_config.dataset_path, encoding="utf-8") as f:
        gen_dataset = json.load(f)

    print(f"Building language modeling data set...")
    dataset = {}
    for stage in gen_dataset.keys():
        sents = []
        for sess in gen_dataset[stage]:
            uttrs = sess["utterances"]
            sents += uttrs
        dataset[stage] = sents

    print(f"Writing dataset json file to {config.dataset_path}...")
    with open(config.dataset_path, "w+", encoding="utf-8") as f:
        json.dump(dataset, f)
    print("Dataset built.")

    print(f"Building word count file...")
    word_count = collections.defaultdict(int)
    for sent in dataset["train"]:
        for token in sent["text"].split(" "):
            word_count[token] += 1
    ordered_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Writing word count file to {config.word_count_path}...")
    with open(config.word_count_path, "w+") as f:
        for word, count in ordered_word_count:
            f.write("{}\t{}\n".format(word, count))
