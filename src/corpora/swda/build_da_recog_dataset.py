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
from .swda_reader.swda import Transcript
from .dataset_split import train_set_idx, dev_set_idx, test_set_idx
from utils.helpers import standardize_english_text

config = Config(task="da_recog")
raw_data_dir = config.raw_data_dir
zipfile_path = f"{config.raw_data_dir}/swda.zip"
extracted_dir = f"{config.raw_data_dir}/swda"


def clean_swda_text(text):
    text = re.sub(r"\{\w (.*?)\}", r"\1", text)
    text = re.sub(r"\{\w (.*?)\ --", r"\1", text)
    text = re.sub(r"\*\[\[.*?\]\]", "", text)
    text = standardize_english_text(text)
    return text


def download_data():
    """Download and unpack dialogs"""

    if not os.path.exists(zipfile_path):
        print(f'Downloading {config.download_url} to {zipfile_path}')
        urlretrieve(config.download_url, zipfile_path)
        print(f'Successfully downloaded {zipfile_path}')

    zip_ref = ZipFile(zipfile_path, 'r')
    zip_ref.extractall(config.raw_data_dir)
    zip_ref.close()

    os.rename(f"{config.raw_data_dir}/swda", extracted_dir)


def build_session(filepath):
    trans = Transcript(filepath, f"{extracted_dir}/swda-metadata.csv")
    conversation_no = f"sw{trans.conversation_no}"
    topic = trans.topic_description.lower()
    prompt = trans.prompt.lower()

    # adapt utterances with "+" dialog act
    uttrs = []
    last_idx = {"A": -1, "B": -1}
    for uttr in trans.utterances:
        text = clean_swda_text(uttr.text)
        da = uttr.damsl_act_tag()
        speaker = uttr.caller

        if da is None or text.strip() == "":
            continue
        elif da == "x":
            continue
        elif da == "+":
            if last_idx[speaker] > -1:
                uttrs[last_idx[speaker]]["text"] += f" {text}"
            else:
                continue
        else:
            uttr = {
                "floor": speaker,
                "text": text,
                "utterance_meta": {
                    "dialog_act": da
                }
            }
            uttrs.append(uttr)
            last_idx[speaker] = len(uttrs)-1

        if da == "" or text.strip() == "":
            code.interact(local=locals())

    session = {
        "utterances": uttrs,
        "dialog_meta": {
            "conversation_no": conversation_no,
            "topic": topic,
            "prompt": prompt,
        }
    }
    return session


def train_dev_test_split_by_conv_no(sessions):
    dataset = {"train": [], "dev": [], "test": []}
    for session in sessions:
        conv_no = session["dialog_meta"]["conversation_no"]
        if conv_no in train_set_idx:
            dataset["train"].append(session)
        elif conv_no in dev_set_idx:
            dataset["dev"].append(session)
        elif conv_no in test_set_idx:
            dataset["test"].append(session)
        else:
            continue
            # raise Exception(f"conversation no {conv_no} not in any dataset split.")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    if not os.path.exists(config.raw_data_dir):
        os.makedirs(config.raw_data_dir)
    if not os.path.exists(config.task_data_dir):
        os.makedirs(config.task_data_dir)

    print("Preparing SwDA data.")
    download_data()
    print('SwDA data prepared!')

    print("Extracting sessions...")
    # Get file paths
    filepaths = []
    for group_dir in os.listdir(extracted_dir):
        if re.match(r"sw\d*utt", group_dir):
            for filename in os.listdir(f"{extracted_dir}/{group_dir}"):
                if filename[-3:] == "csv":
                    filepaths.append(f"{extracted_dir}/{group_dir}/{filename}")
    with Pool(args.n_workers) as pool:
        sessions = list(
            tqdm(
                pool.imap(build_session, filepaths),
                total=len(filepaths)
            )
        )

    print("Building dataset split...")
    dataset = train_dev_test_split_by_conv_no(sessions)

    print(f"Dialog act tags:")
    da_dict = collections.defaultdict(int)
    for sess in dataset["train"]:
        for uttr in sess["utterances"]:
            da = uttr["utterance_meta"]["dialog_act"]
            da_dict[da] += 1
    sorted_da_list = sorted(da_dict.items(), key=lambda x: x[1], reverse=True)
    print(len(sorted_da_list))
    print([k for k, v in sorted_da_list])

    print("Processing sessions...")
    nlp = spacy.load('en_core_web_sm')

    def process_session(session):
        def tokenize(string):
            return [token.text for token in nlp(standardize_english_text(string))]
        for uttr in session["utterances"]:
            uttr["text"] = " ".join(tokenize(uttr["text"]))
    with Pool(args.n_workers) as pool:
        sessions = list(
            tqdm(
                pool.imap(process_session, sessions),
                total=len(sessions)
            )
        )

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

    

