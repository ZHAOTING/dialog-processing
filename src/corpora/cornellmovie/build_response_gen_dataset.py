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

config = Config(task="response_gen")
zipfile_path = f"{config.raw_data_dir}/cornell.zip"
extracted_dir = f"{config.raw_data_dir}/cornell"
lines_file_path = f"{extracted_dir}/movie_lines.txt"
conv_file_path = f"{extracted_dir}/movie_conversations.txt"


def clean_cornellmovie_text(text):
    text = text.replace("<u>", "")
    text = text.replace("</u>", "")
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

        os.rename(f"{config.raw_data_dir}/cornell movie-dialogs corpus", extracted_dir)


def load_lines(fileName,
               fields=["lineID", "characterID", "movieID", "character", "text"],
               delimiter=" +++$+++ "):
    lines = {}

    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(delimiter)

            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]

            lines[lineObj['lineID']] = lineObj

    return lines


def load_conversations(fileName, lines,
                       fields=["character1ID", "character2ID", "movieID", "utteranceIDs"],
                       delimiter=" +++$+++ "):
    conversations = []

    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(delimiter)

            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]

            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])

            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])

            conversations.append(convObj)

    return conversations


def train_dev_test_split_by_conversation(conversations, split_ratio=[0.8, 0.1, 0.1]):
    train_ratio, dev_ratio, test_ratio = split_ratio
    assert train_ratio + dev_ratio + test_ratio == 1.0

    n_conversations = len(conversations)

    # Random shuffle movie list
    random.seed(0)
    random.shuffle(conversations)

    # Train / Validation / Test Split
    train_split = int(n_conversations * train_ratio)
    dev_split = int(n_conversations * (train_ratio + dev_ratio))

    train = conversations[:train_split]
    dev = conversations[train_split:dev_split]
    test = conversations[dev_split:]
    return train, dev, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    if not os.path.exists(config.raw_data_dir):
        os.makedirs(config.raw_data_dir)
    if not os.path.exists(config.task_data_dir):
        os.makedirs(config.task_data_dir)

    print("Preparing Cornell data.")
    download_data()
    print('Cornell data prepared!')

    print("Loading lines")
    lines = load_lines(lines_file_path)
    print('Number of lines:', len(lines))

    print("Loading conversations...")
    conversations = load_conversations(conv_file_path, lines)
    print('Number of conversations:', len(conversations))

    print('Train/Valid/Test Split')
    train, dev, test = train_dev_test_split_by_conversation(conversations)
    print(f'Train set: {len(train)} conversations')
    print(f'Validation set: {len(dev)} conversations')
    print(f'Test set: {len(test)} conversations')

    nlp = spacy.load('en_core_web_sm')
    
    def tokenize_conversation(conv):
        def tokenize(string):
            return [token.text for token in nlp(clean_cornellmovie_text(string))]
        for line in conv["lines"]:
            line["tokens"] = tokenize(line['text'])
        return conv

    print("Building dataset json file...")
    dataset = {"train": [], "dev": [], "test": []}
    for split_type, conv_objects in [('train', train), ('dev', dev), ('test', test)]:
        print(f"Preprocessing {split_type} data...")

        print("Tokenizing...")
        with Pool(args.n_workers) as pool:
            conversations = list(
                tqdm(
                    pool.imap(tokenize_conversation, conv_objects),
                    total=len(conv_objects))
                )

        if split_type == "train":
            print("Building word count...")
            word_count = collections.defaultdict(int)
            for conv in conversations:
                for line in conv["lines"]:
                    for token in line["tokens"]:
                        word_count[token] += 1
            ordered_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

            with open(config.word_count_path, "w+") as f:
                for word, count in ordered_word_count:
                    f.write("{}\t{}\n".format(word, count))

        for conv in conversations:
            session = {
                "utterances": [],
                "dialog_meta": {
                    "character1ID": conv["character1ID"],
                    "character2ID": conv["character2ID"],
                    "movieID": conv["movieID"]
                }
            }
            for line in conv["lines"]:
                if len(line["tokens"]) > 0:
                    uttr = {
                        "floor": "A" if line["characterID"] == conv["character1ID"] else "B",
                        "text": " ".join(line["tokens"]),
                        "utterance_meta": {
                            "characterID": line["characterID"],
                            "character": line["character"]
                        }
                    }
                    session["utterances"].append(uttr)
            if len(session["utterances"]) >= 2:
                dataset[split_type].append(session)

    print(f"Writing dataset json file to {config.dataset_path}...")
    with open(config.dataset_path, "w+", encoding="utf-8") as f:
        json.dump(dataset, f)
    print("Dataset built.")
