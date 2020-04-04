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
raw_data_dir = config.raw_data_dir
zipfile_path = f"{config.raw_data_dir}/dailydialog.zip"
extracted_dir = f"{config.raw_data_dir}/dailydialog"


def clean_dd_text(text):
    text = standardize_english_text(text)
    text = re.sub(r"(\w)n ' (t\W)", r"\1 n'\2", text)
    text = re.sub(r" ' (m|s|re|ve|d|ll)(\W)", r" '\1\2", text)
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

        os.rename(f"{config.raw_data_dir}/ijcnlp_dailydialog", extracted_dir)


def load_txt_files(text_filepath, da_filepath, topic_filepath, emotion_filepath):
    dialogs = []
    with open(text_filepath, encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            uttrs = line.split("__eou__")[:-1]
            uttrs = [uttr.strip() for uttr in uttrs]
            dialogs.append(uttrs)

    dialog_das = []
    with open(da_filepath, encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            dialog_acts = line.split(" ")
            dialog_das.append(dialog_acts)

    dialog_emotions = []
    with open(emotion_filepath, encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            emotions = line.split(" ")
            dialog_emotions.append(emotions)

    dialog_topics = []
    with open(topic_filepath, encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            dialog_topics.append(line)

    return dialogs, dialog_das, dialog_emotions, dialog_topics


def aggregate_valid_conversations(_dialog_uttrs, _dialog_das, _dialog_emotions, _dialog_topics):
    # Collect valid conversations
    dialogs = []
    n_unmatch = 0
    n_duplicate = 0
    duplicate_uttrs_record = {}
    for uttrs, dialog_acts, emotions, topic in zip(_dialog_uttrs, _dialog_das, _dialog_emotions, _dialog_topics):

        # skip unmatched dialog
        try:
            assert len(uttrs) == len(dialog_acts) and len(uttrs) == len(emotions)
        except AssertionError:
            n_unmatch += 1
            continue

        # skip duplicate dialog
        n_duplicate_uttrs = 0
        for uttr in uttrs:
            if uttr in duplicate_uttrs_record:
                n_duplicate_uttrs += 1
        if n_duplicate_uttrs/len(uttrs) > 0.5:
            n_duplicate += 1
            continue
        for uttr in uttrs:
            duplicate_uttrs_record[uttr] = True
        dialogs.append([uttrs, dialog_acts, emotions, topic])
    print(f"{n_unmatch} dialogs unmatched")
    print(f"{n_duplicate} dialogs duplicated")

    return dialogs


def train_dev_test_split_by_topic(sessions):
    topic2sessions = collections.defaultdict(list)
    for sess in sessions:
        topic = sess["dialog_meta"]["topic"]
        topic2sessions[topic].append(sess)

    dataset = {"train": [], "test": [], "dev": []}
    for topic_sessions in topic2sessions.values():
        n_sess = len(topic_sessions)

        train_split, dev_split, test_split = topic_sessions[:int(0.8*n_sess)], topic_sessions[int(0.8*n_sess):int(0.9*n_sess)], topic_sessions[int(0.9*n_sess):]
        dataset["train"] += train_split
        dataset["dev"] += dev_split
        dataset["test"] += test_split

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    if not os.path.exists(config.raw_data_dir):
        os.makedirs(config.raw_data_dir)
    if not os.path.exists(config.task_data_dir):
        os.makedirs(config.task_data_dir)

    print("Preparing DailyDialog data.")
    download_data()
    print('DailyDialog data prepared!')

    print("Aggregating conversation information...")
    dialogs, dialog_das, dialog_emotions, dialog_topics = load_txt_files(
        text_filepath=f"{extracted_dir}/dialogues_text.txt",
        da_filepath=f"{extracted_dir}/dialogues_act.txt",
        topic_filepath=f"{extracted_dir}/dialogues_topic.txt",
        emotion_filepath=f"{extracted_dir}/dialogues_emotion.txt"
    )
    dialogs = aggregate_valid_conversations(
        _dialog_uttrs=dialogs, 
        _dialog_das=dialog_das, 
        _dialog_emotions=dialog_emotions, 
        _dialog_topics=dialog_topics
    )

    print("Building sessions...")
    nlp = spacy.load('en_core_web_sm')
    
    def build_session(dialog):
        uttrs, dialog_acts, emotions, topic = dialog
        session = {
            "utterances": [],
            "dialog_meta": {
                "topic": config.id2topic[int(topic)]
            }
        }
        speaker_idx = 0
        for uttr_text, dialog_act, emotion in zip(uttrs, dialog_acts, emotions):
            dialog_act = config.id2dialog_act[int(dialog_act)]
            emotion = config.id2emotion[int(emotion)]
            floor = ["A", "B"][speaker_idx]
            speaker_idx = 1 - speaker_idx

            def tokenize(string):
                return [token.text for token in nlp(clean_dd_text(string))]
            uttr_tokens = tokenize(uttr_text)
            tokenized_uttr_text = " ".join(uttr_tokens)
            uttr = {
                "floor": floor,
                "text": tokenized_uttr_text,
                "utterance_meta": {
                    "emotion": emotion,
                    "dialog_act": dialog_act,
                }
            }
            session["utterances"].append(uttr)
        return session
    with Pool(args.n_workers) as pool:
        sessions = list(
            tqdm(
                pool.imap(build_session, dialogs),
                total=len(dialogs)
            )
        )

    print("Building dataset split...")
    dataset = train_dev_test_split_by_topic(sessions)

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
    
