import code
import json
import argparse

import gensim

from .config import Config
from utils.helpers import load_partial_pretrained_word_embedding_as_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embedding_type", type=str, required=True, help="[glove, gensim_txt, gensim_bin]")
    parser.add_argument("-t", "--task", type=str, required=True, help="[dialog, ...]")
    parser.add_argument("-p", "--pretrained_embedding_path", type=str, required=True, help="path to pretrained embeddings")
    parser.add_argument("-o", "--output_embedding_path", type=str, required=True, help="path to output embeddings")
    args = parser.parse_args()

    config = Config(task=args.task)

    vocab = {}
    with open(config.word_count_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            token, count = line.strip().split("\t")
            if token not in vocab:
                vocab[token] = len(vocab)

    word_embedding = load_partial_pretrained_word_embedding_as_dict(vocab, args.pretrained_embedding_path, args.embedding_type)

    with open(args.output_embedding_path, "w+", encoding="utf-8") as f:
        json.dump(word_embedding, f)

