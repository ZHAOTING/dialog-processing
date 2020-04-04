import code
import collections

import torch


class WhiteSpaceTokenizer(object):

    def __init__(self, word_count_path, vocab_size,
                 pad_token="<pad>", bos_token="<s>", eos_token="</s>",
                 unk_token="<unk>", sep_token="<sep>", cls_token="<cls>",
                 mask_token="<mask>", special_token_dict={}):
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.mask_token = mask_token

        # basic vocab dict
        self.word2id = {}
        # word prob dict
        self.word2prob = {}

        # fill in self.word2id and self.word2prob
        self.init_vocab(word_count_path, vocab_size, special_token_dict)

        # revserse vocab dict
        self.id2word = {}
        for k, v in self.word2id.items():
            self.id2word[v] = k

        # set special token ids
        for token_type in ["pad_token", "bos_token", "eos_token",
                           "unk_token", "sep_token", "cls_token",
                           "mask_token"]:
            token = getattr(self, token_type)
            setattr(self, f"{token_type}_id", self.word2id[token])
        for token_type, token in special_token_dict.items():
            setattr(self, f"{token_type}_id", self.word2id[token])

    def __len__(self):
        return len(self.word2id)

    def init_vocab(self, word_count_path, vocab_size, special_token_dict):
        # special token vocab
        for token in [self.pad_token, self.bos_token, self.eos_token,
                      self.unk_token, self.sep_token, self.cls_token,
                      self.mask_token]:
            self.word2id[token] = len(self.word2id)
        for token_type, token in special_token_dict.items():
            self.word2id[token] = len(self.word2id)

        # vocab from file
        word_count = {}
        with open(word_count_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[:vocab_size]
            for line in lines:
                if len(self.word2id) == vocab_size:
                    break
                token, count = line.split("\t")
                if token not in self.word2id:
                    self.word2id[token] = len(self.word2id)
                if token not in word_count:
                    word_count[token] = float(count)

        total_word_count = sum(list(word_count.values()))
        for word, count in word_count.items():
            self.word2prob[word] = count/total_word_count

    def convert_tokens_to_string(self, tokens):
        sent = " ".join(tokens)
        return sent

    def convert_string_to_tokens(self, sent):
        if len(sent) == 0:
            return []
        tokens = sent.split(" ")
        return tokens

    def convert_tokens_to_ids(self, tokens, bos_and_eos=False, add_eos=False, add_cls=False):
        ids = []
        if len(tokens) == 0:
            return ids
        if bos_and_eos:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        elif add_eos:
            tokens = tokens + [self.eos_token]
        if add_cls:
            tokens = [self.cls_token] + tokens
        for token in tokens:
            if token in self.word2id:
                token_id = self.word2id[token]
            else:
                token_id = self.word2id[self.unk_token]
            ids.append(token_id)
        return ids

    def convert_ids_to_tokens(self, ids, trim_bos=False, trim_pad=False, trim_from_eos=False, trim_after_eos=False):
        tokens = []
        for i in ids:
            if trim_bos and i == self.bos_token_id:
                continue
            if trim_pad and i == self.pad_token_id:
                continue
            if trim_from_eos and i == self.eos_token_id:
                break
            tokens.append(self.id2word[i])
            if trim_after_eos and i == self.eos_token_id:
                break
        return tokens

    def convert_batch_ids_to_tensor(self, batch_ids):
        """Turning a list token id sequences `batch_ids` into a mini-batch tensor.
        Sequences are right-padded with `self.pad_token_id`.
        """
        batch_lens = [len(ids) for ids in batch_ids]
        max_len = max(batch_lens)

        padded_batch_ids = [ids + [self.pad_token_id]*(max_len-len(ids)) for ids in batch_ids]
        batch_tensor = torch.LongTensor(padded_batch_ids)

        return batch_tensor
