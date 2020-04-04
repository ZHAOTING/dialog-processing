import code
import collections

import torch
from transformers import GPT2Tokenizer


class ModGPT2Tokenizer(object):

    def __init__(self, model_size, pad_token="<pad>", bos_token="<s>", eos_token="</s>",
                 unk_token="<unk>", sep_token="</s>", cls_token="<s>",
                 mask_token="<mask>", special_token_dict={}):
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.mask_token = mask_token

        # load from pretrained tokenizer
        model_size2model_name = {
            "small": "gpt2",
            "medium": "gpt2-medium",
            "large": "gpt2-large",
            "xl": "gpt2-xl",
            "distil": "distilgpt2"
        }
        assert model_size in model_size2model_name
        self.pretrained = GPT2Tokenizer.from_pretrained(model_size2model_name[model_size])

        # add special tokens
        self._adapt_vocab(special_token_dict)

        # vocab dict and revserse vocab dict
        self.word2id = self.pretrained.encoder
        self.word2id.update(self.pretrained.added_tokens_encoder)
        self.id2word = self.pretrained.decoder
        self.id2word.update(self.pretrained.added_tokens_decoder)

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

    def _adapt_vocab(self, special_token_dict):
        for token in [self.pad_token, self.bos_token, self.eos_token, self.unk_token,
                      self.sep_token, self.cls_token, self.mask_token]:
            if token != "<|endoftext|>":
                self.pretrained.add_tokens([token])
        self.pretrained.add_tokens(list(special_token_dict.values()))

    def convert_tokens_to_string(self, tokens):
        decoded_sent = self.pretrained.decode(self.pretrained.convert_tokens_to_ids(tokens), clean_up_tokenization_spaces=False)
        sent = decoded_sent.\
            replace(self.bos_token, f"{self.bos_token} ").\
            replace(self.eos_token, f" {self.eos_token}").\
            replace(self.pad_token, f" {self.pad_token}")

        sent = sent.strip()  # NOTE: quick-fix to preceding whitespace, see "https://github.com/huggingface/pytorch-transformers/issues/1285"

        return sent

    def convert_string_to_tokens(self, sent):
        if len(sent) == 0:
            return []
        else:
            return self.pretrained.tokenize(sent)

    def convert_tokens_to_ids(self, tokens, bos_and_eos=False, add_eos=False):
        ids = self.pretrained.convert_tokens_to_ids(tokens)
        if len(ids) == 0:
            return ids
        if bos_and_eos:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        elif add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def convert_ids_to_tokens(self, ids, trim_bos=False, trim_pad=False, trim_from_eos=False, trim_after_eos=False):
        _tokens = self.pretrained.convert_ids_to_tokens(ids)
        tokens = []
        for token in _tokens:
            if trim_bos and token == self.id2word[self.bos_token_id]:
                continue
            if trim_pad and token == self.id2word[self.pad_token_id]:
                continue
            if trim_from_eos and token == self.id2word[self.eos_token_id]:
                break
            tokens.append(token)
            if trim_after_eos and token == self.id2word[self.eos_token_id]:
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
