import code
import collections

import torch
from transformers import RobertaTokenizer
        
class ModRobertaTokenizer(object):
    def __init__(self, model_size):
        assert model_size in ["base", "large", "large-mnli"]
        self.pretrained = RobertaTokenizer.from_pretrained(f"roberta-{model_size}")

        self.special_tokens = ["<speaker1>", "<speaker2>"]
        self.adapt_vocab()

        self.word2id = self.pretrained.encoder
        self.word2id.update(self.pretrained.added_tokens_encoder)
        self.id2word = self.pretrained.decoder
        self.id2word.update(self.pretrained.added_tokens_decoder)

        self.speaker1_token_id = self.word2id["<speaker1>"]
        self.speaker2_token_id = self.word2id["<speaker2>"]
        self.pad_token_id = self.word2id[self.pretrained.pad_token]
        self.cls_token_id = self.word2id[self.pretrained.cls_token]
        self.sep_token_id = self.word2id[self.pretrained.sep_token]

    def __len__(self):
        return len(self.pretrained)

    def adapt_vocab(self):
        self.pretrained.add_tokens(self.special_tokens)

    def convert_tokens_to_string(self, tokens):
        sent = self.pretrained.decode(self.pretrained.convert_tokens_to_ids(tokens))
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
        return ids

    def convert_ids_to_tokens(self, ids, trim_pad=False, **kwargs):
        _tokens = self.pretrained.convert_ids_to_tokens(ids)
        tokens = []
        for token in _tokens:
            if trim_pad and token == self.id2word[self.pad_token_id]:
                continue
            tokens.append(token)
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
