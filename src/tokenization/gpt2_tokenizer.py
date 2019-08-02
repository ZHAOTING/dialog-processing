import code
import collections

import torch
from pytorch_transformers import GPT2Tokenizer
        
class ModGPT2Tokenizer(object):
    def __init__(self):
        self.pretrained = GPT2Tokenizer.from_pretrained(
            'gpt2', 
            unk_token="<|endoftext|>" 
        ) # Initializing `unk_token` is a quick fix for a pytorch-transformers bug. See Issue 799.

        self.special_tokens = ["<pad>", "<s>", "</s>", "<speaker1>", "<speaker2>","<cls>"]
        self.adapt_vocab()

        self.word2id = self.pretrained.encoder
        self.word2id.update(self.pretrained.added_tokens_encoder)
        self.id2word = self.pretrained.decoder
        self.id2word.update(self.pretrained.added_tokens_decoder)

        self.pad_id = self.word2id["<pad>"]
        self.bos_id = self.word2id["<s>"]
        self.eos_id = self.word2id["</s>"]
        self.speaker1_id = self.word2id["<speaker1>"]
        self.speaker2_id = self.word2id["<speaker2>"]
        self.cls_id = self.word2id["<cls>"]

    def __len__(self):
        return len(self.pretrained)

    def adapt_vocab(self):
        self.pretrained.add_tokens(self.special_tokens)

    def convert_tokens_to_sent(self, tokens):
        decoded_sent = self.pretrained.decode(self.pretrained.convert_tokens_to_ids(tokens), clean_up_tokenization_spaces=False)
        sent = decoded_sent.\
            replace("<s>", "<s> ").\
            replace("</s>", " </s>").\
            replace("<pad>", " <pad>").\
            replace("<speaker1>", "").\
            replace("<speaker2>", "").\
            replace("<cls>", "")
        return sent

    def convert_sent_to_tokens(self, sent):
        if len(sent) == 0:
            return []
        else:
            return self.pretrained.tokenize(sent)

    def convert_tokens_to_ids(self, tokens, bos_and_eos=False, add_eos=False):
        ids = self.pretrained.convert_tokens_to_ids(tokens)
        if len(ids) == 0:
            return ids
        if bos_and_eos:
            ids = [self.bos_id] + ids + [self.eos_id]
        elif add_eos:
            ids = ids + [self.eos_id]
        return ids

    def convert_ids_to_tokens(self, ids, trim_bos=False, trim_pad=False, trim_from_eos=False, trim_after_eos=False):
        _tokens = self.pretrained.convert_ids_to_tokens(ids)
        tokens = []
        for token in _tokens:
            if trim_bos and token == self.id2word[self.bos_id]:
                continue
            if trim_pad and token == self.id2word[self.pad_id]:
                continue
            if trim_from_eos and token == self.id2word[self.eos_id]:
                break
            tokens.append(token)
            if trim_after_eos and token == self.id2word[self.eos_id]:
                break
        return tokens

    def convert_batch_ids_to_tensor(self, batch_ids):
        """Turning a list token id sequences `batch_ids` into a mini-batch tensor.
        Sequences are right-padded with `self.pad_id`.
        """
        batch_lens = [len(ids) for ids in batch_ids]
        max_len = max(batch_lens)

        padded_batch_ids = [ids + [self.pad_id]*(max_len-len(ids)) for ids in batch_ids]
        batch_tensor = torch.LongTensor(padded_batch_ids)

        return batch_tensor
