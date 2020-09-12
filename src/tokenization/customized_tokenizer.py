import code
import collections

class CustomizedTokenizer(object):

    def __init__(self, token_dict={}):
        # basic vocab dict
        self.word2id = {}
        
        # fill in self.word2id and self.word2prob
        for token_type, token in token_dict.items():
            self.word2id[token] = len(self.word2id)
            setattr(self, f"{token_type}_id", self.word2id[token])

        # revserse vocab dict
        self.id2word = {}
        for k, v in self.word2id.items():
            self.id2word[v] = k

    def __len__(self):
        return len(self.word2id)
