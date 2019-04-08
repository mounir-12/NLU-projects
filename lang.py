import pickle
from collections import Counter, defaultdict


class Lang:
    def __init__(self, tokenizer=lambda s: s.split(' ')):
        self.tokenizer = tokenizer

        self.UNK_token = 0
        self.BOS_token = 1
        self.EOS_token = 2
        self.PAD_token = 3

        self.token2count = Counter()
        self.token2index = defaultdict(lambda: self.UNK_token)
        self.index2token = {
            self.UNK_token: '<unk>',
            self.BOS_token: '<bos>',
            self.EOS_token: '<eos>',
            self.PAD_token: '<pad>'
        }
        self.n_tokens = len(self.index2token)  # Count UNK, SOS, EOS and PAD

    def tokenize(self, sentence: str):
        return self.tokenizer(sentence)

    def add_token_temp(self, token: str):
        self.token2count[token] += 1

    def add_tokens(self, tokens):
        for token in tokens:
            self.add_token_temp(token)

    def add_sentence(self, sentence: str):
        self.add_tokens(self.tokenize(sentence))

    def build(self, voc_size=20000):
        counts = self.token2count.most_common(voc_size)
        for token, count in counts:
            self.token2index[token] = self.n_tokens
            self.index2token[self.n_tokens] = token
            self.n_tokens += 1

    def pad(self, sentence: str, to_length=30):
        length = sentence.count(' ') + 1
        return sentence + (' %s' % self.index2token[self.PAD_token]) * (to_length - length)

    def dump(self, path: str):
        lang_dict = {"token2index": self.token2index, "token2count": self.token2count,
                     "index2token": self.index2token, "n_tokens": self.n_tokens}

        with open(path, "wb") as f:
            pickle.dump(lang_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str, tokenizer=lambda s: s.split(' ')):
        with open(path, 'rb') as f:
            lang_dict = pickle.load(f)

        lang = Lang(tokenizer)
        lang.token2index = lang_dict["token2index"]
        lang.token2count = lang_dict["token2count"]
        lang.index2token = lang_dict["index2token"]
        lang.n_tokens = lang_dict["n_tokens"]

        return lang
