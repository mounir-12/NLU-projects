import pickle
from collections import Counter, defaultdict


class Lang:
    def __init__(self, tokenizer=lambda s: s.split(' ')):
        self.tokenizer = tokenizer

        self._built = False

        self.PAD_token = 0
        self.BOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3

        self.token2count = Counter()
        self.token2index = defaultdict(lambda: self.UNK_token)
        self.index2token = {
            self.PAD_token: '<pad>',
            self.BOS_token: '<bos>',
            self.EOS_token: '<eos>',
            self.UNK_token: '<unk>'
        }
        self.n_tokens = len(self.index2token)  # Count UNK, SOS, EOS and PAD

    def tokenize(self, sentence: str):
        return self.tokenizer(sentence.strip())

    def add_token_temp(self, token: str):
        if self._built:
            raise RuntimeError("Lang was already built")
        self.token2count[token] += 1

    def add_tokens(self, tokens):
        for token in tokens:
            self.add_token_temp(token)

    def add_sentence(self, sentence: str):
        self.add_tokens(self.tokenize(sentence.strip()))

    def build(self, voc_size=20000):
        if self._built:
            raise RuntimeError("Lang was already built")
        self.token2count = self.token2count.most_common(voc_size)
        for token, count in self.token2count:
            self.token2index[token] = self.n_tokens
            self.index2token[self.n_tokens] = token
            self.n_tokens += 1

        self._built = True

    def embed(self, tokens, length=30):
        assert len(tokens) - 2 < length
        embedded = [self.token2index[token] for token in tokens]
        self.pad_tokens(embedded, length - 2)  # Take into account BOS and EOS tokens

        embedded.insert(0, self.BOS_token)
        embedded.append(self.EOS_token)

        assert len(embedded) == length
        assert embedded[0] == self.BOS_token
        assert embedded[-1] == self.EOS_token

        return embedded

    def pad_tokens(self, tokens: list, to_length=30):
        tokens.extend([self.PAD_token] * (to_length - len(tokens)))
        return tokens

    def pad(self, sentence: str, to_length=30):
        length = sentence.strip().count(' ') + 1
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
