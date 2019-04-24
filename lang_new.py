import pickle
import os
import numpy as np
from collections import Counter, defaultdict

io_dir = os.path.join(os.getcwd(), "lang")

class Vocabulary:
    def __init__(self, vocab_size, read_from_file=None):
        # special tokens
        self.PAD_token = '<pad>'
        self.BOS_token = '<bos>'
        self.EOS_token = '<eos>'
        self.UNK_token = '<unk>'
        # ------------------------read Vocabulary object from dumped file if possible-------------------------
        if read_from_file is not None: # file to read from provided
            in_file = os.path.join(io_dir, read_from_file)
            if os.path.exists(io_dir) and os.path.isfile(in_file): # read if file exists
                print("Found Vocabulary file, Reading Vocabulary Object ...")
                with open(in_file, 'rb') as f:
                    vocab_dict = pickle.load(f)
                
                self.vocab_size = vocab_dict["vocab_size"]
                self.token2id = vocab_dict["token2id"]
                self.id2token = vocab_dict["id2token"]
                self._built = True
                print("Done.")
                return
        # ----------------------------------------------------------------------------------------------------
        # otherwise
        self.vocab_size = vocab_size
        self._built = False # vocabulary not yet built, need to call build_from_corpus()
    
    def build_from_corpus(self, corpus, write_to_file=None):

        print("Building Vocabulary from Corpus ...")
        
        if self._built:
            print("Already built. Done.")
            return

        all_tokens = [] # list of all tockens in corpus
        for tokenized_sentence in corpus.tokenized_sentences:
            for token in tokenized_sentence:
                all_tokens.append(token)
        token2count = Counter(all_tokens).most_common(self.vocab_size) # count corpus tokens and extract the most common ones with their counts
        vocab_tokens = [self.BOS_token, self.EOS_token, self.PAD_token, self.UNK_token] # add special tokens to Vocabulary
        vocab_tokens.extend([token for token, count in token2count]) # add most common corpus tokens to Vocabulary
        self.vocab_size = len(vocab_tokens) # update vocab_size (take special tokens into account)
        self.token2id = {token: i for i, token in enumerate(vocab_tokens)} # assign an id to each vocab token, store mapping in object
        self.id2token = {i: token for token, i in self.token2id.items()} # create reverse map
        # print(self.token2id)
        # print(self.id2token) 
        print("Done.")

        # ----------------------------------------Dump vocabulary object-----------------------------------------
        if write_to_file is not None: # target file name provided
            print("Writing Vocabulary Object ... ")
            vocab_dict = {"vocab_size": self.vocab_size, "token2id": self.token2id, "id2token": self.id2token}
            dump(write_to_file, vocab_dict)
            print("Done.")
        # -------------------------------------------------------------------------------------------------------
        
        
class Corpus:
    def __init__(self, path, sentence_len, read_from_file=None, n_sentences=None):
        # ------------------------read Corpus object from dumped file if possible--------------------------------
        if read_from_file is not None: # file to read from provided
            in_file = os.path.join(io_dir, read_from_file)
            if os.path.exists(io_dir) and os.path.isfile(in_file): # load if file exists
                print("Found Corpus File, Reading Corpus Object ...")
                with open(in_file, 'rb') as f:
                    corpus_dict = pickle.load(f)
                
                self.sentence_len = corpus_dict["sentence_len"]
                self.n_sentences = corpus_dict["n_sentences"]
                self.tokenized_sentences = corpus_dict["tokenized_sentences"]
                self.data = corpus_dict["data"]
                self._built = True
                print("Done.")
                return
        # -------------------------------------------------------------------------------------------------------

        # otherwise read from text file provided with path
        self.sentence_len = sentence_len
        self.n_sentences = n_sentences
        self._built = False # not yet built, need to call build_data_from_vocabulary()

        print("Reading and Tokenizing Corpus sentences from text file ...")
        tokenized_sentences = [] # list of tokenized sentences
        with open(path) as sentences: # open corpus of sentences
            for sentence in sentences: # for each sentence in corpus
                tokens = sentence.strip().split(" ") # get sentence tokens
                if len(tokens) <= (sentence_len-2): # only consider sentences with "nb tokens <= sentence_len-2"
                    tokenized_sentences.append(tokens) # append sentence tokens
                if(n_sentences and len(tokenized_sentences) == n_sentences): # reached max nb of sentences to read
                    break
        self.n_sentences = len(tokenized_sentences) # update self.n_sentences to nb of sentences read (in case n_sentences=None)
        self.tokenized_sentences = tokenized_sentences # store corpus
        print("Done.")

    def build_data_from_vocabulary(self, V, write_to_file=None):
        """ For each tokenized sentence, extend by adding BOS, EOS, PAD (where needed) and replace by UNK (where needed)
            All sentences having same length, create data matrix of size n_sentences x sentence_len with token ids (instead of tokens)
        """
        
        print("Building Data Matrix ...")
        
        if self._built:
            print("Already built. Done.")
            return

        self.data = np.empty([self.n_sentences, self.sentence_len], dtype=int)
        for i in range(len(self.tokenized_sentences)): # index over sentences
            tokenized_sentence = self.tokenized_sentences[i] # read tokenized sentence from list
            for j in range(self.sentence_len): # index over sentence tokens
                length = len(tokenized_sentence) # the true sentence length
                if j == 0: # add BOS
                    self.data[i, j] = V.token2id[V.BOS_token]
                elif j == (self.sentence_len-1): # add EOS
                    self.data[i, j] = V.token2id[V.EOS_token]
                elif j >= length: # no more sentence tokens: add PAD
                    self.data[i, j] = V.token2id[V.PAD_token]
                else:
                    token = tokenized_sentence[j] # read sentence token
                    if token in V.token2id: # token part of vocabulary: replace by id
                        self.data[i, j] = V.token2id[token]
                    else: # token not part of vocabulary: add UNK
                        self.data[i, j] = V.token2id[V.UNK_token]
        # print(self.data)
        print("Done.")

        # ---------------------------------------Dump corpus object if possible--------------------------------
        if write_to_file is not None: # target file name provided
            print("Writing Corpus Object ... ")
            corpus_dict = {"sentence_len": self.sentence_len, "n_sentences": self.n_sentences,
                     "tokenized_sentences": self.tokenized_sentences, "data": self.data}
            dump(write_to_file, corpus_dict) # call function to dump object
            print("Done.")
        # ------------------------------------------------------------------------------------------------------

def dump(write_to_file, dictionary):
    out_file = os.path.join(io_dir, write_to_file) # full path
    if not os.path.exists(io_dir): # create io_dir if not existant
        os.makedirs(io_dir)
    with open(out_file, "wb+") as f: # dump object
        pickle.dump(dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
