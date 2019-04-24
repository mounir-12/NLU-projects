import os, math, datetime
import numpy as np
from time import time
import tensorflow as tf
# import tensorflow.contrib.eager as tfe

from lang_new import Vocabulary, Corpus
from model import LSTM


# tfe.enable_eager_execution()
tf.set_random_seed(9)

# --------------------------------------------------------CONSTANTS---------------------------------------------------------------------
# ------values in comment for cluster deployement-------
hidden_size = 128 # 512
embedding_size = 100 # 100
batch_size = 50 # 64
num_epochs = 20
eval_every = 10
n_lines = 1000 # None 
# ------------------------------------------------------
train_path = os.path.join(os.getcwd(), "data", "sentences.train")
eval_path = os.path.join(os.getcwd(), "data", "sentences.eval")
sentence_len = 30 # padded sentence length including EOS and BOS
vocab_size = 20000 # vocabulary size
clip_grad_norm = 5

# ------------------------------------------------------FUNCTIONS---------------------------------------------------------------------

def get_data(corpus, shuffle=False, batch=False, batch_size=None):
    data = corpus.data
    data_size = data.shape[0]
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        data = data[shuffle_indices]
    x = data[:, :-1]
    y = data[:, 1:]
    num_batches = 1 # default value
    if batch:
        num_batches = int(math.ceil(data_size/batch_size))
        x = np.array_split(x, num_batches) # split into num_batches batches (last batch size may differ from previous ones)
        y = np.array_split(y, num_batches) # split into num_batches batches (last batch size may differ from previous ones)
    return x, y, num_batches

# --------------------------------------------------------START---------------------------------------------------------------------
# Data IO
print("\nData IO ...")
t = time()
C_train = Corpus(train_path, sentence_len, read_from_file="corpus_train.pkl", n_sentences=n_lines) # create training corpus from training file
V_train = Vocabulary(vocab_size, read_from_file="vocab.pkl") # object to represent training vocabulary
V_train.build_from_corpus(C_train, write_to_file="vocab.pkl") # build vocabulary from training corpus
C_train.build_data_from_vocabulary(V_train, write_to_file="corpus_train.pkl") # build corpus data matrix from vocabulary, write object to disk

C_eval = Corpus(eval_path, sentence_len, read_from_file="corpus_eval.pkl", n_sentences=n_lines) # create evaluation corpus from evaluation file
C_eval.build_data_from_vocabulary(V_train, write_to_file="corpus_eval.pkl") # build corpus data matrix from vocabulary, write object to disk
print("Total time (s): ", time() - t, "\n")

print("Vocab size: ", V_train.vocab_size)
print("Train data matrix shape: ", C_train.data.shape)
print("Eval data matrix shape: ", C_eval.data.shape)

# Train and Eval data
batched_x, batched_y, num_batches = get_data(C_train, shuffle=True, batch=True, batch_size=batch_size) # training data, shuffled, batched
eval_x, eval_y, _ = get_data(C_eval) # eval data no shuffling or batching

# Model
vocab_size = V_train.vocab_size # get true vocab size
time_steps = sentence_len-1
lstm = LSTM(vocab_size, embedding_size, hidden_size, time_steps, clip_grad_norm)

# Training loop
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    for e in range(num_epochs):
        for b in range(num_batches):
            _, step, step_loss = lstm.train_step(sess, batched_x[b], batched_y[b])
            time_str = datetime.datetime.now().isoformat()
            print("epoch {}, batch {}:\n{}: step {}, loss {}".format(e+1, b+1, time_str, step, step_loss))
            if step % eval_every == 0:
                step, step_loss = lstm.eval_step(sess, eval_x, eval_y)
                print("\nEvaluation:\n    {}: step {}, loss {}\n".format(time_str, step, step_loss))
                

