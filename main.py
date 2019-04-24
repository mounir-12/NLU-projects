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
MAX_GRAD_NORM = 5

# ------------------------------------------------------FUNCTIONS---------------------------------------------------------------------

def get_dataset(corpus, batch=False): # not used for now
    """ Split copus data matrix to inputs and outputs and get TF dataset representing them """
    inputs = corpus.data[:, :-1] # all columns except the last one (i,e the one containing EOS ids)
    outputs = corpus.data[:, 1:] # all columns except the first one (i,e the one containing BOS ids)
    
    return tf.data.Dataset.from_tensor_slices((inputs, outputs)).batch(BATCH_SIZE)

def clip_gradients(grads_and_vars, clip_ratio): # not used for now
    grads, vars = zip(*grads_and_vars) # extract gradients and corresponding var (i,e grad w.r.t that var)
    clipped, _ = tf.clip_by_global_norm(grads, clip_ratio) # clip gradients
    return zip(clipped, vars) # zip back clipped gradients with their corresponding vars

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
train_x = C_train.data[:, :-1]
train_y = C_train.data[:, 1:]
eval_x = C_eval.data[:, :-1]
eval_y = C_eval.data[:, 1:]

# Model
vocab_size = V_train.vocab_size # get true vocab size
time_steps = sentence_len-1
lstm = LSTM(vocab_size, embedding_size, hidden_size, time_steps)

# Training loop
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    num_batches = int(math.ceil(train_x.shape[0]/batch_size))
    batched_x = np.array_split(train_x, num_batches) # split into num_batches batches (last batch size may differ from previous ones)
    batched_y = np.array_split(train_y, num_batches) # split into num_batches batches (last batch size may differ from previous ones)
    for e in range(num_epochs):
        for b in range(num_batches):
            _, step, step_loss = lstm.train_step(sess, batched_x[b], batched_y[b])
            time_str = datetime.datetime.now().isoformat()
            print("epoch {}, batch {}:\n{}: step {}, loss {}".format(e+1, b+1, time_str, step, step_loss))
            if step % eval_every == 0:
                step, step_loss = lstm.eval_step(sess, eval_x, eval_y)
                print("\nEvaluation:\n    {}: step {}, loss {}\n".format(time_str, step, step_loss))
                

