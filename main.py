import os
from time import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from lang_new import Vocabulary, Corpus
from model import LSTM

tfe.enable_eager_execution()
tf.set_random_seed(9)

# --------------------------------------------------------CONSTANTS---------------------------------------------------------------------
# ------values in comment for cluster deployement-------
BATCH_SIZE = 2 # 64
EMBEDDING_SIZE = 10  # 100
HIDDEN_SIZE = 128 # 512
N_LINES = 1000 # None 
# ------------------------------------------------------
MAX_GRAD_NORM = 5
train_path = os.path.join(os.getcwd(), "data", "sentences.train")
eval_path = os.path.join(os.getcwd(), "data", "sentences.eval")
sentence_len = 30 # padded sentence length including EOS and BOS
vocab_size = 20000 # vocabulary size

# ------------------------------------------------------FUNCTIONS---------------------------------------------------------------------

def get_dataset(corpus):
    """ Split copus data matrix to inputs and outputs and get TF dataset representing them """
    inputs = corpus.data[:, :-1] # all columns except the last one (i,e the one containing EOS ids)
    outputs = corpus.data[:, 1:] # all columns except the first one (i,e the one containing BOS ids)
    # print(inputs)
    # print(outputs)
    return tf.data.Dataset.from_tensor_slices((inputs, outputs)).batch(BATCH_SIZE)

def compute_loss(logits, labels):
    # with tf.GradientTape() as tape:
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

def clip_gradients(grads_and_vars, clip_ratio):
    grads, vars = zip(*grads_and_vars)
    clipped, _ = tf.clip_by_global_norm(grads, clip_ratio)
    return zip(clipped, vars)

# --------------------------------------------------------START---------------------------------------------------------------------
print("\nData IO ...")
t = time()
C_train = Corpus(train_path, sentence_len, read_from_file="corpus_train.pkl", n_sentences=N_LINES) # create training corpus from training file
V_train = Vocabulary(vocab_size, read_from_file="vocab.pkl") # object to represent training vocabulary
V_train.build_from_corpus(C_train, write_to_file="vocab.pkl") # build vocabulary from training corpus
C_train.build_data_from_vocabulary(V_train, write_to_file="corpus_train.pkl") # build corpus data matrix from vocabulary, write object to disk

C_eval = Corpus(eval_path, sentence_len, read_from_file="corpus_eval.pkl", n_sentences=N_LINES) # create evaluation corpus from evaluation file
C_eval.build_data_from_vocabulary(V_train, write_to_file="corpus_eval.pkl") # build corpus data matrix from vocabulary, write object to disk
print("Total time (s): ", time() - t, "\n")

print("Vocab size: ", V_train.vocab_size)
print("Train data matrix shape: ", C_train.data.shape)
print("Eval data matrix shape: ", C_eval.data.shape)

train_ds = get_dataset(C_train)
eval_ds = get_dataset(C_eval)

print(train_ds.output_shapes)

model = LSTM(vocab_size=V_train.vocab_size, embed_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE)
optimizer = tf.train.AdamOptimizer()

for (batch, (input, labels)) in enumerate(train_ds):
    # grads_and_vars = optimizer.compute_gradients(lambda: compute_loss(logits=model(input), labels=labels))
    grads = tfe.implicit_gradients(compute_loss)
    optimizer.apply_gradients(clip_gradients(grads(model(input), labels), MAX_GRAD_NORM))
