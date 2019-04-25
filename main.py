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
batch_size = 64
num_epochs = 1 # to be chosen
eval_every = 100
n_lines = None 
# ------------------------------------------------------
train_path = os.path.join(os.getcwd(), "data", "sentences.train")
eval_path = os.path.join(os.getcwd(), "data", "sentences.eval")
embedding_path = os.path.join(os.getcwd(), "data", "wordembeddings-dim100.word2vec")
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

# Trains the model and returns perplexity values on the eval sentences
def train_model(model, num_epochs, num_batches, batched_x, batched_y, eval_every, eval_x, eval_y, V_train):
    # Training loop
    with tf.Session() as sess:
        # train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        # Initialize all variables
        # sess.run(tf.global_variables_initializer())
        if os.path.exists('./models/model.ckpt'):
            model.load_model(sess, './models/model.ckpt')
        else:
            sess.run(tf.global_variables_initializer())
        for e in range(num_epochs):
            for b in range(num_batches):
                _, step, step_loss = model.train_step(sess, batched_x[b], batched_y[b])
                time_str = datetime.datetime.now().isoformat()
                print("epoch {}, batch {}:\n{}: step {}, loss {}".format(e+1, b+1, time_str, step, step_loss))
                if step % eval_every == 0:
                    step, step_loss = model.eval_step(sess, eval_x, eval_y)
                    print("\nEvaluation:\n    {}: step {}, loss {}\n".format(time_str, step, step_loss))
        
        model.save_model(sess, './models/model.ckpt')

        return model.perplexity(sess, eval_x, eval_y, V_train)

def write_out(array, file_name): # overwrite file if exists
    n = array.shape[0]
    with open(file_name, 'w') as output:
        for row in range(n): #write each row
            output.write(str(array[row]) + '\n')

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

# Constants
vocab_size = V_train.vocab_size # get true vocab size
time_steps = sentence_len-1

# Tensorboard csts
timestamp = str(int(time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
train_summary_dir = os.path.join(out_dir, "summaries", "train")

# Models
with tf.Graph().as_default(): # create graph for Experiment A
    print("\nRunning Experiment A ...")
    # input("Press Enter to continue...")
    modelA = LSTM(V_train, embedding_size=100, hidden_size=512, time_steps=time_steps, clip_norm=clip_grad_norm)
    perp = train_model(modelA, num_epochs, num_batches, batched_x, batched_y, eval_every, eval_x, eval_y, V_train) # train and get perplexities
    write_out(perp, "group17.perplexityA")

with tf.Graph().as_default(): # create graph for Experiment B
    print("\nRunning Experiment B ...")
    # input("Press Enter to continue...")
    modelB = LSTM(V_train, embedding_size=100, hidden_size=512, time_steps=time_steps, clip_norm=clip_grad_norm, load_external_embedding=True, embedding_path=embedding_path)
    perp = train_model(modelB, num_epochs, num_batches, batched_x, batched_y, eval_every, eval_x, eval_y, V_train) # train and get perplexities
    write_out(perp, "group17.perplexityB")

with tf.Graph().as_default(): # create graph for Experiment C
    print("\nRunning Experiment C ...")
    # input("Press Enter to continue...")
    os.rmdir('./models')
    modelC = LSTM(V_train, embedding_size=100, hidden_size=1024, time_steps=time_steps, clip_norm=clip_grad_norm, down_project=True, down_projection_size=512)
    perp = train_model(modelC, num_epochs, num_batches, batched_x, batched_y, eval_every, eval_x, eval_y, V_train) # train and get perplexities
    write_out(perp, "group17.perplexityC")
