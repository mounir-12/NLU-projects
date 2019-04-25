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
eval_every = 1000
print_every = 100 # limit the number of prints during training
n_lines = None 
# ------------------------------------------------------
train_path = os.path.join(os.getcwd(), "data", "sentences.train")
eval_path = os.path.join(os.getcwd(), "data", "sentences.eval")
test_path = os.path.join(os.getcwd(), "data", "sentences_test.txt")
embedding_path = os.path.join(os.getcwd(), "data", "wordembeddings-dim100.word2vec")
sentence_len = 30 # padded sentence length including EOS and BOS
vocab_size = 20000 # vocabulary size
clip_grad_norm = 5
# ------------------------------------------------------FUNCTIONS---------------------------------------------------------------------

def get_data(corpus, shuffle=False, batch_size=None):
    data = corpus.data
    data_size = data.shape[0]
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        data = data[shuffle_indices]
    x = data[:, :-1]
    y = data[:, 1:]
    num_batches = 1 # default value
    if batch_size is not None:
        num_batches = int(math.ceil(data_size/batch_size))
    x = np.array_split(x, num_batches) # split into num_batches batches (last batch size may differ from previous ones)
    y = np.array_split(y, num_batches) # split into num_batches batches (last batch size may differ from previous ones)
    return np.array(x), np.array(y)

# Trains the model and returns perplexity values on the eval sentences
def train_model(model, sess, num_epochs, train_x_batched, train_y_batched, eval_x_batched, eval_y_batched, model_ckpt_name="model.ckpt", eval_every=-1):
    # Training loop
    models_dir = os.path.join(os.getcwd(), "models")
    model_path = os.path.join(models_dir, model_ckpt_name) # path of file to save model
    num_batches_train = train_x_batched.shape[0] # nb of batches in the training set
    num_batches_eval = eval_x_batched.shape[0] # nb of batches in the eval set
    
    # train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    if os.path.exists(models_dir) and model.load_model(sess, model_path): # if successfully loaded model
        step, step_loss = eval_model(model, sess, eval_x_batched, eval_y_batched, num_batches_eval)
        print("\nEvaluating restored model on eval dataset:\n   batches: {}, step {}, loss {}\n".format(num_batches_eval, step, step_loss))
        return

    # otherwise, train model and save
    print("Training model ...")
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    for e in range(num_epochs):
        for b in range(num_batches_train):
            _, step, step_loss = model.train_step(sess, train_x_batched[b], train_y_batched[b])
            if step == 1 or step % print_every == 0:
                time_str = datetime.datetime.now().isoformat()
                print("epoch {}, batch {}:\n{}: step {}, loss {}".format(e+1, b+1, time_str, step, step_loss))
            if eval_every > 0 and step % eval_every == 0: # do not evaluate if eval_every <= 0
                step, step_loss = eval_model(model, sess, eval_x_batched, eval_y_batched, num_batches_eval)                   
                print("\nEvaluation:\n    batches: {}, step {}, loss {}\n".format(num_batches_eval, step, step_loss))
    
    model.save_model(sess, model_path) # save trained model

def eval_model(model, sess, eval_x_batched, eval_y_batched, num_batches):
    sum = 0
    counter = 0
    for i in range(num_batches): # compute the loss over each batch
        step, step_loss = model.eval_step(sess, eval_x_batched[i], eval_y_batched[i])
        if(not np.isnan(step_loss)):
            sum += step_loss
            counter += 1
    step_loss = sum/counter # compute the mean loss over all batches
    return step, step_loss

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

C_test = Corpus(test_path, sentence_len, read_from_file="corpus_test.pkl", n_sentences=n_lines) # create test corpus from test file
C_test.build_data_from_vocabulary(V_train, write_to_file="corpus_test.pkl") # build corpus data matrix from vocabulary, write object to disk

print("Total time (s): ", time() - t, "\n")

print("Vocab size: ", V_train.vocab_size)
print("Train data matrix shape: ", C_train.data.shape)
print("Eval data matrix shape: ", C_eval.data.shape)

# Train and Eval data
train_x_batched, train_y_batched = get_data(C_train, shuffle=True, batch_size=batch_size) # training data, shuffled, batched
eval_x_batched, eval_y_batched = get_data(C_eval, batch_size=batch_size) # eval data no shuffling or batching
test_x = C_test.data[:, :-1] # unbatched data to compute perplexity
test_y = C_test.data[:, 1:] # unbatched data to compute perplexity

# Constants
vocab_size = V_train.vocab_size # get true vocab size
time_steps = sentence_len-1

# Tensorboard csts
timestamp = str(int(time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
train_summary_dir = os.path.join(out_dir, "summaries", "train")
evaluate = True # toggle evaluation step

# Models
with tf.Graph().as_default(): # create graph for Experiment A
    with tf.Session() as sess:
        print("\nRunning Experiment A ...")
        # input("Press Enter to continue...")
        modelA = LSTM(V_train, embedding_size=100, hidden_size=512, time_steps=time_steps, clip_norm=clip_grad_norm)
        train_model(modelA, sess, num_epochs, train_x_batched, train_y_batched, eval_x_batched, eval_y_batched, model_ckpt_name="modelA.ckpt", eval_every=eval_every)
        perp = modelA.perplexity(sess, test_x, test_y, V_train) # compute perplexities on test set
        write_out(perp, "group17.perplexityA")

with tf.Graph().as_default(): # create graph for Experiment B
    with tf.Session() as sess:
        print("\nRunning Experiment B ...")
        # input("Press Enter to continue...")
        modelB = LSTM(V_train, embedding_size=100, hidden_size=512, time_steps=time_steps, clip_norm=clip_grad_norm, load_external_embedding=True, embedding_path=embedding_path)
        train_model(modelB, sess, num_epochs, train_x_batched, train_y_batched, eval_x_batched, eval_y_batched, model_ckpt_name="modelB.ckpt", eval_every=eval_every)
        perp = modelB.perplexity(sess, test_x, test_y, V_train) # compute perplexities on test set
        write_out(perp, "group17.perplexityB")

with tf.Graph().as_default(): # create graph for Experiment C
    with tf.Session() as sess:
        print("\nRunning Experiment C ...")
        # input("Press Enter to continue...")
        modelC = LSTM(V_train, embedding_size=100, hidden_size=1024, time_steps=time_steps, clip_norm=clip_grad_norm, down_project=True, down_projection_size=512)
        train_model(modelC, sess, num_epochs, train_x_batched, train_y_batched, eval_x_batched, eval_y_batched, model_ckpt_name="modelC.ckpt", eval_every=eval_every)
        perp = modelC.perplexity(sess, test_x, test_y, V_train) # compute perplexities on test set
        write_out(perp, "group17.perplexityC")

