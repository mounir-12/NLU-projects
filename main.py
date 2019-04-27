import datetime
import os
import sys
from argparse import ArgumentParser
from time import time

import math
import numpy as np
import tensorflow as tf

from lang_new import Vocabulary, Corpus
from model import LSTM

parser = ArgumentParser()
parser.add_argument("-l", "--lines", type=int,
                    help="Number of lines to read. If not provided, reads the whole file")

parser.add_argument("-bs", "--batch-size", type=int, default=64,
                    help="Train batch size")
parser.add_argument("-e", "--epochs", type=int, default=1,
                    help="Training epochs")

parser.add_argument("-pe", "--print-every", type=int, default=100,
                    help="Number of steps between info printing")
parser.add_argument("-ve", "--val-every", type=int, default=1000,
                    help="Number of steps between evaluation metric computation")
parser.add_argument("-se", "--save-every", type=int, default=1000,
                    help="Number of steps between model checkpoint saving")

parser.add_argument('-t', "--task", choices=["1a", "1b", "1c", "2"], required=True,
                    help="The task to run")

parser.add_argument("-rs", "--random-seed", type=int, default=9,
                    help="Random seed")


args = parser.parse_args()

tf.set_random_seed(args.random_seed)
np.random.seed(args.random_seed)

# --------------------------------------------------------CONSTANTS---------------------------------------------------------------------
# ------values in comment for cluster deployement-------
batch_size = args.batch_size
num_epochs = args.epochs
eval_every = args.val_every
print_every = args.print_every  # limit the number of prints during training
save_every = args.save_every
n_lines = args.lines
max_length = 20  # maximum number of words per sentence during completion
# ------------------------------------------------------
task = args.task

train_path = os.path.join(os.getcwd(), "data", "sentences.train")
eval_path = os.path.join(os.getcwd(), "data", "sentences.eval")
test_path = os.path.join(os.getcwd(), "data", "sentences_test.txt")
embedding_path = os.path.join(os.getcwd(), "data", "wordembeddings-dim100.word2vec")
prompt_path = os.path.join(os.getcwd(), "data", "sentences.continuation")

sentence_len = 30  # padded sentence length including EOS and BOS
vocab_size = 20000  # vocabulary size
clip_grad_norm = 5
model_task2 = "modelC.ckpt"


# ------------------------------------------------------FUNCTIONS---------------------------------------------------------------------

def get_data(corpus, shuffle=False, batch_size=None):
    data = corpus.data
    data_size = data.shape[0]
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        data = data[shuffle_indices]
    x = data[:, :-1]
    y = data[:, 1:]
    num_batches = 1  # default value
    if batch_size is not None:
        num_batches = int(math.ceil(data_size / batch_size))
    x = np.array_split(x, num_batches)  # split into num_batches batches (last batch size may differ from previous ones)
    y = np.array_split(y, num_batches)  # split into num_batches batches (last batch size may differ from previous ones)
    return np.array(x), np.array(y)


# Trains the model and returns perplexity values on the eval sentences
def train_model(model, sess, num_epochs, train_x_batched, train_y_batched, eval_x_batched, eval_y_batched,
                model_ckpt_name="model.ckpt", eval_every=-1):
    # Training loop
    models_dir = os.path.join(os.getcwd(), "models")
    model_path = os.path.join(models_dir, model_ckpt_name)  # path of file to save model
    num_batches_train = train_x_batched.shape[0]  # nb of batches in the training set
    num_batches_eval = eval_x_batched.shape[0]  # nb of batches in the eval set

    # train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    if os.path.exists(models_dir) and model.load_model(sess, model_path):  # if successfully loaded model
        step, step_loss = eval_model(model, sess, eval_x_batched, eval_y_batched, num_batches_eval)
        perps = get_perplexity(model, sess, eval_x_batched, eval_y_batched,
                                       V_train)  # compute perplexities over eval dataset
        mean, median = np.mean(perps), np.median(perps)
        print(
            "\nEvaluating restored model on eval dataset:\n   batches: {}, step {}, loss {}, perp_mean: {}, perp_median {}\n"
            .format(num_batches_eval, step, step_loss, mean, median))
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
                print("epoch {}, batch {}:\n{}: step {}, loss {}".format(e + 1, b + 1, time_str, step, step_loss))
            if step % save_every == 0:
                model.save_model(sess, model_path)  # save trained model
            if eval_every > 0 and step % eval_every == 0:  # do not evaluate if eval_every <= 0
                perps = get_perplexity(model, sess, eval_x_batched, eval_y_batched,
                                       V_train)  # compute perplexities over eval dataset
                mean, median = np.mean(perps), np.median(perps)
                print("\nEvaluation:\n    batches: {}, step: {}, perp_mean: {}, perp_median: {}\n".format(
                    num_batches_eval, step, mean, median))

    model.save_model(sess, model_path)  # save trained model


def eval_model(model, sess, eval_x_batched, eval_y_batched, num_batches):
    sum = 0
    counter = 0
    for i in range(num_batches):  # compute the loss over each batch
        step, step_loss = model.eval_step(sess, eval_x_batched[i], eval_y_batched[i])
        if (not np.isnan(step_loss)):
            sum += step_loss
            counter += 1
    step_loss = sum / counter  # compute the mean loss over all batches
    return step, step_loss


def get_perplexity(model, sess, test_x_batched, test_y_batched, V_train):
    num_batches = test_x_batched.shape[0]
    perp = None
    for i in range(num_batches):
        batch_perp = model.perplexity(sess, test_x_batched[i], test_y_batched[i], V_train)
        if perp is None:
            perp = batch_perp
        else:
            perp = np.concatenate((perp, batch_perp))
    return perp


def write_out(array, file_name):  # overwrite file if exists
    n = array.shape[0]
    with open(file_name, 'w') as output:
        for row in range(n):  # write each row
            output.write(str(array[row]) + '\n')
    
# --------------------------------------------------------START---------------------------------------------------------------------
# Data IO
print("\nData IO ...")
t = time()
C_train = Corpus(train_path, sentence_len, read_from_file="corpus_train.pkl",
                 n_sentences=n_lines)  # create training corpus from training file
V_train = Vocabulary(vocab_size, read_from_file="vocab.pkl")  # object to represent training vocabulary
V_train.build_from_corpus(C_train, write_to_file="vocab.pkl")  # build vocabulary from training corpus
C_train.build_data_from_vocabulary(V_train,
                                   write_to_file="corpus_train.pkl")  # build corpus data matrix from vocabulary, write object to disk

C_eval = Corpus(eval_path, sentence_len, read_from_file="corpus_eval.pkl",
                n_sentences=n_lines)  # create evaluation corpus from evaluation file
C_eval.build_data_from_vocabulary(V_train,
                                  write_to_file="corpus_eval.pkl")  # build corpus data matrix from vocabulary, write object to disk

C_test = Corpus(test_path, sentence_len, read_from_file="corpus_test.pkl",
                n_sentences=n_lines)  # create test corpus from test file
C_test.build_data_from_vocabulary(V_train,
                                  write_to_file="corpus_test.pkl")  # build corpus data matrix from vocabulary, write object to disk

print("Total time (s): ", time() - t, "\n")

print("Vocab size: ", V_train.vocab_size)
print("Train data matrix shape: ", C_train.data.shape)
print("Eval data matrix shape: ", C_eval.data.shape)

# Train and Eval data
train_x_batched, train_y_batched = get_data(C_train, shuffle=True,
                                            batch_size=batch_size)  # training data, shuffled, batched
eval_x_batched, eval_y_batched = get_data(C_eval, batch_size=batch_size)  # eval data batched no shuffling
test_x_batched, test_y_batched = get_data(C_test,
                                          batch_size=batch_size)  # test data batched no shuffling (used for perplexity computation)

# Constants
vocab_size = V_train.vocab_size  # get true vocab size
time_steps = sentence_len - 1

# Tensorboard constants
timestamp = str(int(time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
train_summary_dir = os.path.join(out_dir, "summaries", "train")

# Models
if task == "1a":
    with tf.Graph().as_default():  # create graph for Experiment A
        with tf.Session() as sess:
            print("\nRunning Experiment A ...")
            # input("Press Enter to continue...")
            modelA = LSTM(V_train, embedding_size=100, hidden_size=512, time_steps=time_steps, clip_norm=clip_grad_norm)
            train_model(modelA, sess, num_epochs, train_x_batched, train_y_batched, eval_x_batched, eval_y_batched,
                        model_ckpt_name="modelA.ckpt", eval_every=eval_every)
            perp = get_perplexity(modelA, sess, test_x_batched, test_y_batched,
                                  V_train)  # compute perplexities on test set
            write_out(perp, "group17.perplexityA")
elif task == "1b":
    with tf.Graph().as_default():  # create graph for Experiment B
        with tf.Session() as sess:
            print("\nRunning Experiment B ...")
            # input("Press Enter to continue...")
            modelB = LSTM(V_train, embedding_size=100, hidden_size=512, time_steps=time_steps, clip_norm=clip_grad_norm,
                          load_external_embedding=True, embedding_path=embedding_path)
            train_model(modelB, sess, num_epochs, train_x_batched, train_y_batched, eval_x_batched, eval_y_batched,
                        model_ckpt_name="modelB.ckpt", eval_every=eval_every)
            perp = get_perplexity(modelB, sess, test_x_batched, test_y_batched,
                                  V_train)  # compute perplexities on test set
            write_out(perp, "group17.perplexityB")
elif task == "1c":
    with tf.Graph().as_default():  # create graph for Experiment C
        with tf.Session() as sess:
            print("\nRunning Experiment C ...")
            # input("Press Enter to continue...")
            modelC = LSTM(V_train, embedding_size=100, hidden_size=1024, time_steps=time_steps,
                          clip_norm=clip_grad_norm, down_project=True, down_projection_size=512)
            train_model(modelC, sess, num_epochs, train_x_batched, train_y_batched, eval_x_batched, eval_y_batched,
                        model_ckpt_name="modelC.ckpt", eval_every=eval_every)
            perp = get_perplexity(modelC, sess, test_x_batched, test_y_batched,
                                  V_train)  # compute perplexities on test set
            write_out(perp, "group17.perplexityC")
elif task == "2":
    with tf.Graph().as_default():  # create graph for task 2
        print("\nRunning Task 2 ...")
        # Read sentences to complete
        sentences = []
        with open(prompt_path) as sentences_file:
            for sentence in sentences_file:
                tokens = sentence.strip().split(" ")
                sentences.append(tokens)

        # Pick the model to use
        models_dir = os.path.join(os.getcwd(), "models")
        model_path = os.path.join(models_dir, model_task2)
        if model_task2 == "modelA.ckpt":
            model2 = LSTM(V_train, embedding_size=100, hidden_size=512, time_steps=time_steps, clip_norm=clip_grad_norm)
        elif model_task2 == "modelB.ckpt":
            model2 = LSTM(V_train, embedding_size=100, hidden_size=512, time_steps=time_steps, clip_norm=clip_grad_norm,
                          load_external_embedding=True, embedding_path=embedding_path)
        elif model_task2 == "modelC.ckpt":
            model2 = LSTM(V_train, embedding_size=100, hidden_size=1024, time_steps=time_steps,
                          clip_norm=clip_grad_norm, down_project=True, down_projection_size=512)
        
        with tf.Session() as sess:
            # Load model
            if not os.path.exists(models_dir) or not model2.load_model(sess, model_path):  # if couldn't load model
                print("Please train the model first.")
                sys.exit(-1)

            # Complete sentences
            model2.build_sentence_completion_graph()
            print("\nSentence Completion Graph Built")
            completed_sentences = []
            nb_completed = 0
            for sentence in sentences: # for each tokenized sentence
                prompt = V_train.get_tokens_ids(sentence)
                continuation = model2.sentence_continuation(sess, prompt, V_train, max_length)
                nb_completed += 1
                if nb_completed == 1 or nb_completed % print_every == 0:
                    print("Completed {} sentence".format(nb_completed))
                completed_sentences.append(sentence + continuation)

        # Write out completed sentences
        write_path = os.path.join(os.getcwd(), "group17.continuation")
        print("\nWriting Completed Sentences")
        with open(write_path, "w") as file:
            for sentence in completed_sentences:
                for word in sentence:
                    if(word == V_train.PAD_token or word == V_train.BOS_token):
                        continue
                    file.write(word + ' ')
                file.write('\n')
else:
    raise ValueError("Wut unrecognized task {}".format(task))
