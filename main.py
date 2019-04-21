import os

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from lang import Lang
from model import LSTM

tfe.enable_eager_execution()
tf.set_random_seed(9)

MAX_LEN = 28  # 30 - 2 (BOS and EOS)
BATCH_SIZE = 64
BATCH_SIZE = 2  # For local dev
EMBEDDING_SIZE = 100
EMBEDDING_SIZE = 10  # For local dev
HIDDEN_SIZE = 512
HIDDEN_SIZE = 128  # For local dev
N_LINES = None  # Number of training lines to read
N_LINES = 20  # For local dev
MAX_GRAD_NORM = 5


def tokenize(path, lang, max_len, nlines=None):
    all_tokens = []

    with open(path) as in_file:
        for i, line in enumerate(in_file):
            tokens = lang.tokenize(line)
            if len(tokens) < max_len:
                all_tokens.append(tokens)

            if nlines and i == nlines:
                break

    return all_tokens


def build_lang(lang, all_tokens):
    for tokens in all_tokens:
        lang.add_tokens(tokens)

    lang.build()


def get_dataset(lang, all_tokens):
    all_embedded = [None] * len(all_tokens)
    all_targets = [None] * len(all_tokens)
    for i, tokens in enumerate(all_tokens):
        in_tokens = lang.numericalize(tokens, MAX_LEN)
        out_tokens = list(in_tokens)

        in_tokens.insert(0, lang.BOS_token)
        out_tokens.append(lang.EOS_token)  # Input shifted in time by one and having EOS

        all_embedded[i] = in_tokens
        all_targets[i] = out_tokens

    return tf.data.Dataset.from_tensor_slices((all_embedded, all_targets)).batch(BATCH_SIZE)


lang = Lang()
train_tokens = tokenize(os.path.join('data', 'sentences.train'), lang, MAX_LEN, N_LINES)
eval_tokens = tokenize(os.path.join('data', 'sentences.eval'), lang, MAX_LEN, N_LINES)

build_lang(lang, train_tokens)

train_ds = get_dataset(lang, train_tokens)
eval_ds = get_dataset(lang, eval_tokens)

print(train_ds.output_shapes)


def compute_loss(logits, labels):
    # with tf.GradientTape() as tape:
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))


model = LSTM(vocab_size=len(lang), embed_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE)

optimizer = tf.train.AdamOptimizer()


def clip_gradients(grads_and_vars, clip_ratio):
    grads, vars = zip(*grads_and_vars)
    clipped, _ = tf.clip_by_global_norm(grads, clip_ratio)
    return zip(clipped, vars)


for (batch, (input, labels)) in enumerate(train_ds):
    # grads_and_vars = optimizer.compute_gradients(lambda: compute_loss(logits=model(input), labels=labels))
    grads = tfe.implicit_gradients(compute_loss)
    optimizer.apply_gradients(clip_gradients(grads(model(input), labels), MAX_GRAD_NORM))
