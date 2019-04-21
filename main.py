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

lang = Lang()

all_tokens = []

with open(os.path.join('data', 'sentences.train')) as in_file:
    i = 0
    for i, l in enumerate(in_file):
        tokens = lang.tokenize(l)
        if len(tokens) < MAX_LEN:
            lang.add_tokens(tokens)
            all_tokens.append(tokens)

        if N_LINES and i == N_LINES:
            break

lang.build()

all_embedded = [None] * len(all_tokens)
all_targets = [None] * len(all_tokens)

for i, tokens in enumerate(all_tokens):
    in_tokens = lang.numericalize(tokens, MAX_LEN)
    out_tokens = list(in_tokens)

    in_tokens.insert(0, lang.BOS_token)
    out_tokens.append(lang.EOS_token)  # Input shifted in time by one and having EOS

    all_embedded[i] = in_tokens
    all_targets[i] = out_tokens


# print("\n".join(str(e) for e in all_tokens))
# print("\n".join(str(e) for e in all_embedded))

train = tf.data.Dataset.from_tensor_slices((all_embedded, all_targets)).batch(BATCH_SIZE)

print(train.output_shapes)


def compute_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))


model = LSTM(vocab_size=len(lang), embed_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE)

optimizer = tf.train.AdamOptimizer()

for (batch, (input, labels)) in enumerate(train):
    optimizer.minimize(lambda: compute_loss(logits=model(input), labels=labels))

