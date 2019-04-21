import os

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from lang import Lang

tfe.enable_eager_execution()
tf.set_random_seed(9)

MAX_LEN = 28  # 30 - 2 (BOS and EOS)
BATCH_SIZE = 64
N_LINES = 20  # Number of training lines to read

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

for i, tokens in enumerate(all_tokens):
    all_embedded[i] = lang.embed(tokens)


# print("\n".join(str(e) for e in all_tokens))
# print("\n".join(str(e) for e in all_embedded))

train = tf.data.Dataset.from_tensor_slices(all_embedded).batch(BATCH_SIZE)

print(train.output_shapes)

hidden_size = 512

weights = {
    'out': tf.Variable(tf.random_normal([hidden_size, lang.n_tokens]))
}
biases = {
    'out': tf.Variable(tf.random_normal([lang.n_tokens]))
}
