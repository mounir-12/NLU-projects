import os

import tensorflow as tf

from lang import Lang

lang = Lang()

with open(os.path.join('data', 'sentences.train')) as in_file:
    i = 0
    for l in in_file:
        lang.add_sentence(l)

lang.build()

hidden_size = 512

weights = {
    'out': tf.Variable(tf.random_normal([hidden_size, lang.n_tokens]))
}
biases = {
    'out': tf.Variable(tf.random_normal([lang.n_tokens]))
}
