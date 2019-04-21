import tensorflow as tf
from tensorflow.nn.rnn_cell import LSTMCell
from tensorflow.nn import rnn

# Can we and should we use an implementation optimized for GPU ?
# https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnLSTM

def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x, n_input, 1)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = LSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']