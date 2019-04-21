import tensorflow as tf
from tensorflow.nn.rnn_cell import LSTMCell

# Can we and should we use an implementation optimized for GPU ?
# https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnLSTM


class LSTM:
    def __init__(self, vocab_size, embed_size, hidden_size):
        initializer = tf.contrib.layers.xavier_initializer()

        self.hidden_size = hidden_size

        self.weights = tf.Variable(initializer((hidden_size, vocab_size)))
        self.biases = tf.Variable(initializer([vocab_size]))
        self.embed_matrix = tf.Variable(initializer((vocab_size, embed_size)))

        self.rnn = LSTMCell(hidden_size, initializer=initializer)

    def forward(self, x):

        # generate prediction
        tf.zeros(self.hidden_size)
        embedded = tf.nn.embedding_lookup(self.embed_matrix, x)
        print(embedded.shape)
        outputs, states = self.rnn(embedded, tf.zeros(self.hidden_size))  # TODO Is this the right initialization for the hidden state ?

        # there are n_input outputs but
        # we only want the last output
        return tf.matmul(outputs[-1], self.weights) + self.biases

    def __call__(self, x):
        self.forward(x)
