import tensorflow as tf
from tensorflow.nn.rnn_cell import LSTMCell

# Can we and should we use an implementation optimized for GPU ?
# https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnLSTM


class LSTM:
    def __init__(self, vocab_size, embed_size, hidden_size):
        initializer = tf.contrib.layers.xavier_initializer()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.weights = tf.Variable(initializer((hidden_size, vocab_size)))
        self.biases = tf.Variable(initializer([vocab_size]))
        self.embed_matrix = tf.Variable(initializer((vocab_size, embed_size)))

        self.rnn = LSTMCell(hidden_size, initializer=initializer)

    def forward(self, x):
        embedded = tf.nn.embedding_lookup(self.embed_matrix, x)

        logits = tf.Variable(tf.zeros((embedded.shape[0], embedded.shape[1], self.vocab_size)))  # TODO Initialize with empty tensor
        states = (tf.zeros((embedded.shape[0], self.hidden_size)), tf.zeros((embedded.shape[0], self.hidden_size)))  # TODO Is this the right initialization for the hidden state ?

        for i in range(embedded.shape[1]):  # TODO Use tf.while_loop for compatibility with graph mode ?
            output, states = self.rnn(embedded[:, i, :], states)
            logits[:, i, :].assign(tf.matmul(output, self.weights) + self.biases)

        # TODO Use the last output instead ?
        return logits

    def __call__(self, x):
        return self.forward(x)
