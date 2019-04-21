import tensorflow as tf
from tensorflow.nn.rnn_cell import LSTMCell

# Can we and should we use an implementation optimized for GPU ?
# https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnLSTM


class LSTM:
    def __init__(self, vocab_size, embed_size, hidden_size):
        initializer = tf.contrib.layers.xavier_initializer()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.weights = tf.Variable(initializer((hidden_size, vocab_size)), name="weights")
        self.biases = tf.Variable(initializer([vocab_size]), name="biases")
        self.embed_matrix = tf.Variable(initializer((vocab_size, embed_size)), name="embed")

        self.rnn = LSTMCell(hidden_size, initializer=initializer)

        self.trainable_variables = [self.weights, self.biases, self.embed_matrix, self.rnn]

    def forward(self, x):
        embedded = tf.nn.embedding_lookup(self.embed_matrix, x)

        states = self.rnn.zero_state(embedded.shape[0], tf.float32)

        """
        embed_seq = tf.unstack(embedded, axis=1)
        logits = []
        for el in embed_seq:
            output, states = self.rnn(el, states)
            logits.append(tf.matmul(output, self.weights) + self.biases)

        logits = tf.stack(logits, axis=1)
        """

        logits = tf.Variable(
            tf.zeros((embedded.shape[0], embedded.shape[1], self.vocab_size)))  # TODO Initialize with empty tensor
        for i in range(embedded.shape[1]):  # TODO Use tf.while_loop for compatibility with graph mode ?
            output, states = self.rnn(embedded[:, i, :], states)
            logits[:, i, :].assign(tf.matmul(output, self.weights) + self.biases)

        return logits

    def __call__(self, x):
        return self.forward(x)
