import tensorflow as tf
from tensorflow.nn.rnn_cell import LSTMCell
from scipy.special import softmax
import numpy as np

# Can we and should we use an implementation optimized for GPU ?
# https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnLSTM


class LSTM:
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size):
        initializer = tf.contrib.layers.xavier_initializer()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size

        self.weights = tf.Variable(initializer((hidden_size, vocab_size)), name="weights")
        self.biases = tf.Variable(initializer([vocab_size]), name="biases")
        self.embed_matrix = tf.Variable(initializer((vocab_size, embed_size)), name="embed")

        words = tf.placeholder(tf.int32, [batch_size, time_steps])


        self.rnn = LSTMCell(hidden_size, initializer=initializer)

        self.initial_state = state = self.rnn.zero_state(batch_size, tf.float32)

        self.logits = []

        for i in range(time_steps):
            embedded = tf.nn.embedding_lookup(self.embed_matrix, words[:,i])
            output, state = self.rnn(embedded, state)
            logits.append(tf.matmul(output, self.weights) + self.biases)

        self.final_state = state

        self.trainable_variables = [self.weights, self.biases, self.embed_matrix, self.rnn]

    def forward(self, sess, input_words):
        
        logits = sess.run(self.logits, feed_dict={words: input_words})

        return logits

    def preplexity(self, sess, input_sentences):

        logits = sess.run(self.logits, feed_dict={words: input_sentences})

        probabs = softmax(logits, axis=1)

        perp = np.zeros(input_sentences.shape[0])

        for i in range(input_sentences.shape[0]):
            temp = 1
            for j in range(input_sentences.shape[1]):
                temp*=probabs[i,input_sentences[i,j]]
            perp[i] = (1/temp)**(1/input_sentences.shape[1])

        return perp


    def __call__(self, x):
        return self.forward(x)
