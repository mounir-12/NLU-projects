import tensorflow as tf
from tensorflow.nn.rnn_cell import LSTMCell
# from scipy.special import softmax
import numpy as np

# Can we and should we use an implementation optimized for GPU ?
# https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnLSTM


class LSTM:
    def __init__(self, vocab_size, embedding_size, hidden_size, time_steps, clip_norm):
        initializer = tf.contrib.layers.xavier_initializer()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.time_steps = time_steps
        # Model Variables
        self.weights = tf.Variable(initializer((hidden_size, vocab_size)), name="weights")
        self.biases = tf.Variable(initializer([vocab_size]), name="biases")
        self.embedding_matrix = tf.Variable(initializer((vocab_size, embedding_size)), name="embedding")
        
        # Create Model graph
        self.input_x = tf.placeholder(tf.int32, [None, time_steps]) # the input words of shape [batch_size, time_steps]
        self.input_y = tf.placeholder(tf.int32, [None, time_steps]) # the target words of shape [batch_size, time_steps]
        embedded_x = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x) # the embedded input of shape [batch_size, time_steps, embedding_size]

        self.rnn = LSTMCell(hidden_size, initializer=initializer) # LSTM cell with hidden state of size hidden_size

        self.initial_state = state = self.rnn.zero_state(tf.shape(embedded_x)[0], tf.float32) # LSTM cell initial state

        logits = None

        for t in range(time_steps):
            output, state = self.rnn(embedded_x[:, t], state) # input the slice t (i,e all embedded vectors of the batch at timestep t)
            logit = tf.matmul(output, self.weights) + self.biases
            logit = tf.reshape(logit, (tf.shape(logit)[0], 1, tf.shape(logit)[1])) # reshape for later concat

            if logits is None: # first output logit
                logits = logit
            else: # concat to previous output logits
                logits = tf.concat((logits, logit), axis=1)

        self.logits = logits
        self.final_state = state

        # Optimizer and loss
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer()
        self.loss = self.compute_loss(logits=self.logits, labels=self.input_y)
        self.clip_norm = clip_norm
        self.train_op = self.get_train_op() # training operation with clipped gradients

        self.trainable_variables = [self.weights, self.biases, self.embedding_matrix, self.rnn]

    def compute_loss(self, logits, labels):
        # with tf.GradientTape() as tape:
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    def get_train_op(self): # perform gradient clipping and return train operation
        grads_and_vars = self.optimizer.compute_gradients(self.loss) # compute gradient of loss function, returns list of tensor-variable pair
        grads, vars = zip(*grads_and_vars) # unzip
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.clip_norm) # clip
        return self.optimizer.apply_gradients(zip(clipped_grads, vars), self.global_step) # apply clipped gradients and return train op

    def train_step(self, sess, input_words, output_words):
        feed_dict = {self.input_x: input_words, self.input_y: output_words}
        fetches = [self.train_op, self.global_step, self.loss] # train and report loss

        return sess.run(fetches, feed_dict)
    
    def eval_step(self, sess, input_words, output_words):
        feed_dict = {self.input_x: input_words, self.input_y: output_words}
        fetches = [self.global_step, self.loss] # only report loss

        return sess.run(fetches, feed_dict)

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

    def __call__(self, sess, x, y):
        return self.train_step(sess, x, y)
