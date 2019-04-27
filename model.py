import tensorflow as tf
from tensorflow.nn.rnn_cell import LSTMCell as Cell
import numpy as np
from load_embedding import load_embedding

# Can we and should we use an implementation optimized for GPU ?
# https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnLSTM


class LSTM:
    def __init__(self, V, embedding_size, hidden_size, time_steps, clip_norm, down_project=False, down_projection_size=None,
                        load_external_embedding=False, embedding_path=None):
        vocab_size = V.vocab_size
        self.dtype = tf.float32
        initializer = tf.contrib.layers.xavier_initializer(dtype=self.dtype)

        self.hidden_size = hidden_size
        self.vocab_size = V.vocab_size
        self.time_steps = time_steps
        self.down_project = down_project

        # Model Variables
        if down_project:
            self.W_p = tf.Variable(initializer((hidden_size, down_projection_size)), name="W_p", dtype=self.dtype) # projection matrix
            self.biases_p = tf.Variable(initializer([down_projection_size]), name="biases_p", dtype=self.dtype)
            self.W = tf.Variable(initializer((down_projection_size, vocab_size)), name="W", dtype=self.dtype) # weights
        else:
            self.W = tf.Variable(initializer((hidden_size, vocab_size)), name="W", dtype=self.dtype) # weights

        self.biases = tf.Variable(initializer([vocab_size]), name="biases", dtype=self.dtype)

        if load_external_embedding: # we're loading the embedding matrix
            matrix = load_embedding(V.token2id, embedding_path, embedding_size, vocab_size) # load embedding matrix from file
            self.embedding_matrix = tf.Variable(matrix, name="embedding", dtype=self.dtype) # make it trainable too (further train embeddings)
        else: # not loading external embedding => make embedding_matrix a tf.Variable (i,e to be trainable)
            self.embedding_matrix = tf.Variable(initializer((vocab_size, embedding_size)), name="embedding", dtype=self.dtype)
        
        # Create Model graph
        self.input_x = tf.placeholder(tf.int32, [None, time_steps]) # the input words of shape [batch_size, time_steps]
        self.input_y = tf.placeholder(tf.int32, [None, time_steps]) # the target words of shape [batch_size, time_steps]
        embedded_x = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x) # the embedded input of shape [batch_size, time_steps, embedding_size]

        self.rnn = Cell(num_units=hidden_size, initializer=initializer, dtype=self.dtype, name="cell") # LSTM cell with hidden state of size hidden_size

        self.initial_state = state = self.rnn.zero_state(tf.shape(embedded_x)[0], dtype=self.dtype) # LSTM cell initial state

        logits = None

        for t in range(time_steps):
            output, state = self.rnn(embedded_x[:, t], state) # input the slice t (i,e all embedded vectors of the batch at timestep t)
            if down_project:
                logit = tf.matmul(tf.matmul(output, self.W_p) + self.biases_p, self.W) + self.biases # first project then compute logit
            else:
                logit = tf.matmul(output, self.W) + self.biases
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

        self.trainable_variables = [self.W, self.biases, self.embedding_matrix, self.rnn]
        if down_project:
            self.trainable_variables.append(self.W_p)
        self.saver = tf.train.Saver()

    def compute_loss(self, logits, labels):
        # with tf.GradientTape() as tape:
        self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return tf.reduce_sum(self.losses)

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

    def perplexity(self, sess, input_sentences, output_sentences, V):
        losses_vals = sess.run(self.losses, feed_dict={self.input_x: input_sentences, self.input_y: input_sentences})
        
        s = output_sentences.shape[0] # nb of sentences
        perp = np.zeros(s)
        pad_id = V.token2id[V.PAD_token] # the id of the PAD token

        for i in range(s): # loop over output sentences i, we consider output sentences since <EOS> is used in perplexity computation while
                           # <BOS> isn't (since model shouldn't predict <BOS>)
            indices = np.where(output_sentences[i] != pad_id) # retrieve indices in the output sentence where the token is not PAD
            exponent = np.mean(losses_vals[indices]) # compute the mean, only sum the losses vals at the specified indices
            perp[i] = np.exp(exponent) # use exp() since the losses are computed using Napierian logarithm ln()

        return perp

    def build_sentence_completion_graph(self):

        self.c_in = tf.placeholder(tf.float32, shape=[1, self.rnn.state_size.c], name='c_in')
        self.h_in = tf.placeholder(tf.float32, shape=[1, self.rnn.state_size.h], name='h_in')

        self.int_state = tf.contrib.rnn.LSTMStateTuple(self.c_in, self.h_in) # internal state
        self.current_word = tf.placeholder(tf.int32, [1])
        self.current_embed = tf.nn.embedding_lookup(self.embedding_matrix, self.current_word)

        next_output, self.next_state = self.rnn(self.current_embed, self.int_state)

        if self.down_project:
            next_logit = tf.matmul(tf.matmul(next_output, self.W_p) + self.biases_p, self.W) + self.biases
        else:
            next_logit = tf.matmul(next_output, self.W) + self.biases

        self.next_probabs = tf.nn.softmax(next_logit)


    def sentence_continuation(self, sess, prompt, V, max_length):
        # prompt: prefix to be completed
        c_init = np.zeros((1, self.rnn.state_size.c), np.float32)
        h_init = np.zeros((1, self.rnn.state_size.h), np.float32)
        current_state = [c_init, h_init]
        prompt_length = len(prompt)
        for word in prompt:
            current_c, current_h = current_state
            feed_dict = {self.c_in: current_c, self.h_in: current_h, self.current_word: [word]}
            next_probabs, current_state = sess.run([self.next_probabs, self.next_state], feed_dict)

        current_word = np.argmax(next_probabs)


        predicted_continuation = []

        for i in range(max_length-prompt_length):
            predicted_continuation.append(V.id2token[current_word])

            current_c, current_h = current_state

            feed_dict = {self.c_in: current_c, self.h_in: current_h, self.current_word: [current_word]}

            next_probabs, current_state = sess.run([self.next_probabs, self.next_state], feed_dict)
            current_word = np.argmax(next_probabs)
            if(current_word == V.token2id[V.EOS_token]):
                break

        return predicted_continuation


    def save_model(self, sess, path):
        self.saver.save(sess, path)
        print("Model saved at %s" %path)

    def load_model(self, sess, path):
        try: # try to load the model
            self.saver.restore(sess, path)
            print("Model restored from %s" %path)
            return True
        except ValueError:
            print("Couldn't restore model")
            return False

