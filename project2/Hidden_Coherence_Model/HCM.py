
# coding: utf-8

# In[1]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
tf.set_random_seed(7)
np.random.seed(7)
import time
from tqdm import tqdm
import pickle
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from collections import Counter
import os
from tensorflow.contrib.rnn import LSTMCell as Cell
import sys


# In[2]:


nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


# In[3]:

print("Loading Data...\n")

train = pd.read_csv('data/train_stories.csv')
val = pd.read_csv('data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv')
test = pd.read_csv('data/cloze_test_test__spring2016 - cloze_test_ALL_test.csv')

train = train.drop("storytitle", axis=1).drop("storyid", axis=1)

val = val.drop("InputStoryid", axis=1)
val_answer = val["AnswerRightEnding"]
val_sentences = val.drop("AnswerRightEnding", axis=1)

test = test.drop("InputStoryid", axis=1)
test_answer = test["AnswerRightEnding"]
test_sentences = test.drop("AnswerRightEnding", axis=1)


# In[4]:


def get_stories_as_lists(dataframe):
    a = time.time()
    print("Reading stories to memory ...")
    stories = []
    stories_flat = []
    for index, row in dataframe.iterrows():
        story = []
        for col in dataframe.columns:
            story.append(word_tokenize(row[col]))
        stories.append(story)
    print("Done in {} s".format(time.time() - a))
    return stories

def sentence_sentiment(sentence, sentiment_analyzer):
    n_positive = 0
    n_negative = 0
#     sentiment_analyzer = SentimentIntensityAnalyzer()
    for word in sentence:
        sentiment = sentiment_analyzer.polarity_scores(word)
        if sentiment['pos']>sentiment['neg'] and sentiment['pos']>sentiment['neu']:
            n_positive+=1
        elif sentiment['neg']>sentiment['pos'] and sentiment['neg']>sentiment['neu']:
            n_negative+=1
        else:
            continue
    return np.sign(n_positive-n_negative)+1

def sentiment_features(begining, body, climax, option1, option2, P1, P2, P3, P4, sentiment_analyzer):
    s_begining = sentence_sentiment(begining, sentiment_analyzer)
    s_body = sentence_sentiment(body, sentiment_analyzer)
    s_climax = sentence_sentiment(climax, sentiment_analyzer)
    s_option1 = sentence_sentiment(option1, sentiment_analyzer)
    s_option2 = sentence_sentiment(option2, sentiment_analyzer)
    s_context = sentence_sentiment(begining+body+climax, sentiment_analyzer)
    
    features = np.zeros(12)
    
    features[0] = P1[s_option1, s_begining, s_body, s_climax]
    features[1] = P2[s_option1, s_body, s_climax]
    features[2] = P3[s_option1, s_climax]
    features[3] = P4[s_option1, s_context]
    
    features[4] = P1[s_option2, s_begining, s_body, s_climax]
    features[5] = P2[s_option2, s_body, s_climax]
    features[6] = P3[s_option2, s_climax]
    features[7] = P4[s_option2, s_context]
    
    features[8] = 0 if features[0]>features[4] else 1
    features[9] = 0 if features[1]>features[5] else 1
    features[10] = 0 if features[2]>features[6] else 1
    features[11] = 0 if features[3]>features[7] else 1
    
    return features

def generate_vocab(train_stories, vocab_size, vocab_file, corpus_file):
    a = time.time()
    count = 0
    corpus = []
    all_tokens = []
    for i in tqdm(range(len(train_stories))):
        story = train_stories[i]
        events = ['<bos>']
        for sentence in story:
            sentence_pos = nltk.pos_tag(sentence)
            for word_pos in sentence_pos:
                if 'NN' in word_pos[1] or 'PR' in word_pos[1] or 'VB' in word_pos[1]:
                    events.append(word_pos[0])
                    all_tokens.append
        
        corpus.append(events)
        all_tokens.extend(events)
    
    token2count = Counter(all_tokens).most_common(vocab_size)
    vocab_tokens = ['<unk>']
    vocab_tokens.extend([token for token, count in token2count])
    token2id = {token: i for i, token in enumerate(vocab_tokens)}
    id2token = {i: token for token, i in token2id.items()}
    print("Writing Vocabulary Object ... ")
    vocab_dict = {"vocab_size": vocab_size, "token2id": token2id, "id2token": id2token}
    dump(vocab_file, vocab_dict)
    corpus_dict = {"corpus_size": len(corpus), "tokenized_events": corpus}
    dump(corpus_file, corpus_dict)
    print("Done in {} s".format(time.time() - a))
    
    return token2id, id2token, corpus
    
def dump(write_to_file, dictionary):
    out_file = os.path.join(os.getcwd(), write_to_file) # full path
    with open(out_file, "wb+") as f: # dump object
        pickle.dump(dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
    

def sentiment_features_train(train_stories):
    a = time.time()
    P1 = np.zeros((3,3,3,3))
    P2 = np.zeros((3,3,3))
    P3 = np.zeros((3,3))
    P4 = np.zeros((3,3))
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    for i in tqdm(range(len(train_stories))):
        story = train_stories[i]
        begining = story[0]
        body = story[1] + story[2]
        climax = story[3]
        ending = story[4]
        s_begining = sentence_sentiment(begining, sentiment_analyzer)
        s_body = sentence_sentiment(body, sentiment_analyzer)
        s_climax = sentence_sentiment(climax, sentiment_analyzer)
        s_ending = sentence_sentiment(ending, sentiment_analyzer)
        s_context = sentence_sentiment(begining+body+climax, sentiment_analyzer)
        
        P1[s_ending, s_begining, s_body, s_climax]+=1
        P2[s_ending, s_body, s_climax]+=1
        P3[s_ending, s_climax]+=1
        P4[s_ending, s_context]+=1
    
    P1 /= np.sum(P1, axis=0)
    P2 /= np.sum(P2, axis=0)
    P3 /= np.sum(P3, axis=0)
    P4 /= np.sum(P4, axis=0)
    
    print("Done in {} s".format(time.time() - a))
    
    return P1, P2, P3, P4
    

def pos_split(sentences):
    sentences_pos = nltk.pos_tag(sentences)
    sentences_nouns = []
    sentences_verbs = []
    for word_pos in sentences_pos:
        if 'NN' in word_pos[1] or 'PR' in word_pos[1]:
            sentences_nouns.append(word_pos[0])
        elif 'VB' in word_pos[1]:
            sentences_verbs.append(word_pos[0])
    
    return sentences_nouns, sentences_verbs

def extract_event_embeddings(sentences, encoder):
    sentences_pos = nltk.pos_tag(sentences)
    sentences_event_embeds = []
    for word_pos in sentences_pos:
        if 'NN' in word_pos[1] or 'PR' in word_pos[1] or 'VB' in word_pos[1]:
            sentences_event_embeds.append(encoder[word_pos[0]])
    
    return sentences_event_embeds

def topical_consistency(context, option1, option2, encoder):
    context_nouns, context_verbs = pos_split(context)
    option1_nouns, option1_verbs = pos_split(option1)
    option2_nouns, option2_verbs = pos_split(option2)
    
    encoded_context_nouns = [encoder[context_noun] for context_noun in context_nouns]
    encoded_option1_nouns = [encoder[option1_noun] for option1_noun in option1_nouns]
    encoded_option2_nouns = [encoder[option2_noun] for option2_noun in option2_nouns]
    encoded_context_verbs = [encoder[context_verb] for context_verb in context_verbs]
    encoded_option1_verbs = [encoder[option1_verb] for option1_verb in option1_verbs]
    encoded_option2_verbs = [encoder[option2_verb] for option2_verb in option2_verbs]
    
    option1_nouns_similarity_matrix = cosine_similarity(encoded_option1_nouns, encoded_context_nouns)
    option2_nouns_similarity_matrix = cosine_similarity(encoded_option2_nouns, encoded_context_nouns)
    option1_verbs_similarity_matrix = cosine_similarity(encoded_option1_verbs, encoded_context_verbs)
    option2_verbs_similarity_matrix = cosine_similarity(encoded_option2_verbs, encoded_context_verbs)
    
    option1_nouns_similarity = np.max(option1_nouns_similarity_matrix, axis=1)
    option2_nouns_similarity = np.max(option2_nouns_similarity_matrix, axis=1)
    option1_verbs_similarity = np.max(option1_verbs_similarity_matrix, axis=1)
    option2_verbs_similarity = np.max(option2_verbs_similarity_matrix, axis=1)
    
    option1_score = np.mean(np.concatenate(option1_nouns_similarity,option1_verbs_similarity))
    option2_score = np.mean(np.concatenate(option2_nouns_similarity,option2_verbs_similarity))
    
    comparative = 0 if option1_score>option2_score else 1
    
    return option1_score, option2_score, comparative

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model


# In[22]:


train_stories = get_stories_as_lists(train)


# In[40]:

print("Learning sentiment parameters...\n")

P1, P2, P3, P4 = sentiment_features_train(train_stories)


# In[41]:


pkl_file = open('sentement_params.pkl', 'wb')

data = {'P1':P1, 'P2': P2, 'P3': P3, 'P4': P4}

pickle.dump(data, pkl_file)
pkl_file.close()


# In[45]:


# gloveFile = './glove.6B.50d.txt'
# encoder = loadGloveModel(gloveFile)


# In[5]:


valid_stories = get_stories_as_lists(val_sentences)


# In[99]:

print("Extracting valid sentiment features...\n")
a = time.time()
sentiment_analyzer = SentimentIntensityAnalyzer()
n = len(valid_stories)
features = np.zeros((n, 12))
y = np.zeros(n, dtype = int)
for i in tqdm(range(n)):
    story = valid_stories[i]
    begining = story[0]
    body = story[1] + story[2]
    climax = story[3]
    option1 = story[4]
    option2 = story[5]
    
    sf = sentiment_features(begining, body, climax, option1, option2, P1, P2, P3, P4, sentiment_analyzer)
    
    context = begining + body + climax
    
#     tf = topical_consistency(context, option1, option2, encoder)
    
#     features[i] = np.concatenate((sf, tf))
    features[i] = sf
    
    y[i] = val_answer[i]-1

print("Done in {} s".format(time.time() - a))


# In[6]:


class LogisticRegressionClass():
    def __init__(self, learning_rate=0.01, stopping_criterion=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.stopping_criterion = stopping_criterion
        self.max_epochs = max_epochs
        self.w = None
    
    def fit(self, x, y):
        m = x.shape[0]
        n = x.shape[1]
        self.w = np.random.randn(n)
        gradient = np.zeros(n)
        cost = np.zeros(self.max_epochs)
        
        for epoch in tqdm(range(0,self.max_epochs)):
            for i in range(0,m):
                y_hat=expit(np.dot(self.w,x[i,:]))
                gradient=(y_hat-y[i])*x[i,:]
                self.w-=self.learning_rate*gradient
                cost[epoch]-=(y[i]*np.log(y_hat)+(1-y[i])*np.log(1-y_hat))
        
        plt.plot(cost)
        return self.w
    
    def predict(self, x):
        m = x.shape[0]
        n = x.shape[1]
        y = np.zeros(m)
        for i in range(0,m):
            y[i] = 1 if expit(np.dot(self.w,x[i,:]))<0.5 else 2
        
        return y


# In[102]:


model = LogisticRegressionClass()


# In[103]:


w = model.fit(features, y)


# In[104]:


y_pred = model.predict(features)

print(accuracy_score(y+1, y_pred))


# In[81]:


test_stories = get_stories_as_lists(test_sentences)


# In[105]:

print("Extracting test sentiment features...\n")

a = time.time()
sentiment_analyzer = SentimentIntensityAnalyzer()
n = len(test_stories)
test_features = np.zeros((n, 12))
y_gt = np.zeros(n, dtype = int)
for i in tqdm(range(n)):
    story = test_stories[i]
    begining = story[0]
    body = story[1] + story[2]
    climax = story[3]
    option1 = story[4]
    option2 = story[5]
    
    sf = sentiment_features(begining, body, climax, option1, option2, P1, P2, P3, P4, sentiment_analyzer)
    
    context = begining + body + climax
    
#     tf = topical_consistency(context, option1, option2, encoder)
    
#     features[i] = np.concatenate((sf, tf))
    test_features[i] = sf
    
    y_gt[i] = test_answer[i]

print("Done in {} s".format(time.time() - a))


# In[106]:


y_pred = model.predict(test_features)

print(accuracy_score(y_gt, y_pred))


# In[119]:

print("Building Corpus and Vocabulary...\n")

vocab_size = 20000
vocab_file = './vocab.pkl'
corpus_file = './corpus.pkl'
token2id, id2token, corpus = generate_vocab(train_stories, vocab_size, vocab_file, corpus_file)


# In[16]:


class Model():
    def __init__(self, token2id, hidden_size=512, embedding_size=100, vocab_size=20000):
        self.dtype = tf.float32
        initializer = tf.contrib.layers.xavier_initializer(dtype=self.dtype)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.token2id = token2id
        
        self.W = tf.Variable(initializer((hidden_size, vocab_size)), name="W", dtype=self.dtype)
        self.biases = tf.Variable(initializer([vocab_size]), name="biases", dtype=self.dtype)
        
        self.embedding_matrix = tf.Variable(initializer((vocab_size, embedding_size)), name="embedding",
                                                dtype=self.dtype)
        self.rnn = Cell(num_units=hidden_size, initializer=initializer, name="cell")
        self.c_in = tf.placeholder(tf.float32, shape=[1, self.rnn.state_size.c], name='c_in')
        self.h_in = tf.placeholder(tf.float32, shape=[1, self.rnn.state_size.h], name='h_in')

        self.int_state = tf.contrib.rnn.LSTMStateTuple(self.c_in, self.h_in)  # internal state
        self.current_word = tf.placeholder(tf.int32, [1])
        self.next_word = tf.placeholder(tf.int32, [1])
        self.current_embed = tf.nn.embedding_lookup(self.embedding_matrix, self.current_word)

        next_output, self.next_state = self.rnn(self.current_embed, self.int_state)
        next_logits = tf.matmul(next_output, self.W) + self.biases
        self.next_probabs = tf.nn.softmax(next_logits)
        
        self.loss = self.compute_loss(logits=next_logits, labels=self.next_word)
        self.optimizer = tf.train.AdamOptimizer()
        self.saver = tf.train.Saver()
        self.train_op = self.get_train_op()
        
    def compute_loss(self, logits, labels):
        self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return tf.reduce_sum(self.losses)
    
    def step(self, sess, feed_dict):
        next_probabs, current_state = sess.run([self.next_probabs, self.next_state], feed_dict)
        return next_probabs, current_state
    
    def get_train_op(self):
        return self.optimizer.minimize(self.loss)
    
    def save_model(self, sess, path):
        self.saver.save(sess, path)
        print("Model saved at %s" % path)

    def load_model(self, sess, path):
        try:  # try to load the model
            self.saver.restore(sess, path)
            print("Model restored from %s" % path)
            return True
        except:
            print("Couldn't restore model")
            return False
    def event_features(self, context, option1, option2, sess, D):
        if len(context)>D:
            event_sequence_context = ['<bos>'] + context[-D+1:]
        else:
            event_sequence_context = ['<bos>']*(D-len(context)) + context
        
        event_context_ind = []
        for word in event_sequence_context:
            try:
                ind = self.token2id[word]
            except KeyError as outlier:
                # print('The word ' + word + ' doesn\'t exist in the vocabulary')
                ind = self.token2id['<unk>']
            event_context_ind.append(ind)
        
        features = np.zeros(3*D, dtype=float)
        
        #option 1:
        option1_ind = []
        for word in option1:
            try:
                ind = self.token2id[word]
            except KeyError as outlier:
                # print('The word ' + word + ' doesn\'t exist in the vocabulary')
                ind = self.token2id['<unk>']
            option1_ind.append(ind)
        
        print(option1_ind)
        
        #option 2:
        option2_ind = []
        for word in option2:
            try:
                ind = self.token2id[word]
            except KeyError as outlier:
                # print('The word ' + word + ' doesn\'t exist in the vocabulary')
                ind = self.token2id['<unk>']
            option2_ind.append(ind)
        
        for i in range(D-1, -1, -1):
            c_init = np.zeros((1, self.rnn.state_size.c), np.float32)
            h_init = np.zeros((1, self.rnn.state_size.h), np.float32)
            for j in range(i,D-1,1):
                current_word = event_context_ind[j]
                feed_dict = {self.current_word: [current_word], self.c_in: c_init, self.h_in: h_init}
                next_prob, state = self.step(sess, feed_dict)
                c_init, h_init = state
            p = 1
            current_word = event_context_ind[D-1]
            cont_state = c_init, h_init
            for ind in option1_ind:
                feed_dict = {self.current_word: [current_word], self.c_in: c_init, self.h_in: h_init}
                current_word = ind
                next_prob, state = self.step(sess, feed_dict)
                c_init, h_init = state
                p*=next_prob[0][ind]
            features[i] = p
            
            current_word = event_context_ind[D-1]
            c_init, h_init = cont_state
            p=1
            for ind in option2_ind:
                feed_dict = {self.current_word: [current_word], self.c_in: c_init, self.h_in: h_init}
                current_word = ind
                next_prob, state = self.step(sess, feed_dict)
                c_init, h_init = state
                p*=next_prob[0][ind]
            features[D+i] = p
            
            if(features[i] > features[D+i]):
                features[2*D+i] = -1
            else:
                features[2*D+i] = 1
        
        return features


# In[138]:


n = len(corpus)
num_epochs = 2
model = Model(token2id = token2id, hidden_size=512, embedding_size=100, vocab_size=20000)
print("Training LSTM language model...\n")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    with tqdm(total=n*num_epochs) as pbar:
        for epoch in range(num_epochs):
            count = 0
            arr = np.arange(n)
            np.random.shuffle(arr)
            for i in arr:
                count+=1
                c_init = np.zeros((1, model.rnn.state_size.c), np.float32)
                h_init = np.zeros((1, model.rnn.state_size.h), np.float32)
                total_loss = 0
                for j in range(len(corpus[i])-1):
                    word = corpus[i][j]
                    next_word = corpus[i][j+1]
                    try:
                        ind = token2id[word]
                    except KeyError as outlier:
                        # print('The word ' + word + ' doesn\'t exist in the vocabulary')
                        ind = token2id['<unk>']
                    try:
                        next_ind = token2id[next_word]
                    except KeyError as outlier:
                        # print('The word ' + next_word + ' doesn\'t exist in the vocabulary')
                        next_ind = token2id['<unk>']
                    feed_dict = {model.current_word: [ind], model.c_in: c_init, model.h_in: h_init,
                                model.next_word: [next_ind]}
                    _,loss = sess.run([model.train_op, model.loss], feed_dict)
                    total_loss+=loss
                    pbar.update(1)
                print("epoch {}, story {}, loss {}".format(epoch+1, count, total_loss))
    model.save_model(sess, os.path.join(os.getcwd(), 'model.ckpt'))
    
    D = 10

    print("Extracting event sequence features...\n")

    sentiment_analyzer = SentimentIntensityAnalyzer()
    n = len(valid_stories)
    features = np.zeros((n, 3*D+12))
    y = np.zeros(n, dtype = int)
    for i in tqdm(range(n)):
        story = valid_stories[i]
        begining = story[0]
        body = story[1] + story[2]
        climax = story[3]
        option1 = story[4]
        option2 = story[5]

        sf = sentiment_features(begining, body, climax, option1, option2, P1, P2, P3, P4, sentiment_analyzer)


        context = begining+body+climax

        context_events = []
        for sentence in context:
            sentence_pos = nltk.pos_tag(sentence)
            for word_pos in sentence_pos:
                if 'NN' in word_pos[1] or 'PR' in word_pos[1] or 'VB' in word_pos[1]:
                    context_events.append(word_pos[0])
        option1_events = []
        sentence_pos = nltk.pos_tag(option1)
        for word_pos in sentence_pos:
            if 'NN' in word_pos[1] or 'PR' in word_pos[1] or 'VB' in word_pos[1]:
                option1_events.append(word_pos[0])
        option2_events = []
        sentence_pos = nltk.pos_tag(option2)
        for word_pos in sentence_pos:
            if 'NN' in word_pos[1] or 'PR' in word_pos[1] or 'VB' in word_pos[1]:
                option2_events.append(word_pos[0])

        ef = model.event_features(context_events, option1_events, option2_events, sess, D)

        features[i] = np.concatenate((sf, ef))

        y[i] = val_answer[i]-1


    print("Extracting test features...\n")

    n = len(test_stories)
    test_features = np.zeros((n, 3*D+12))
    y_gt = np.zeros(n, dtype = int)
    for i in tqdm(range(n)):
        story = test_stories[i]
        begining = story[0]
        body = story[1] + story[2]
        climax = story[3]
        option1 = story[4]
        option2 = story[5]

        sf = sentiment_features(begining, body, climax, option1, option2, P1, P2, P3, P4, sentiment_analyzer)


        context = begining+body+climax

        context_events = []
        for sentence in context:
            sentence_pos = nltk.pos_tag(sentence)
            for word_pos in sentence_pos:
                if 'NN' in word_pos[1] or 'PR' in word_pos[1] or 'VB' in word_pos[1]:
                    context_events.append(word_pos[0])
        option1_events = []
        sentence_pos = nltk.pos_tag(option1)
        for word_pos in sentence_pos:
            if 'NN' in word_pos[1] or 'PR' in word_pos[1] or 'VB' in word_pos[1]:
                option1_events.append(word_pos[0])
        option2_events = []
        sentence_pos = nltk.pos_tag(option2)
        for word_pos in sentence_pos:
            if 'NN' in word_pos[1] or 'PR' in word_pos[1] or 'VB' in word_pos[1]:
                option2_events.append(word_pos[0])

        ef = model.event_features(context_events, option1_events, option2_events, sess, D)

        test_features[i] = np.concatenate((sf, ef))

        y_gt[i] = test_answer[i]


# In[ ]:


print("Training Logistic Regresion Model")

lr_model = LogisticRegressionClass()
w = lr_model.fit(features, y)
y_pred = lr_model.predict(features)

print("Training accuracy: "+ str(accuracy_score(y+1, y_pred)))

y_pred = lr_model.predict(test_features)

print("Testing accuracy: "+ str(accuracy_score(y_gt, y_pred)))

