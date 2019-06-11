import numpy as np
import time, datetime
import math

global_seed = 5
np.random.seed(global_seed)

def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y.%m.%d-%H:%M:%S")

def get_stories_as_lists(dataframe):
    a = time.time()
    print("Reading stories to memory ...")
    stories = []
    stories_flat = []
    for index, row in dataframe.iterrows():
        story = []
        for col in dataframe.columns:
            story.append(row[col])
            stories_flat.append(row[col])
        stories.append(story)
    print("Done in {} s".format(time.time() - a))
    return stories, stories_flat

def encode_stories(encoder, stories_flat, encoding_batch_size=2000):
    encoded_sentences = np.zeros([len(stories_flat), 4800]) # each sentence encoded to a 4800-dim vector

    nb_batches = int(math.ceil(len(stories_flat) / encoding_batch_size))
    print("nb sentences per batch: {}, nb batches: {}".format(encoding_batch_size, nb_batches))
    for i in range(nb_batches):
        a = time.time()
        print("Encoding batch {} of sentences ...".format(i))
        encoded_sentences[i*encoding_batch_size : (i+1)*encoding_batch_size] = encoder.encode(stories_flat[i*encoding_batch_size : (i+1)*encoding_batch_size], verbose=False)
        print("Done in {} s".format(time.time() - a))
    return encoded_sentences.reshape([-1, 6, 4800])

def get_train_valid_split(x_data, y_data, valid_percent=0.1, shuffle=True):
    data_size = x_data.shape[0]
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        x_data = x_data[shuffle_indices]
        y_data = y_data[shuffle_indices]
    split_index = int((1-valid_percent)*data_size)
    return x_data[:split_index], x_data[split_index:], y_data[:split_index], y_data[split_index:]

def split_pos_neg_endings(encoded_stories):
    encoded_stories_context = encoded_stories[:, 0:4] # first 4 sentences (i,e the context) 
    print(encoded_stories_context.shape)
    encoded_stories_context = np.repeat(encoded_stories_context, 2, axis=0) # repeat each context twice (once per possible ending)
    print(encoded_stories_context.shape)
    encoded_stories_endings = encoded_stories[:, 4:6] # the 2 possible endings
    print(encoded_stories_endings.shape)
    encoded_stories_endings = encoded_stories_endings.reshape([-1, 1, 4800]) # one ending per row
    print(encoded_stories_endings.shape)
    encoded_stories_split = np.concatenate([encoded_stories_context, encoded_stories_endings], axis=1)
    print(encoded_stories_split.shape)
    return encoded_stories_split

def create_stories_labels(answers):
    answers_split = np.zeros([2*answers.shape[0], 1])
    print("Answers shape:", answers_split.shape)
    for i,answer in enumerate(answers):
        if answer == 1:
            answers_split[2*i] = 1.
            answers_split[2*i+1] = 0.
        else: # answer == 2
            answers_split[2*i] = 0.
            answers_split[2*i+1] = 1.
    return answers_split
    
    
    
    
    
    

