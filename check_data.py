import os
from collections import Counter

train_data_path = "/data/sentences.train"
test_data_path = "/data/sentences_test.txt"

def read(relative_path):

    # Load data from file
    cwd = os.getcwd() # get current working dir
    lengths_list = []

    with open(cwd + relative_path) as data_file:
        for line in data_file: # for each sentence 
            tokens = line.strip().split(" ") # get sentence tokens
            lengths_list.append(len(tokens)) # append nb of tokens in sentence
    
    total = len(lengths_list)
    lengths2counts = Counter(lengths_list) # map from length of sentence to nb of sentences with that length
    print("Total #(sentences): {}".format(total))
    for length, count in sorted(lengths2counts.items()): # loop over the key-value pairs ordered by key
        print("Length: {} -> Sentences: {}".format(length, count))

read(train_data_path)
