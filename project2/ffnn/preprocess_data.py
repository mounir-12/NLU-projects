import skipthoughts
import pandas as pd
import numpy as np
import h5py
import utils
import os, time
from argparse import ArgumentParser

global_seed = 5

np.random.seed(global_seed)

parser = ArgumentParser()
parser.add_argument("-v", "--verbose", help="Print more data", action="store_true")
args = parser.parse_args()

VERBOSE = args.verbose
DATA_DIR = "./data"
ENCODED_DATA_DIR = "./encoded_data"

print("\nReading data ...")
train = pd.read_csv(os.path.join(DATA_DIR, "train_stories.csv"))
val = pd.read_csv(os.path.join(DATA_DIR, "cloze_test_val__spring2016 - cloze_test_ALL_val.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test_for_report-stories_labels.csv"))

pred_sentences = pd.read_csv(os.path.join(DATA_DIR, "test-stories.csv"))

train = train.drop("storytitle", axis=1).drop("storyid", axis=1)

val = val.drop("InputStoryid", axis=1)
val_answers = np.array(val["AnswerRightEnding"]).reshape([-1, 1])
val_sentences = val.drop("AnswerRightEnding", axis=1)

test = test.drop("InputStoryid", axis=1)
test_answers = np.array(test["AnswerRightEnding"]).reshape([-1, 1])
test_sentences = test.drop("AnswerRightEnding", axis=1)

# Skipthoughts model from: https://github.com/ryankiros/skip-thoughts
a = time.time()
print("\nLoading skipthoughts model ...")
model_skipthoughts = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model_skipthoughts)
print("Done in {} s".format(time.time() - a))

# Labeled Train Data
print("\nVal stories/sentences ...")
val_stories, val_stories_flat = utils.get_stories_as_lists(val_sentences)
if VERBOSE:
    print("\nNb of stories:", len(val_stories))
    print("Example story with right and wrong ending:", val_stories[0])
    print("Nb of sentences per story:", len(val_stories[0]), "\n")

# Encode validation sentences using skipthoughts encoding
print("\nEncoding sentences ...")
val_encoded_stories = utils.encode_stories(encoder, val_stories_flat)
if VERBOSE:
    print("\n", val_encoded_stories.shape)
    print(val_encoded_stories[0], "\n") # print story 0 (both endings in the 5th and 6th sentence)

# Save encoded sentences for faster loading the next run
os.makedirs(ENCODED_DATA_DIR) # create save dir
val_file = h5py.File(os.path.join(ENCODED_DATA_DIR, "val_data.h5"), 'w')
val_file.create_dataset("val_encoded_stories", data=val_encoded_stories)
val_file.create_dataset("val_answers", data=val_answers)
val_file.close()

# Same operations for test set
print("\nTest stories/sentences ...")
test_stories, test_stories_flat = utils.get_stories_as_lists(test_sentences)
if VERBOSE:
    print("\nNb of stories:", len(test_stories))
    print("Example story with right and wrong ending:", test_stories[0])
    print("Nb of sentences per story:", len(test_stories[0]), "\n")

print("\nEncoding sentences ...")
test_encoded_stories = utils.encode_stories(encoder, test_stories_flat)
if VERBOSE:
    print("\n",test_encoded_stories.shape)
    print(test_encoded_stories[0], "\n") # print story 0 (both endings in the 5th and 6th sentence)

test_file = h5py.File(os.path.join(ENCODED_DATA_DIR, "test_data.h5"), 'w')
test_file.create_dataset("test_encoded_stories", data=test_encoded_stories)
test_file.create_dataset("test_answers", data=test_answers)
test_file.close()

# Same operations for prediction stories
print("\nPred stories/sentences ...")
pred_stories, pred_stories_flat = utils.get_stories_as_lists(pred_sentences)
if VERBOSE:
    print("\nNb of stories:", len(pred_stories))
    print("Example story with right and wrong ending:", pred_stories[0])
    print("Nb of sentences per story:", len(pred_stories[0]), "\n")

print("\nEncoding sentences ...")
pred_encoded_stories = utils.encode_stories(encoder, pred_stories_flat)
if VERBOSE:
    print("\n",pred_encoded_stories.shape)
    print(pred_encoded_stories[0], "\n") # print story 0 (both endings in the 5th and 6th sentence)
    
pred_file = h5py.File(os.path.join(ENCODED_DATA_DIR, "pred_data.h5"), 'w')
pred_file.create_dataset("pred_encoded_stories", data=pred_encoded_stories)
pred_file.close()
