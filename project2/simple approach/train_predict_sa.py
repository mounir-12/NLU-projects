import pandas as pd
import utils
import numpy as np
import h5py
import time, gc, os, sys, glob
import tensorflow as tf
from sklearn.metrics import accuracy_score as accuracy
from keras.models import load_model

from model_sa import ANN

print("\nFixing seeds ...")
tf.set_random_seed(5)
np.random.seed(5)

SUM_LAST_WITH_ENDING=False
DROPOUT=0.2
NUM_EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=0.001
SKIP_TRAINING=False # when true, directly loads the final trained model and predicts using it

VERBOSE = True
ENCODED_DATA_DIR = "./encoded_data"

# Read data
if os.path.exists(ENCODED_DATA_DIR): # if already saved encoded sentences, read saved data
    print("\nReading encoded data ...")
    
    val_file = h5py.File(os.path.join(ENCODED_DATA_DIR, "val_data.h5"), 'r')
    test_file = h5py.File(os.path.join(ENCODED_DATA_DIR, "test_data.h5"), 'r')
    pred_file = h5py.File(os.path.join(ENCODED_DATA_DIR, "pred_data.h5"), 'r')
    
    print("\nVal data ...")
    a = time.time()
    val_encoded_stories = np.array(val_file["val_encoded_stories"])
    val_answers = np.array(val_file["val_answers"])
    print("Done in {} s".format(time.time() - a))
    
    print("\nTest data ...")
    a = time.time()
    test_encoded_stories = np.array(test_file["test_encoded_stories"])
    test_answers = np.array(test_file["test_answers"])
    print("Done in {} s".format(time.time() - a))

    print("\nPred data ...")
    a = time.time()
    pred_encoded_stories = np.array(pred_file["pred_encoded_stories"])
    print("Done in {} s".format(time.time() - a))

    val_file.close()
    test_file.close()
    pred_file.close()
    
    if VERBOSE:
        print("\n", val_encoded_stories.shape)
        print(val_answers.shape)
        print(val_encoded_stories[0]) # print story 0 (both endings in the 5th and 6th sentence)
        print(test_encoded_stories.shape)
        print(test_answers.shape)
        print(test_encoded_stories[0], "\n") # print story 0 (both endings in the 5th and 6th sentence)

else:
    print("Error, please first run preprocess_data.py ...")
    sys.exit(-1)

#sys.exit(0)

# split labeled data to train and validation data
print("\nSplitting data to train/validation data ...")
train_stories, valid_stories, train_answers, valid_answers = utils.get_train_valid_split(val_encoded_stories, val_answers, valid_percent=0.1, shuffle=True)
if VERBOSE:
    print("\n", train_stories.shape)
    print(train_answers.shape)
    print(valid_stories.shape)
    print(valid_answers.shape)
    print(train_stories[0], "\n") # print train story 0 (both endings in the 5th and 6th sentence)

# split each train story to its positive and negative ending versions and convert labels
print("\nSplitting each story to positive and negative story ...")
print("Train data ...")
train_stories_split = utils.split_pos_neg_endings(train_stories)
train_labels_split = utils.create_stories_labels(train_answers) # 0 means the corresponding story in train_stories_split 
                                                      # has a wrong ending, and 1 means it has a right ending
if VERBOSE:
    print(train_stories_split[0:2]) # print same story as before with right and wrong endings on different rows
    print(train_labels_split[0:10])

# do the same for validation and test data
print("\nValidation data ...")
valid_stories_split = utils.split_pos_neg_endings(valid_stories)
valid_labels_split = utils.create_stories_labels(valid_answers) # 0 means the corresponding story in valid_stories_split 
                                                      # has a wrong ending, and 1 means it has a right ending

print("\nTest data...")
test_stories_split = utils.split_pos_neg_endings(test_encoded_stories)

print("\nPred data...")
pred_stories_split = utils.split_pos_neg_endings(pred_encoded_stories)

#sys.exit(0)


# ---> Training and validation

print("\nUsing only last context sentence and story ending ...")
# train: last sentence only + ending
if SUM_LAST_WITH_ENDING:
    train_stories_ls = train_stories_split[:, 3] + train_stories_split[:, 4]
else:
    train_stories_ls = np.concatenate([train_stories_split[:, 3], train_stories_split[:, 4]], axis=1)

if VERBOSE:
    print(train_stories_ls.shape)

# valid: last sentence only + ending
if SUM_LAST_WITH_ENDING:
    valid_stories_ls = valid_stories_split[:, 3] + valid_stories_split[:, 4]
else:
    valid_stories_ls = np.concatenate([valid_stories_split[:, 3], valid_stories_split[:, 4]], axis=1)

if VERBOSE:
    print(valid_stories_ls.shape)

# test: last sentence only + ending
if SUM_LAST_WITH_ENDING:
    test_stories_ls = test_stories_split[:, 3] + test_stories_split[:, 4]
else:
    test_stories_ls = np.concatenate([test_stories_split[:, 3], test_stories_split[:, 4]], axis=1)
    
if VERBOSE:
    print(test_stories_ls.shape)
    
# pred: last sentence only + ending
if SUM_LAST_WITH_ENDING:
    pred_stories_ls = pred_stories_split[:, 3] + pred_stories_split[:, 4]
else:
    pred_stories_ls = np.concatenate([pred_stories_split[:, 3], pred_stories_split[:, 4]], axis=1)
    
if VERBOSE:
    print(pred_stories_ls.shape)

gc.collect() # free memory
print("\nBuilding ANN model ...")
ann = ANN(summed=SUM_LAST_WITH_ENDING, dropout_rate=DROPOUT, learning_rate=LEARNING_RATE)
#sys.exit(0)
if not SKIP_TRAINING:
    ann.train_validate_test(train_stories_ls, train_labels_split, train_answers, 
                          valid_stories_ls, valid_labels_split, valid_answers, 
                          test_stories_ls, test_answers, BATCH_SIZE, NUM_EPOCHS)

# ---> Predictions Generation

print("\n\nRestoring best found model ...")
list_of_files = glob.glob('./logs/*')
LOG_DIR = max(list_of_files, key=os.path.getctime) # latest created dir for latest experiment
print("Restoring model from {}".format(LOG_DIR))
model_path = glob.glob(os.path.join(LOG_DIR, "*.h5"))[-1]
print("model path: {}\n\n".format(model_path))
ann.model.load_weights(model_path)

train_pred = ann.get_predicted_right_endings(train_stories_ls)
valid_pred = ann.get_predicted_right_endings(valid_stories_ls)
test_pred = ann.get_predicted_right_endings(test_stories_ls)

train_acc = accuracy(train_answers, train_pred)
valid_acc = accuracy(valid_answers, valid_pred)
test_acc = accuracy(test_answers, test_pred)

print("Best Model restored:\n   Train Accuracy: {}, Validation Accuracy: {}, Test Accuracy: {}".format(train_acc, valid_acc, test_acc))

pred_stories_pred = ann.get_predicted_right_endings(pred_stories_ls).reshape([-1, 1])

np.savetxt(os.path.join(LOG_DIR, "predictions.csv"), pred_stories_pred, delimiter=",")

    
