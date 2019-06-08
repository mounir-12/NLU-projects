import tensorflow as tf
import numpy as np
import keras
import os, glob
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score as accuracy
import utils

class ANN:
    def __init__(self, summed=False, dropout_rate=0.0, learning_rate=0.001):
    
        print("Dropout {}, Learning_rate: {}".format(dropout_rate, learning_rate))
        model = Sequential()
        if not summed: # if last sentence concatenated with ending, then the input has 9600 dims
            model.add(Dense(units=4800, activation='relu', input_dim=9600))
            model.add(Dropout(rate=dropout_rate))
            model.add(Dense(units=2400, activation='relu'))
            model.add(Dropout(rate=dropout_rate))
        else: # otherwise, input has 4800 dims
            model.add(Dense(units=2400, activation='relu', input_dim=4800))
            model.add(Dropout(rate=dropout_rate))
            
        
        model.add(Dense(units=1200, activation='relu'))
        model.add(Dropout(rate=dropout_rate))
        model.add(Dense(units=600, activation='relu'))
        model.add(Dropout(rate=dropout_rate))
        model.add(Dense(units=2, activation='softmax')) # output layer
        
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=learning_rate))
        
        self.model = model
        
        self.LOG_DIR = "./logs/{}_dropout:{}_lr:{}_summed:{}".format(utils.timestamp(), dropout_rate, learning_rate, summed)
        
        
    
    def train_validate_test(self, train_inputs, train_labels, train_answers, valid_inputs, valid_labels, valid_answers, test_inputs, test_answers, batch_size, epochs):
        train_labels_categorical = keras.utils.to_categorical(train_labels, num_classes=2, dtype='float32')
        
        
        valid_labels_categorical = keras.utils.to_categorical(valid_labels, num_classes=2, dtype='float32')
        
        self.writerSaverCallback = WriterSaverCallback(self, train_inputs, train_answers, valid_inputs, valid_answers, test_inputs, test_answers, self.LOG_DIR)
        self.model.fit(train_inputs, train_labels_categorical, epochs=epochs, batch_size=batch_size, 
                  callbacks=[self.writerSaverCallback], validation_data=[valid_inputs, valid_labels_categorical])
                  
    def get_predicted_right_endings(self, data):
        # expects the data to be a succession of pairs which correspond to the 2 possible endings (ending 1 and ending 2) of the same story
        preds = self.model.predict(data)
        preds_endings = []
        for i in range(preds.shape[0]//2):
            if(preds[2*i][1] > preds[2*i+1][1]): # if the first ending is "more right" then the second ending
                preds_endings.append(1)
            else:
                preds_endings.append(2)
        return np.array(preds_endings)
        
          
class WriterSaverCallback(keras.callbacks.Callback):
    def __init__(self, ann, train_inputs, train_answers, valid_inputs, valid_answers, test_inputs, test_answers, LOG_DIR):
        print("\nInitializing Writer ...\n")
        self.writer = tf.summary.FileWriter(LOG_DIR, K.get_session().graph)
        
        self.train_loss_pl = tf.placeholder(dtype=tf.float32)
        self.valid_loss_pl = tf.placeholder(dtype=tf.float32)
        self.train_accuracy_pl = tf.placeholder(dtype=tf.float32)
        self.valid_accuracy_pl = tf.placeholder(dtype=tf.float32)
        self.test_accuracy_pl = tf.placeholder(dtype=tf.float32)
        self.summary = tf.summary.merge(
                [tf.summary.scalar("train_loss", self.train_loss_pl), 
                 tf.summary.scalar("valid_loss", self.valid_loss_pl),
                 tf.summary.scalar("train_accuracy", self.train_accuracy_pl),
                 tf.summary.scalar("valid_accuracy", self.valid_accuracy_pl),
                 tf.summary.scalar("test_accuracy", self.test_accuracy_pl)
                ])
        
        
        self.ann = ann
        self.LOG_DIR = LOG_DIR
        
        self.train_inputs = train_inputs
        self.train_answers = train_answers
        self.valid_inputs = valid_inputs
        self.valid_answers = valid_answers
        self.test_inputs = test_inputs
        self.test_answers = test_answers
        
        self.best_valid_acc = 0.0 # best validation accuracy while training
        self.best_train_acc = 0.0 # train accuracy of the model with best validation accuracy
        self.best_test_acc = 0.0 # test accuracy of the model with best validation accuracy
        self.best_epoch = 0 # epoch nb of the model with best validation accuracy
    
    def on_epoch_end(self, epoch, logs={}):
        train_pred = self.ann.get_predicted_right_endings(self.train_inputs)
        valid_pred = self.ann.get_predicted_right_endings(self.valid_inputs)
        test_pred = self.ann.get_predicted_right_endings(self.test_inputs)
        
        train_loss = logs["loss"]
        valid_loss = logs["val_loss"]
        
        print("\n")
        print("Train Loss: {}, Validation Loss: {}".format(train_loss, valid_loss))
        
        print("Train:")
#         print("   Predicted endings:", train_pred)
#         print("   Correct endings:", self.train_answers)
        train_acc = accuracy(self.train_answers, train_pred)
        print("   Train Accuracy: {}".format(train_acc))
        
        print("Validation:")
#         print("   Predicted endings:", valid_pred)
#         print("   Correct endings:", self.valid_answers)
        valid_acc = accuracy(self.valid_answers, valid_pred)
        print("   Validation Accuracy: {}".format(valid_acc))
        
        print("Test:")
#         print("   Predicted endings:", test_pred)
#         print("   Correct endings:", self.test_answers)
        test_acc = accuracy(self.test_answers, test_pred)
        print("   Test Accuracy: {}".format(test_acc))
        print("\n")
        
        sess = K.get_session()
        summary = sess.run(self.summary, {self.train_loss_pl: train_loss,
                                      self.valid_loss_pl: valid_loss,
                                      self.train_accuracy_pl: train_acc,
                                      self.valid_accuracy_pl: valid_acc,
                                      self.test_accuracy_pl: test_acc})
        self.writer.add_summary(summary, epoch)
        self.writer.flush()
        
        if valid_acc > self.best_valid_acc:
            print("New best model, saving ...\n")
            self.best_valid_acc = valid_acc
            self.best_train_acc = train_acc
            self.best_test_acc = test_acc
            self.best_epoch = epoch
            self.ann.model.save_weights(os.path.join(self.LOG_DIR, "best_model.h5".format(epoch)))
            
            
        
    def on_train_end(self, logs=None):
        print("Best Model at epoch {}:\n   Train Accuracy: {}, Validation Accuracy: {}, Test Accuracy: {}".format(self.best_epoch, self.best_train_acc, self.best_valid_acc, self.best_test_acc))
        self.writer.flush()
        self.writer.close()
        
        
        

