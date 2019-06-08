import pandas as pd
from keras_preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.models import Sequential

from process import process, concat_sentences

MAX_SEQUENCE_LENGTH = 76
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/val.csv')
test = pd.read_csv('data/test.csv')

all_sentences = concat_sentences(train).tolist()
all_sentences.extend(concat_sentences(val))
all_sentences.extend(concat_sentences(test))

# first, build index mapping words in the embeddings set
# to their embedding vector
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token=True)
tokenizer.fit_on_texts(all_sentences)

X_train, y_train = process(train, tokenizer, MAX_SEQUENCE_LENGTH)
X_val, y_val = process(val, tokenizer, MAX_SEQUENCE_LENGTH)
X_test, y_test = process(test, tokenizer, MAX_SEQUENCE_LENGTH)

batch_size = 32

# input = Input(shape=(maxlen,))
# x = Embedding(MAX_NUM_WORDS, 128, input_length=maxlen)(input)
model = Sequential()
model.add(Embedding(MAX_NUM_WORDS, 128, input_length=MAX_SEQUENCE_LENGTH))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=[X_val, y_val])
