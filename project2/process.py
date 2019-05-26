from functools import reduce

import numpy as np
from tensorflow.keras.preprocessing import sequence


def concat_sentences(df, sentences=range(1, 6), sep=' '):
    if isinstance(sentences, int):
        return df['sentence' + str(sentences)].values
    else:
        return reduce(lambda x, y: x.astype(str) + sep + y.astype(str),
                      [df['sentence' + str(col)] for col in sentences]).values


def process(df, tokenizer, max_seq_len):
    X = concat_sentences(df)
    y = df.label

    # vectorize the text samples into a 2D integer tensor
    X = tokenizer.texts_to_sequences(X)

    # Pad sequences (samples x time)
    X = sequence.pad_sequences(X, maxlen=max_seq_len)
    print('x shape:', X.shape)
    y = np.array(y)

    return X, y
