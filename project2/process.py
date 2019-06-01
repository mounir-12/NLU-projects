from functools import reduce
from typing import Union

import numpy as np
from tensorflow.keras.preprocessing import sequence


def concat_sentences(df, sentences: Union[int, range] = range(1, 6), sep=' '):
    """
    Concatenates a subset of the five sentences with "sep"

    :param df: The dataframe containing the sentences. Its columns must be "sentencex", where x is the sentence number
    :param sentences: the columns to concatenate. If a single int is given, will return the values of this column
    :param sep:
    :return: Numpy array containing the list of concatenated sentences
    """
    if isinstance(sentences, int):
        return df['sentence' + str(sentences)].values
    else:
        return reduce(lambda x, y: x.astype(str) + sep + y.astype(str),
                      [df['sentence' + str(col)] for col in sentences]).values


def process(df, tokenizer, max_seq_len):
    """
    Mostly for the baseline RNN task

    :param df:
    :param tokenizer:
    :param max_seq_len:
    :return: X, the padded, tokenized and "sequencized" sentences and y, the labels
    """
    X = concat_sentences(df)
    y = df.label

    # vectorize the text samples into a 2D integer tensor
    X = tokenizer.texts_to_sequences(X)

    # Pad sequences (samples x time)
    X = sequence.pad_sequences(X, maxlen=max_seq_len)
    print('x shape:', X.shape)
    y = np.array(y)

    return X, y
