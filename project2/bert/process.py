from functools import reduce
from typing import Union

import numpy as np


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


def accuracy(preds):
    """

    :param preds:
    :return: Choice accuracy: the accuracy of choosing the right story ending
             Classification accuracy: the accuracy of classifying each ending as either correct or not
    """
    grouped = preds.groupby('id')

    choices = []
    for storyid, group in grouped:
        true_prob_true = group.loc[group.label == 1].log_prob_true.values[0]
        fake_prob_true = group.loc[group.label == 0].log_prob_true.values[0]
        if true_prob_true > fake_prob_true:
            choices.append(1)
        else:
            choices.append(0)

    choice_acc = sum(choices) / len(choices)

    preds_probs = preds[['log_prob_fake', 'log_prob_true']]
    preds_probs.columns = np.arange(len(preds_probs.columns))
    class_acc = (preds.label == preds_probs.idxmax(axis=1)).mean()

    return choice_acc, class_acc


def prediction_choice(preds):
    """

    :param preds:
    :return: Choice accuracy: the accuracy of choosing the right story ending
             Classification accuracy: the accuracy of classifying each ending as either correct or not
    """
    grouped = preds.groupby('id')

    choices = []
    for storyid, group in grouped:
        one_prob_true = group.loc[group.label == 1].log_prob_true.values[0]
        two_prob_true = group.loc[group.label == 0].log_prob_true.values[0]
        if one_prob_true > two_prob_true:
            choices.append(1)
        else:
            choices.append(2)

    return choices
