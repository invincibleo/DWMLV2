#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 09.08.17 11:42
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : Metrics
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import tensorflow as tf
import numpy as np

def top3_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

def apk(actual, predicted):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    actual = keras.backend.get_value(actual)
    predicted = keras.backend.get_value(predicted)
    k = 43

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return keras.backend.variable(score / min(len(actual), k))

def mapk(actual, predicted, k=100):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return keras.backend.mean(tf.metrics.sparse_average_precision_at_k(tf.cast(actual, tf.int64), predicted, 43)[0])
