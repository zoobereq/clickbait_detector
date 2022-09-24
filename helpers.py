#!/usr/bin/env python

# imports
import string

import numpy as np
from numpy import ndarray
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB


def get_data(path: str) -> list:
    """reads .txt file into a list of strings"""
    list_of_lines = []
    with open(path, "r") as source:
        for line in source:
            line = line.rstrip()
            if line == False:
                continue
            else:
                list_of_lines.append(line)
    return list_of_lines


def score_m(data: ndarray, target: ndarray) -> float:
    """runs the Multinomial Naive Bayes classifier with 10-fold cross validation
    and reports mean accuracy """
    X = data
    Y = np.array(target)
    score = cross_val_score(MultinomialNB(), X, Y, scoring="accuracy", cv=10)
    return round(score.mean(), 3)


def score_b(data: ndarray, target: ndarray) -> float:
    """runs the Bernoulli Naive Bayes classifier with 10-fold cross validation
    and reports mean accuracy """
    X = data
    Y = np.array(target)
    score = cross_val_score(BernoulliNB(), X, Y, scoring="accuracy", cv=10)
    return round(score.mean(), 3)
