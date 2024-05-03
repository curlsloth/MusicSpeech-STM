#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:31:58 2024

@author: andrewchang
"""

from sklearn.datasets import make_multilabel_classification
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC, SVC
X, y = load_iris(return_X_y=True)
from sklearn.linear_model import RidgeClassifierCV
# clf = LinearSVC().fit(X, y)
# clf = LogisticRegression(solver="liblinear", multi_class="ovr").fit(X, y)
clf = SVC(probability=True).fit(X, y)

# roc_auc_score(y, clf.decision_function(X), multi_class='ovr')
roc_auc_score(y, clf.predict_proba(X), multi_class='ovr')


import pandas as pd


from sklearn.model_selection import train_test_split

iris = load_iris()
X = pd.DataFrame(iris.data)
y = pd.DataFrame(iris.target)

X_train, X_val, y_train, y_val = train_test_split(X, y,
    test_size=0.2, shuffle = True, random_state = 123)