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
from sklearn.model_selection import train_test_split
import numpy as np
# clf = LinearSVC().fit(X, y)
# clf = LogisticRegression(solver="liblinear", multi_class="ovr").fit(X, y)
clf = SVC(probability=True).fit(X, y)

# roc_auc_score(y, clf.decision_function(X), multi_class='ovr')
roc_auc_score(y, clf.predict_proba(X), multi_class='ovr')


import pandas as pd


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
iris = load_iris()
X = pd.DataFrame(iris.data)
y = pd.DataFrame(iris.target)

y_ohe = OneHotEncoder(sparse_output=False).fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle = True, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

sdgc = SGDClassifier()
sdgc.fit(X_train,np.array(y_train))

y_test_encoded = OneHotEncoder(sparse_output=False).fit_transform(y_test)
y_pred = sdgc.decision_function(X_test)
roc_auc_score(y_test_encoded, sdgc.decision_function(X_test), multi_class='ovr')