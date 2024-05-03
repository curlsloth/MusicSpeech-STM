#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:16:28 2024

@author: andrewchang
"""

clf.fit(X_train, y_train)
y_val_pred = clf.predict(X_val)

roc_auc_score(y_val, clf.predict_proba(X_val), multi_class='ovr')

y_val_prob = clf.predict_proba(X_val)

cm = confusion_matrix(y_val, y_val_pred, normalize='true')
print(classification_report(y_val, y_val_pred))

print("Confusion matrix")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['speech: non-tonal', 'speech: tonal', 'music: vocal', 'music: non-vocal', 'env'])
disp.plot()
plt.show()