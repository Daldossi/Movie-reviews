# -*- coding: utf-8 -*-
"""
@author: alice
Title: Analyze the results obtained from the script C_TrainClassifier
"""

import numpy as np
from C_TrainClassifier import *


data = np.load("model.npz")
SET = data["arr_0"]
X_train = data["arr_1"]
Y_train = data["arr_2"]
X_test = data["arr_3"]
Y_test = data["arr_4"]
predictions_test = data["arr_5"]
w = data["arr_6"]
b = data["arr_7"]


if SET == 'smalltrain':
    f = open('vocabulary_smalltrain.txt')
if SET == 'train':
    f = open('vocabulary_train.txt')
    
    
## Find the most common words in negative and positive reviews
words = f.read().split()
f.close()
indices = np.argsort(w)
print('Negative')
for i in range(20):
    print(i + 1, words[indices[i]], w[indices[i]])
print('Positive')
for i in range(20):
    print(i + 1, words[indices[-(i + 1)]], w[indices[-(i + 1)]])
    
    
## Identify the worst errors on the test set
X_wrong = X_test[predictions_test != Y_test, :]
predictions_wrong, logit_wrong = multinomial_naive_bayes_inference(X_wrong, w, b)
logit = list(np.absolute(logit_wrong))
DICT = dict.fromkeys(logit, X_wrong)
indices = np.argsort(logit)
for i in range(5):
    logit_i = logit[indices[i+1]]
    text = DICT[logit_i]
    print('text in BoW representation: ', text)
    
    