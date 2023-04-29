# -*- coding: utf-8 -*-
"""
@author: alice
Title: Effects of the size of the vocabulary on the accuracy of the model
"""

from A_BuildVocabulary import *
from B_FeatureExtraction import *
from C_TrainClassifier import *
import numpy as np
import collections
import os 
import porter
import matplotlib.pyplot as plt



## 1. Initialize

n = 1000
voc = collections.Counter() 

# SET_tr = 'smalltrain'
SET_tr = 'train'
# SET = 'test'
SET = 'validation'
stopw = True
stemming = False


## 2. Build a vocabulary

if SET_tr == 'smalltrain':
    direction_pos_tr = 'smalltrain/pos/'
    direction_neg_tr = 'smalltrain/neg/'
elif SET_tr == 'train':
    direction_pos_tr = 'train/pos/'
    direction_neg_tr = 'train/neg/'
if SET == 'test':
    tlt = 'Test set'
    direction_pos = 'test/pos/'
    direction_neg = 'test/neg/'
elif SET == 'validation':
    tlt = 'Validation set'
    direction_pos = 'validation/pos/'
    direction_neg = 'validation/neg/'
    
for f in os.listdir(direction_pos_tr): # put the direction to the file as input
    filename = direction_pos_tr + f
    words = read_document(filename, stopw, 'stopwords.txt', stemming)
    # voc_tr.update(words)
    voc.update(words)
for f in os.listdir(direction_neg_tr):
    filename = direction_neg_tr + f
    words = read_document(filename, stopw, 'stopwords.txt', stemming)
    # voc_tr.update(words)
    voc.update(words)
# print('VOC1: ', voc, 'length: ', len(voc))
    
Accs_tr = []
Accs = []
    
for n in range(10):
    n = (n+1)*1000
    print('n:', n)
    
    if SET_tr == 'smalltrain':
        txt_voc = 'vocabulary_smalltrain.txt'
        write_vocabulary(voc, txt_voc, n)
    elif SET_tr == 'train':
        txt_voc = 'vocabulary_train.txt'
        write_vocabulary(voc, txt_voc, n)
    
    voc_n = load_vocabulary(txt_voc)
    # print('VOC2: ', voc, 'length: ', len(voc))
    
    
    ## 3. Feature extraction
    
    documents_tr = [] 
    labels_tr = [] 
    documents = [] 
    labels = []
    for f in os.listdir(direction_pos_tr):
        filename = direction_pos_tr + f
        bow_tr = create_bow(filename, voc_n)
        documents_tr.append(bow_tr)
        # collect class labels:
        labels_tr.append(1)
    for f in os.listdir(direction_neg_tr):
        filename = direction_neg_tr + f
        bow_tr = create_bow(filename, voc_n)
        documents_tr.append(bow_tr)
        # collect class labels:
        labels_tr.append(0)
    for f in os.listdir(direction_pos):
        filename = direction_pos + f
        bow = create_bow(filename, voc_n)
        documents.append(bow)
        # collect class labels:
        labels.append(1)
    for f in os.listdir(direction_neg):
        filename = direction_neg + f
        bow = create_bow(filename, voc_n)
        documents.append(bow)
        # collect class labels:
        labels.append(0)

        
    ## 4. Train classifier

    X_tr = np.stack(documents_tr) ## takes a list of vectors and build a matrix attaching them
    ## number of rows in X = number of files
    Y_tr = np.array(labels_tr)
    data_tr = np.concatenate([X_tr, Y_tr[:, None]], 1) # we merge in a single matrix
    X_train = data_tr[:, :-1]
    Y_train = data_tr[:, -1].astype(int)
    # print('shape of X = ', X.shape, 'shape of Y = ', Y.shape)
    w, b = multinomial_naive_bayes_train(X_train, Y_train)
    # predictions, logit = multinomial_naive_bayes_inference(X_train, w, b)
    # accuracy = (predictions == Y_train).mean()
    # print('Training accuracy =', accuracy * 100)
    # Accs_tr.append(accuracy)


    ## 5. Test or validation 

    X = np.stack(documents) # takes a list of vectors and build a matrix attaching them
    ## number of rows in X = number of files
    Y = np.array(labels)
    data = np.concatenate([X, Y[:, None]], 1)
    X_set = data[:, :-1]
    Y_set = data[:, -1].astype(int)
    predictions, logit = multinomial_naive_bayes_inference(X_set, w, b)
    accuracy = (predictions == Y_set).mean()
    print('Accuracy =', accuracy * 100)
    Accs.append(accuracy)


## 6. Plot the results
    
plt.plot([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], Accs)
plt.title(tlt)
plt.xlabel('Vocabulary size')
plt.ylabel('Accuracy')
plt.figure()

