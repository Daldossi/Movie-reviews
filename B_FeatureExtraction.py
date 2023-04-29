# -*- coding: utf-8 -*-
"""
@author: alice
Title: Feature extraction
"""
import numpy as np # we need numpy because the BoW will create vectors
import os 


def load_vocabulary(filename):
    '''    Creates a vocabulary that maps all the words to their position 
    (it will be the position in the BoW representation)    '''
    voc = {}
    f = open(filename)
    text = f.read()
    n = 0 # counter that counts the words it reads
    for word in text.split():
        voc[word] = n
        n += 1
    f.close()
    return voc


def remove_punctuation(text):
    '''    Remove punctuation from the words in the list text    '''
    punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for p in punct:
        text = text.replace(p, ' ')
    return text


def create_bow(filename, voc):
    '''    Create the Bag of Words (bow) representation    '''
    f = open(filename, encoding='utf-8') # we force utf-8 in order to silence possible errors
    text = f.read() 
    f.close()
    bow = np.zeros(len(voc)) # Bag of Word representation
    text = text.lower() # remove upper letters in the words
    text = remove_punctuation(text) # remove punctuation
    for word in text.split(): # increment the counter in the BoW representaion
        if word in voc:
            index = voc[word]
            bow[index] += 1
    return bow




# -----------------------------------------------
#   MAIN function
# -----------------------------------------------

if __name__ == "__main__":
    # SET = 'smalltrain'
    SET = 'train'
    # SET = 'test'
    # SET = 'validation'
    documents = []
    labels = []
    
    if SET == 'smalltrain':
        voc = load_vocabulary('vocabulary_smalltrain.txt')
        # print(voc)
        direction_pos = 'smalltrain/pos/'
        direction_neg = 'smalltrain/neg/'
    elif SET == 'train':
        voc = load_vocabulary('vocabulary_train.txt')
        direction_pos = 'train/pos/'
        direction_neg = 'train/neg/'
    elif SET == 'test':
        voc = load_vocabulary('vocabulary_train.txt')
        direction_pos = 'test/pos/'
        direction_neg = 'test/neg/'
    elif SET == 'validation':
        voc = load_vocabulary('vocabulary_train.txt')
        direction_pos = 'validation/pos/'
        direction_neg = 'validation/neg/'
            
    for f in os.listdir(direction_pos): 
        filename = direction_pos + f
        bow = create_bow(filename, voc)
        documents.append(bow)
        # collect class labels:
        labels.append(1)
    for f in os.listdir(direction_neg):
        filename = direction_neg + f
        bow = create_bow(filename, voc)
        documents.append(bow)
        # collect class labels:
        labels.append(0)
    
    X = np.stack(documents) # takes a list of vectors and build a matrix attaching them
    ## number of rows in X = number of files
    Y = np.array(labels)
    data = np.concatenate([X, Y[:, None]], 1) # we merge in a single matrix
    
    if SET == 'smalltrain':
        # np.savetxt('train.txt', data)
        ## Issue: this representation could be quite large
        ## Solution: work with compressed files adding .gz as follow
        np.savetxt('smalltrain.txt.gz', data)
    elif SET == 'train':
        np.savetxt('train.txt.gz', data)
    elif SET == 'test':
        np.savetxt('test.txt.gz', data)
    elif SET == 'validation':
        np.savetxt('validation.txt.gz', data)

