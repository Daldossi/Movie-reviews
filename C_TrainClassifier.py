# -*- coding: utf-8 -*-
"""
@author: alice
Title: Binary classifier
"""
import numpy as np


# def multinomial_naive_bayes_train(X, Y):
#     '''
#     Estimate the coefficients of the Naive Bayes classifier

#     Parameters
#     ----------
#     X : data in input
#     Y : classification of the training data

#     Returns
#     -------
#     w : weights, coefficient of x
#     b : bias, it is equal to log(P(y))

#     '''    
#     m, n = X.shape
#     print('n =', n, 'm =', m)
#     k = int(Y.max() + 1)
#     print('k =', k)
#     probs = np.empty((k, n))
#     for c in range(k):
#         counts = X[Y == 1, :].sum(0)
#         tot = counts.sum()
#         probs[c, :] = (counts + 1) / (tot + n)
#     priors = np.bincount(Y) / m
#     W = np.log(probs)
#     w = W[1, :] - W[0, :]
#     B = np.log(priors)
#     b = B[1] - B[0]
#     return w, b


def multinomial_naive_bayes_train(X, Y):
    '''
    Estimate the coefficients of the Naive Bayes classifier

    Parameters
    ----------
    X : data in input
    Y : classification of the training data

    Returns
    -------
    w : weights, coefficient of x
    b : bias, it is equal to log(P(y))

    ''' 
    ## Positive class:
    pos_pi = X[Y == 1, :].sum(0) # sum(0) does the sum row by row
    pos_pi = pos_pi + 1 # Laplacian smoothing
    pos_pi /= pos_pi.sum() # normalization
    ## Negative class:
    neg_pi = X[Y == 0, :].sum(0) 
    neg_pi = neg_pi + 1 # Laplacian smoothing
    neg_pi /= neg_pi.sum() 
    ## Estimate the weights:
    w = np.log(pos_pi) - np.log(neg_pi)
    ## Estimate the bias:
    pos_prior = Y.sum() / Y.size # sum of Y gives the number of elements of class 1
    neg_prior = 1 - pos_prior # we have 2 classes, so we can calculate the number of neg subtracting the positive's
    b = np.log(pos_prior) - np.log(neg_prior)
    return w, b
    
    
def multinomial_naive_bayes_inference(X, w, b):
    '''
    Assign each X_j to a class using the Naive Bayes classifier with 
    coefficient W and b

    Parameters
    ----------
    X : data in input
    W : weight of the trained model for the Naive Bayes classifier
    b : bias of the trained model for the Naive Bayes classifier

    Returns 
    ----------
    Classification in the 2 classes (1: positive, 0: negative)
    Score : logit

    '''
    ## z1-z0 = (W1^T x + b1 - W0^T x - b0) = (W1^T-W0^T) x + (b1-b0) = w x + b
    ## so w is a vector!
    score = X @ w + b # a single score for each document
    labels = (score > 0).astype(int)
    return labels, score # look at the sign of each score
    



# -----------------------------------------------
#   MAIN function
# -----------------------------------------------

if __name__ == "__main__":
    
    ## 1. TRAIN THE MODEL
    
    # SET = 'smalltrain'
    SET = 'train' 
    if SET == 'smalltrain':
        data = np.loadtxt('smalltrain.txt.gz')
        f = open('vocabulary_smalltrain.txt')
    elif SET == 'train':
        data = np.loadtxt('train.txt.gz')
        f = open('vocabulary_train.txt')
    
    X_train = data[:, :-1]
    Y_train = data[:, -1].astype(int)
    # print('shape of X = ', X.shape, 'shape of Y = ', Y.shape)
    w, b = multinomial_naive_bayes_train(X_train, Y_train)
    # print('w = ', w)
    # print('b = ', b)
    predictions, _ = multinomial_naive_bayes_inference(X_train, w, b)
    accuracy = (predictions == Y_train).mean()
    print('Training accuracy =', accuracy * 100)
    
    
    ## 2. TEST SET
    
    data = np.loadtxt('test.txt.gz')
    X_test = data[:, :-1]
    Y_test = data[:, -1].astype(int)
    predictions_test, logit = multinomial_naive_bayes_inference(X_test, w, b)
    accuracy = (predictions_test == Y_test).mean()
    print('Test accuracy =', accuracy * 100)
    
    
    ## 3. VALIDATION SET
    
    data = np.loadtxt('validation.txt.gz')
    X_val = data[:, :-1]
    Y_val = data[:, -1].astype(int)
    predictions, _ = multinomial_naive_bayes_inference(X_val, w, b)
    accuracy = (predictions == Y_val).mean()
    print('Validation accuracy =', accuracy * 100)
    
    
    ## 4. SAVE THE RESULTS
    
    np.savez("model.npz", SET, X_train, Y_train, X_test, Y_test, predictions_test, w, b)


