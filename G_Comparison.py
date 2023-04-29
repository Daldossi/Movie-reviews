# -*- coding: utf-8 -*-
"""
@author: alice
Title: Classification using logistic regression or Support Vector Machine
"""
import numpy as np


## LOGISTIC REGRESSION FUNCTIONS

def logreg_inference(x, w, b):
    logit = x @ w + b
    probability = 1 / (1 + np.exp(-logit))
    return probability

def cross_entropy(P, Y):
    loss = (-Y * np.log(P) - (1 - Y) * np.log(1 - P)).mean()
    return loss

def logreg_train(X, Y, lr=0.05):
    '''    Optimization algorithm of the gradient descent    '''    
        # 1. Starting point
    m, n = X.shape
    b = 0
    w = np.zeros(n)
        # 2. Perform some amount of iterations (1000)
    for i in range(100000):
        P = logreg_inference(X, w, b) 
        grad_w = X.T @ (P - Y) / m 
        grad_b = (P - Y).mean() 
        # print(i) 
        w -= lr * grad_w
        b -= lr * grad_b
        if i % 100 == 0: # print only for multiplier if 100
            predictions = (P > 0.5)
            accuracy = (predictions == Y).mean()
            loss = cross_entropy(P, Y)
            # print('Step:', i, 'Loss:', loss, 'Accuracy:', accuracy * 100)
    return w, b


## SUPPORT VECTOR MACHINE FUNCTIONS

def ksvm_train(X, Y, kfun, lambda_, lr=1e-3, steps=1000):
    K = kernel(X, X, kfun, kparam)
    m, n = X.shape
    alpha = np.zeros(m)
    b = 0
    for steps in range(steps):
        ka = K @ alpha
        z = ka + b
        hinge_diff = -Y * (z < 1) + (1 - Y) * (z > -1)
        grad_alpha = (hinge_diff @ K) / m + lambda_ * ka
        grad_b = hinge_diff.mean()
        alpha -= lr * grad_alpha
        b -= lr * grad_b
    return alpha, b

def ksvm_inference(X, Xtrain, alpha, b, kfun, kparam):
    K = kernel(X, Xtrain, kfun, kparam)
    z = K @ alpha + b
    labels = (z > 0).astype(int)
    return labels

def kernel(X1, X2, kfun, kparam):
    if kfun == 'polynomial':
        return (X1 @ X2.T + 1) ** kparam
    elif kfun == 'rbf':
        qx1 = (X1 ** 2).sum(1, keepdims=True)
        qx2 = (X2 ** 2).sum(1, keepdims=True)
        cross = 2 * X1 @ X2.T
        return np.exp(-kparam * (qx1 - cross + qx2.T))
    else:
        raise ValueError("Unknown kernel ('%s')" % kfun)




# -----------------------------------------------
#   MAIN function
# -----------------------------------------------

if __name__ == "__main__":
    
    # method = 'SVM'
    method = 'logistic regression'
    
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
    
    data_test = np.loadtxt('test.txt.gz')
    X_test = data_test[:, :-1]
    Y_test = data_test[:, -1].astype(int)
    
    data_val = np.loadtxt('validation.txt.gz')
    X_val = data_val[:, :-1]
    Y_val = data_val[:, -1].astype(int)
    
    if method == 'logistic regression':
        ## 0. PARAMETERS
        lr = 0.001
        
        ## 1. TRAIN THE MODEL
        w, b = logreg_train(X_train, Y_train, lr)
        # print('w = ', w)
        # print('b = ', b)
        P = logreg_inference(X_train, w, b)
        predictions = (P > 0.5)
        accuracy = (predictions == Y_train).mean()
        print('Training accuracy =', accuracy * 100)
        
        ## 2. TEST THE MODEL
        P = logreg_inference(X_test, w, b)
        predictions = (P > 0.5)
        accuracy = (predictions == Y_test).mean()
        print('Test accuracy =', accuracy * 100)
        
        ## 3. VALIDATION SET
        P = logreg_inference(X_val, w, b)
        predictions = (P > 0.5)
        accuracy = (predictions == Y_val).mean()
        print('Validation accuracy =', accuracy * 100)
    
    elif method == 'SVM':
        ## 0. PARAMETERS
        kfun = 'rbf'
        lambda_ = 1
        lr = 0.001
        steps = 1000
        kparam = 2

        ## 1. TRAIN THE MODEL
        alpha, b = ksvm_train(X_train, Y_train, kfun, lambda_, lr, steps)
        predictions = ksvm_inference(X, Xtrain, alpha, b, kfun, kparam)
        accuracy = (predictions == Y_train).mean()
        print('Training accuracy =', accuracy * 100)
                    
        ## 2. TEST THE MODEL
        predictions_test = logreg_inference(X_test, w, b)
        accuracy = (predictions == Y_test).mean()
        print('Test accuracy =', accuracy * 100)
        
        ## 3. VALIDATION SET
        predictions = logreg_inference(X_val, w, b)
        accuracy = (predictions == Y_val).mean()
        print('Validation accuracy =', accuracy * 100)

