# -*- coding: utf-8 -*-
"""
@author: alice

The script reads all the training data (in the smalltrain directory) and uses it 
to build a list of the n most frequent words; let's think of this list as a 
vocabulary.
Writes them to the 'vocabulary.txt' file
"""

import collections # collections contains a few data structures, including the counter
# counter is a dictionary specialized in counting things
import os 
from porter import *


def remove_punctuation(text):
    '''    Remove punctuation from the words in the list text    '''
    punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~«»"
    for p in punct:
        text = text.replace(p, ' ')
    return text


# def read_document(filename):
#     '''    Takes the list of the words    '''
#     f = open(filename, encoding='utf-8') # we force utf-8 to silent possible errors
#     text = f.read() 
#     f.close()
#     words = [] # set an empty list that we will fill with all the words in the file
#     text = text.lower()
#     text = remove_punctuation(text)
#     for word in text.split(): # separate with a space the words in the text
#             if len(word) > 2:
#                 words.append(word)
#     return words


def read_document(filename, stopw=False, file_stop=None, stemming=False): 
    '''    Takes the list of the words with or without stop-word or stemming    ''' 
    f = open(filename, encoding='utf-8') # we force utf-8 to silent possible errors
    text = f.read()
    f.close() 
    words = [] # set an empty list that we will fill with all the words in the file 
    text = text.lower() 
    text = remove_punctuation(text) 
    if stopw == True and stemming == False: 
        f_stop = open(file_stop, encoding='utf-8') 
        text_stop = f_stop.read() 
        f_stop.close() 
        for word in text.split(): # separate with a space the words in the text 
                if word not in text_stop.split():
                    if len(word) > 2:
                        words.append(word) 
    elif stopw == False and stemming == True: 
        for word in text.split(): 
                stem_word = stem(word) 
                if len(stem_word) > 2: 
                    words.append(stem_word) 
    elif stopw == False and stemming == False:
        for word in text.split(): 
            if len(word) > 2: 
                words.append(word) 
    elif stopw == True and stemming == True:
        f_stop = open(file_stop, encoding='utf-8') 
        text_stop = f_stop.read() 
        f_stop.close() 
        for word in text.split(): # separate with a space the words in the text 
            stem_word = stem(word) 
            if stem_word not in text_stop.split():
                if len(word) > 2:
                    words.append(stem_word) 
    return words 


def write_vocabulary(voc, filename, n): 
    f = open(filename, 'w') 
    for word, count in voc.most_common(n): # method that extracts the n most common words
        print(word, file=f) # the destination of the print is not the terminal, but instead the file
    f.close() 
        



# -----------------------------------------------
#   MAIN function
# -----------------------------------------------

# Create the vocabulary in a txt file
if __name__ == "__main__":
    n = 1000 # hyperparameter
    voc = collections.Counter()
    
    # SET = 'smalltrain'
    SET = 'train'
    stopw = True 
    stemming = False
    
    if SET == 'smalltrain':
        direction_pos = 'smalltrain/pos/'
        direction_neg = 'smalltrain/neg/'
    elif SET == 'train':
        direction_pos = 'train/pos/'
        direction_neg = 'train/neg/'
    
    for f in os.listdir(direction_pos): 
        filename = direction_pos + f
        words = read_document(filename, stopw, 'stopwords.txt', stemming)
        voc.update(words)
    for f in os.listdir(direction_neg):
        filename = direction_neg + f
        words = read_document(filename, stopw, 'stopwords.txt', stemming)
        voc.update(words)
    
    if SET == 'smalltrain':
        write_vocabulary(voc, 'vocabulary_smalltrain.txt', n)
    elif SET == 'train':
        write_vocabulary(voc, 'vocabulary_train.txt', n)

## vocabulary.txt contains all the words; each one is put in a row and they are 
## ordered from the most common to the least

