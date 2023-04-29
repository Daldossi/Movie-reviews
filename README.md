# Movie-reviews

A sentiment analysis is a case of text classification in which the goal is to predict the emotional state of the writers of text messages. 
So, it is a classification problem, where the classes are the authors’ emotions (angry, delighted, sad, happy, disgusted... ).
The aim of my project is to implement a sentiment analysis trhough a multinomial Naive Bayes classifier in order to analyze 
the movie reviews written by the users of the IMDB website (www.imdb.com); in particular I will predict the polarity of the reviews 
distinguishing the positive comments (class 1) from the negative ones (class 0).
The data set is already divided in a training set of 25'000 reviews, a validation set of 12'500 and a test set of 12'500.
A subset of the training set is also available to allow faster experiments.

A_BuildVocabulary.py
B_FeatureExtraction.py
C_TrainClassifier.py
D_Analysis
F_DictionarySize.py
G_Comparison.py
porter.py : a library that helps me apply the stemming method
