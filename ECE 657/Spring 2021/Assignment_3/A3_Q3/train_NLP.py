import glob
import re
import nltk
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":

    # read all the negative reviews in the train folder. Please change the file path here if needed
    # filelist = glob.glob('/Users/akshaypala/Downloads/aclImdb/train/neg/*.txt')
    filelist = glob.glob('assignment_3/data/aclImdb/train/neg/*.txt')
    neg_list_train = []
    for fle in filelist:
        with open(fle) as f:
            contents = f.read()
            neg_list_train.append(contents)

    # read all the positive reviews in the train folder. Please change the file path here if needed
    # filelist = glob.glob('/Users/akshaypala/Downloads/aclImdb/train/pos/*.txt')
    filelist = glob.glob('assignment_3/data/aclImdb/train/pos/*.txt')
    pos_list_train = []
    for fle in filelist:
        with open(fle) as f:
            contents = f.read()
            pos_list_train.append(contents)


    # Cleaning the dataset is the most prolonged period in the training phase. This part can take up to 15 hours. 
    # clean the raw review train list. Remove any numbers, special chars and punctuation
    punctuation = re.compile(r'[-.?!,":;()|0-9\']')
    punctuation2 = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    for review in neg_list_train:
        neg_list_train = [punctuation.sub("", review) for review in neg_list_train]
        neg_list_train = [punctuation2.sub("", review) for review in neg_list_train]

    for review in pos_list_train:
        pos_list_train = [punctuation.sub("", review) for review in pos_list_train]
        pos_list_train = [punctuation2.sub("", review) for review in pos_list_train]

    # convert to lower case
    neg_list_train_lower = [k.lower() for k in neg_list_train]
    pos_list_train_lower = [k.lower() for k in pos_list_train]


    # combine train data into their own list
    clean_reviews_train = pos_list_train_lower + neg_list_train_lower

    # stemming
    stemmer = PorterStemmer()
    clean_reviews_train = [' '.join([stemmer.stem(word) for word in review.split()]) for review in clean_reviews_train]
    
    # lemmatization
    # lemmatizer = WordNetLemmatizer()
    # clean_reviews_train = [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in clean_reviews_train]


    # using ngram vectorizer to fit the model
    ngram_vectorizer = CountVectorizer(stop_words='english', binary=True, ngram_range=(1, 2))
    ngram_vectorizer.fit(clean_reviews_train)
    X = ngram_vectorizer.transform(clean_reviews_train)


    # create a target list, 1 = positive and 0 = negative
    list_class=[]
    negative_class=np.zeros(12500,dtype=int)
    for i in negative_class:
        list_class.append(i)

    positive_class=np.ones(12500,dtype=int)
    for i in positive_class:
        list_class.append(i)


    # split the dataset into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, list_class, train_size = 0.75)


    # train the models
    for c in [0.25, 0.5, 1, 1.50, 2, 3]:
        lr = LogisticRegression(C=c)
        clf = svm.SVC(C=c)
        mnb = MultinomialNB()
        dt = tree.DecisionTreeClassifier()
        lr.fit(X_train, y_train)
        clf.fit(X_train, y_train)
        mnb.fit(X_train, y_train)
        dt.fit(X_train, y_train)
        print("Accuracy for SVMC=%s: %s"
              % (c, accuracy_score(y_val, clf.predict(X_val))))
        print("Accuracy for LRC=%s: %s"
              % (c, accuracy_score(y_val, lr.predict(X_val))))
        print("Accuracy for MNB=%s: %s"
              % (c, accuracy_score(y_val, mnb.predict(X_val))))
        print("Accuracy for dt=%s: %s"
              % (c, accuracy_score(y_val, dt.predict(X_val))))


    # save the model. Please change the file path here if needed
    filename = 'assignment_3/models/20469485_NLP_Model.sav'
    pickle.dump(lr, open(filename, 'wb'))