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


    # read all the negative reviews in the test folder. Please change the file path here if needed
    # filelist = glob.glob('/Users/akshaypala/Downloads/aclImdb/test/neg/*.txt')
    filelist = glob.glob('assignment_3/data/aclImdb/test/neg/*.txt')
    neg_list = []
    for fle in filelist:
        with open(fle) as f:
            contents = f.read()
            neg_list.append(contents)

    # read all the positive reviews in the test folder. Please change the file path here if needed
    # filelist = glob.glob('/Users/akshaypala/Downloads/aclImdb/test/pos/*.txt')
    filelist = glob.glob('assignment_3/data/aclImdb/test/pos/*.txt')
    pos_list = []
    for fle in filelist:
        with open(fle) as f:
            contents = f.read()
            pos_list.append(contents)

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


    # ******************Cleaning the dataset is the most PROLONGED PERIOD in the training phase. This part can take up to 15 hours.********************** 
    # clean the raw review list. Remove any numbers, special chars and punctuation. 
    punctuation = re.compile(r'[-.?!,":;()|0-9\']')
    punctuation2 = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    for review in neg_list:
        neg_list = [punctuation.sub("", review) for review in neg_list]
        neg_list = [punctuation2.sub("", review) for review in neg_list]

    for review in pos_list:
        pos_list = [punctuation.sub("", review) for review in pos_list]
        pos_list = [punctuation2.sub("", review) for review in pos_list]

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
    neg_list_lower = [k.lower() for k in neg_list]
    pos_list_lower = [k.lower() for k in pos_list]
    neg_list_train_lower = [k.lower() for k in neg_list_train]
    pos_list_train_lower = [k.lower() for k in pos_list_train]


    clean_reviews_test = pos_list_lower + neg_list_lower
    clean_reviews_train = pos_list_train_lower + neg_list_train_lower

    # stemming
    stemmer = PorterStemmer()
    clean_reviews_train = [' '.join([stemmer.stem(word) for word in review.split()]) for review in clean_reviews_train]
    clean_reviews_test = [' '.join([stemmer.stem(word) for word in review.split()]) for review in clean_reviews_test]
    
    # lemmatization
    # lemmatizer = WordNetLemmatizer()
    # clean_reviews_train = [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in clean_reviews_train]
    # clean_reviews_test = [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in clean_reviews_test]


    # ngram vectorizer to fit the model
    ngram_vectorizer = CountVectorizer(stop_words='english', binary=True, ngram_range=(1, 2))
    ngram_vectorizer.fit(clean_reviews_train)
    X = ngram_vectorizer.transform(clean_reviews_train)
    X_test = ngram_vectorizer.transform(clean_reviews_test)


    # class or target. 1 = positive and 0 = negative
    list_class=[]
    negative_class=np.zeros(12500,dtype=int)
    for i in negative_class:
        list_class.append(i)

    positive_class=np.ones(12500,dtype=int)
    for i in positive_class:
        list_class.append(i)


    # load the model and test. Please change the file path here if needed
    filename = 'assignment_3/models/ECE657/20469485_NLP_Model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    loaded_model.fit(X, list_class)
    print ("Final Accuracy: %s"
           % accuracy_score(list_class, loaded_model.predict(X_test)))