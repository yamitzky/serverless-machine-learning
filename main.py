# -*- coding: utf-8 -*-
import os
import ctypes


for d, dirs, files in os.walk('lib'):
    for f in files:
        if f.endswith('.a'):
            continue
        ctypes.cdll.LoadLibrary(os.path.join(d, f))


from gensim.corpora.dictionary import Dictionary
from gensim.matutils import corpus2csc
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


def load_corpus(path):
    """コーパスをファイルから取得"""
    categories = []
    docs = []
    with open(path) as f:
        for line in f:
            category, line = line.split('\t')
            doc = line.strip().split(' ')
            categories.append(category)
            docs.append(doc)
    return categories, docs


def train_model(documents, categories):
    """学習用API"""
    dictionary = Dictionary(documents)
    X = corpus2csc([dictionary.doc2bow(doc) for doc in documents]).T
    return MultinomialNB().fit(X, categories), dictionary


def predict(classifier, dictionary, document):
    """分類用API"""
    X = corpus2csc([dictionary.doc2bow(document)], num_terms=len(dictionary)).T
    return classifier.predict(X)[0]


def classify(event, context):
    if os.path.exists('model.pkl'):
        classifier, dictionary = joblib.load('model.pkl')
        sentence = event['sentence'].split()
        return predict(classifier, dictionary, sentence)
    else:
        return "model not trained"
