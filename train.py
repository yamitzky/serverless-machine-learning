# -*- coding: utf-8 -*-
from main import load_corpus, train_model
from sklearn.externals import joblib


def train():
    categories, documents = load_corpus('corpus.txt')
    classifier, dictionary = train_model(documents, categories)
    joblib.dump((classifier, dictionary), 'model.pkl', compress=9)
    return "trained"


if __name__ == '__main__':
    train()
