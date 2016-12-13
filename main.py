from typing import Tuple, List
import os.path

from bottle import route, run, request
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import corpus2csc
from sklearn.naive_bayes import MultinomialNB, BaseDiscreteNB
from sklearn.externals import joblib


def load_corpus(path) -> Tuple[List[str], List[List[str]]]:
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


def train_model(documents: List[List[str]], categories: List[str])\
        -> Tuple[BaseDiscreteNB, Dictionary]:
    """学習用API"""
    dictionary = Dictionary(documents)
    X = corpus2csc([dictionary.doc2bow(doc) for doc in documents]).T
    return MultinomialNB().fit(X, categories), dictionary


def predict(classifier: BaseDiscreteNB, dictionary: Dictionary,
            document: List[str]) -> str:
    """分類用API"""
    X = corpus2csc([dictionary.doc2bow(document)], num_terms=len(dictionary)).T
    return classifier.predict(X)[0]


@route('/train')
def train():
    categories, documents = load_corpus('corpus.txt')
    classifier, dictionary = train_model(documents, categories)
    joblib.dump((classifier, dictionary), 'model.pkl', compress=9)
    return "trained"


@route('/classify')
def classify():
    if os.path.exists('model.pkl'):
        classifier, dictionary = joblib.load('model.pkl')
        sentence = request.params.sentence.split()
        return predict(classifier, dictionary, sentence)
    else:
        return "model not trained"


run(host='localhost', port=8080)
