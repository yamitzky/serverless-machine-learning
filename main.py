from typing import Tuple, List

from gensim.corpora.dictionary import Dictionary
from gensim.matutils import corpus2csc
from sklearn.naive_bayes import MultinomialNB, BaseDiscreteNB


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


categories, documents = load_corpus('corpus.txt')
classifier, dictionary = train_model(documents, categories)

predict_sentence = 'a dollar of 115 yen or more at the market price of the trump market 4% growth after the latter half of next year'.split()  # NOQA
predict(classifier, dictionary, predict_sentence)
