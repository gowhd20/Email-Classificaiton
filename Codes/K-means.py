"""
DATA AND TEXT MINING COURSE WORK PART 3,
K-means
Logistic regression classifier
"""

print(__doc__)

import math
import nltk
import os
import random
#import numpy as np

from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from stop_words import get_stop_words

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from PreProcessing import PreProcessing

class_names = ['spam', 'ham']
class1 = 'spam'
class2 = 'ham'

class Kmeans(object):
    def __init__(self, kernel="linear", n_clusters=2, max_iter=1000, stop_words=True, stemmer=True, entity_remover=True, lowercase=True):
        if stop_words:
            self.en_stop = get_stop_words('en')
        else:
            self.en_stop = None
        self.pp = PreProcessing()
        self.cnt_vec = CountVectorizer(min_df=1, stop_words=self.en_stop, 
                tokenizer=lambda text: self.pp.entity_stem_tokner(text=text, stemmer=stemmer, entity_remover=entity_remover), decode_error='replace', lowercase=lowercase)
        self.tfidf = TfidfVectorizer(min_df=1, stop_words=self.en_stop, 
                tokenizer=lambda text: self.pp.entity_stem_tokner(text=text, stemmer=stemmer, entity_remover=entity_remover), decode_error='replace', lowercase=lowercase)
        self.tsne_model = TSNE(n_components=2, random_state=0)
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=0, max_iter=max_iter)    # two clusters (yes or no)
        self.lr_model = LogisticRegression()
        self.sgd_model = SGDClassifier()

###########################################################
## k-means 
    def kmeans_pipelined(self):
        pipeline = Pipeline([
            ('tfidf', self.tfidf),
            ('kmean', self.kmeans_model),
        ])

        corpus, feature_label = self.pp.sort_corpus(class1)
        corpus, feature_label = self.pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)

        train_set, test_set = self.pp.data_dividor(corpus, feature_label)

        #   define idf matrix
        pipeline.fit(train_set[0], train_set[1])

        pred = pipeline.predict(test_set[0])
        print(accuracy_score(test_set[1], pred))
        print(classification_report(test_set[1], pred, target_names=class_names))

###########################################################
## k-means 
"""
corpus, feature_label = pp.sort_corpus(class1)
corpus, feature_label = pp.sort_corpus_custom(class2, corpus=corpus, feature_label=feature_label)

train_set, test_set = pp.data_dividor(corpus, feature_label)

#   define idf matrix
idfmtx = tfidf.fit_transform(train_set[0])
#   train
kmeans_model.fit(idfmtx)

#   transform normal documents 
idfmtx_test = tfidf.transform(test_set[0])

class_names = ['spam', 'ham']
pred = kmeans_model.predict(idfmtx_test)

import numpy as np

tc = np.count_nonzero(pred)
if(len(pred) - tc) > tc:     # 1 is used to indicate 'spam' class
    true_class = 0
else:
    true_class = 1

_pred = []
for p in pred:
    if p == true_class:
        _pred.append(class2)
    else:
        _pred.append(class1)

print(accuracy_score(test_set[1], _pred))
"""

###########################################################
## logistic regression classification 
"""
corpus, feature_label = pp.sort_corpus(class1)
corpus, feature_label = pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)

train_set, test_set = pp.data_dividor(corpus, feature_label)

#   define idf matrix
idfmtx = tfidf.fit_transform(train_set[0])
#   train
sgd_model.fit(idfmtx, train_set[1])

#   transform normal documents 
idfmtx_test = tfidf.transform(test_set[0])

pred = sgd_model.predict(idfmtx_test)

print(accuracy_score(test_set[1], pred))
print(classification_report(test_set[1], pred, target_names=class_names))

#print(accuracy_score(test_set[1], kmeans_model.predict(idfmtx_test)))
"""

if __name__ == "__main__":
    kmean = Kmeans()
    kmean.kmeans()