"""
DATA AND TEXT MINING COURSE WORK PART 2,
NaiveBayer
1. Gaussian
2. Multi class
"""

print(__doc__)

import math
import nltk
import os
#import numpy as np
import matplotlib.pyplot as plt

from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from stop_words import get_stop_words

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score # use this instead of manual loop 
from PreProcessing import PreProcessing,DenseTransformer
from sklearn.feature_selection import SelectKBest, chi2

class1 = 'spam'
class2 = 'ham'
class_names = ['ham', 'spam']
itera = 10
features = 100000

class NaiveBayer(object):
    def __init__(self, binary=False, svd_features=5, pca_features=5, stop_words=True, stemmer=True, entity_remover=True, lowercase=True):
        if stop_words:
            self.en_stop = get_stop_words('en')
        else:
            self.en_stop = None
        self.pp = PreProcessing()
        self.tsne = TSNE(n_components=2, random_state=0)
        self.cnt_vec = CountVectorizer(min_df=1, stop_words=self.en_stop, binary=binary,
            tokenizer=lambda text: self.pp.entity_stem_tokner(text=text, stemmer=stemmer, entity_remover=entity_remover), decode_error='replace', lowercase=lowercase)
        self.tfidf = TfidfVectorizer(min_df=1, stop_words=self.en_stop, 
            tokenizer=lambda text: self.pp.entity_stem_tokner(text=text, stemmer=stemmer, entity_remover=entity_remover), decode_error='replace', lowercase=lowercase)
        self.hv = HashingVectorizer(stop_words=self.en_stop, 
            tokenizer=lambda text: self.pp.entity_stem_tokner(text=text, stemmer=stemmer, entity_remover=entity_remover), decode_error='replace', lowercase=lowercase)
        
        self.sbf = SelectKBest(chi2, k=features)
        self.pca = PCA(n_components=pca_features)
        self.svd = TruncatedSVD(n_components=svd_features, random_state=0)
        self.clfMNB = MultinomialNB()
        self.clfGNB = GaussianNB()
        self.clfBNB = BernoulliNB()
        self.to_dense = DenseTransformer()
        super(NaiveBayer, self).__init__()


###########################################################
## NaiveBayer -- GaussianNB() 

    def naivebayer_ga(self):
        corpus, feature_label = pp.sort_corpus(class1)
        corpus, feature_label = pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)

        train_set, test_set = pp.data_divider(corpus, feature_label)

        #   define idf matrix
        idfmtx = self.cnt_vec.fit_transform(train_set[0])
        #   train
        self.clfGNB.fit(idfmtx.toarray(), train_set[1])

        #   transform normal documents 
        idfmtx_test = self.cnt_vec.transform(test_set[0])

        #print(clf.predict(idfmtx_test))
        #print(accuracy_score(test_set[1], clfGNB.predict(idfmtx_test.toarray())))

        pred = self.clfGNB.predict(idfmtx_test.toarray())
        print(accuracy_score(test_set[1], pred))
        print(classification_report(test_set[1], pred, target_names=class_names))


###########################################################
## NaiveBayer -- GaussianNB() 
    def naivebayer_gnb_pipelined(self):
        pipeline = Pipeline([
            ('tfidf', self.tfidf),
            ('to_dense', self.to_dense),
            ('clf', self.clfGNB),
        ])

        corpus, feature_label = self.pp.sort_corpus(class1)
        corpus, feature_label = self.pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)
        results = []
        for c in range(itera):
            train_set, test_set = self.pp.data_divider(corpus, feature_label)

            #   train
            pipeline.fit(train_set[0], train_set[1])

            pred = pipeline.predict(test_set[0])
            results.append(accuracy_score(test_set[1], pred))
            print "iteration %s" % len(results)

        print sum(i for i in results)/itera
        #print(accuracy_score(test_set[1], pred))
        #print(classification_report(test_set[1], pred, target_names=class_names))

###########################################################
## NaiveBayer -- BernoulliNB() 
    def naivebayer_bnb_pipelined(self):
        pipeline = Pipeline([
            ('tfidf', self.cnt_vec),
            ('clf', self.clfBNB),
        ])

        corpus, feature_label = self.pp.sort_corpus(class1)
        corpus, feature_label = self.pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)
        results = []
        for c in range(itera):
            train_set, test_set = self.pp.data_divider(corpus, feature_label)

            #   train
            pipeline.fit(train_set[0], train_set[1])

            pred = pipeline.predict(test_set[0])
            results.append(accuracy_score(test_set[1], pred))
            print "iteration %s" % len(results)

        print sum(i for i in results)/itera
        #print(accuracy_score(test_set[1], pred))
        #print(classification_report(test_set[1], pred, target_names=class_names))

###########################################################
## NaiveBayer -- MultinomialNB() 
    def naivebayer_mnb_svd_pipelined(self):
        pipeline = Pipeline([
            ('tfidf', self.tfidf),
            ('svd', self.svd),
            ('clf', self.clfMNB),
        ])

        corpus, feature_label = self.pp.sort_corpus(class1)
        corpus, feature_label = self.pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)
        results = []
        for c in range(itera):
            train_set, test_set = self.pp.data_divider(corpus, feature_label)

            #   train
            pipeline.fit(train_set[0], train_set[1])

            pred = pipeline.predict(test_set[0])
            results.append(accuracy_score(test_set[1], pred))
            print "iteration %s" % len(results)

        print sum(i for i in results)/itera
        #print(accuracy_score(test_set[1], pred))
        #print(classification_report(test_set[1], pred, target_names=class_names))


###########################################################
## NaiveBayer -- MultinomialNB() 
    def naivebayer_mnb_pipelined(self):
        pipeline = Pipeline([
            ('tfidf', self.cnt_vec),
            ('clf', self.clfMNB),
        ])

        corpus, feature_label = self.pp.sort_corpus(class1)
        corpus, feature_label = self.pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)
        results = []
        for c in range(itera):
            train_set, test_set = self.pp.data_divider(corpus, feature_label)

            #   train
            pipeline.fit(train_set[0], train_set[1])

            pred = pipeline.predict(test_set[0])
            results.append(accuracy_score(test_set[1], pred))
            print "iteration %s" % len(results)

        print sum(i for i in results)/itera
        #print(accuracy_score(test_set[1], pred))
        #print(classification_report(test_set[1], pred, target_names=class_names))

###########################################################
## NaiveBayer -- MultinomialNB() 
    def naivebayer_mnb(self, weight_model):
        corpus, feature_label = self.pp.sort_corpus(class1)
        corpus, feature_label = self.pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)
        train_set, test_set = self.pp.data_divider(corpus, feature_label)

        if weight_model == 'tfidf':
            print "vector model"
            model = self.tfidf
        elif weight_model == 'count':
            print "count model"
            model = self.cnt_vec
        #   define idf matrix
        idfmtx = model.fit_transform(train_set[0])
        idfmtx_test = model.transform(test_set[0])
        print idfmtx[0].shape
        print idfmtx[1].shape

        idfmtx = self.sbf.fit_transform(idfmtx, train_set[1])
        idfmtx_test = self.sbf.transform(idfmtx_test)

        #   train
        self.clfMNB.fit(idfmtx, train_set[1])


        #print(clf.predict(idfmtx_test))
        #print(accuracy_score(test_set[1], clfGNB.predict(idfmtx_test.toarray())))
        scores = cross_val_score(self.clfMNB, idfmtx, train_set[1], cv=100)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        pred = self.clfMNB.predict(idfmtx_test)
        print(accuracy_score(test_set[1], pred))
        print(classification_report(test_set[1], pred, target_names=class_names))

        cnf_mtx = confusion_matrix(test_set[1], pred)
        plt.figure()
        self.pp.plot_confusion_matrix(cnf_mtx, classes=class_names, title="confusion matrix, Naive Bayes classifier")
        plt.show()

"""
clf = GaussianNB()

for l in range(file_total_count):
    sort_corpus(l, class1)
    sort_corpus(l, class2)

train_set, test_set = data_dividor(corpus, feature_label)

#   define idf matrix
idfmtx = tfidf.fit_transform(train_set[0])
#   train
clf.fit(idfmtx.toarray(), train_set[1])

feature_label = []

#   transform normal documents 
idfmtx_test = tfidf.transform(test_set[0])

#print(clf.predict(idfmtx_test))
cnf_matrix = confusion_matrix(test_set[1], clf.predict(idfmtx_test.toarray()))
corpus_test = []
corpus = []
feature_label = []

class_names = ['spam', 'ham']

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
"""

if __name__ == "__main__":

    clf = NaiveBayer(binary=True, stop_words=False, stemmer=False, entity_remover=False, lowercase=False)
    print features
    #clf = NaiveBayer(binary=False)
    clf.naivebayer_mnb("count")