"""
DATA AND TEXT MINING COURSE WORK PART 4,
K Nearest Neighbors (KNN)

"""

print(__doc__)

import math
import nltk
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from nltk.stem.porter import *
from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from stop_words import get_stop_words
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, chi2
from PreProcessing import PreProcessing

class_names = ['ham', 'spam']
class1 = 'spam'
class2 = 'ham'
itera = 10
features = 3000

class KNN(object):
    def __init__(self, binary=False, n_neighbors=3, svd_features=5, stop_words=True, stemmer=True, entity_remover=True, lowercase=True):
        if stop_words:
            self.en_stop = get_stop_words('en')
        else:
            self.en_stop = None
        self.pp = PreProcessing()
        self.tsne = TSNE(n_components=2, random_state=0)
        self.svd = TruncatedSVD(n_components=svd_features, random_state=0)
        self.cnt_vec = CountVectorizer(min_df=1, stop_words=self.en_stop, binary=binary,
            tokenizer=lambda text: self.pp.entity_stem_tokner(text=text, stemmer=stemmer, entity_remover=entity_remover), decode_error='replace', lowercase=lowercase)
        self.tfidf = TfidfVectorizer(min_df=1, stop_words=self.en_stop, 
            tokenizer=lambda text: self.pp.entity_stem_tokner(text=text, stemmer=stemmer, entity_remover=entity_remover), decode_error='replace', lowercase=lowercase)
        self.hv = HashingVectorizer(stop_words=self.en_stop, 
            tokenizer=lambda text: self.pp.entity_stem_tokner(text=text, stemmer=stemmer, entity_remover=entity_remover), decode_error='replace', lowercase=lowercase)
        self.k = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance") #metric='Euclidean'
        self.sbf = SelectKBest(chi2, k=features)
        super(KNN, self).__init__()

###########################################################
## KNN 

    def knn(self):
        pipeline = Pipeline([
            ('tfidf', self.tfidf),
            ('knn', self.k),
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

###########################################################
## KNN with plot

    def kplot(self):
        """
        n_neighbors = 15

        # import some data to play with
        iris = datasets.load_iris()
        
        X = iris.data[:, :2]  # we only take the first two features. We could
                              # avoid this ugly slicing by using a two-dim dataset
        print X.shape
        y = iris.target
        """
        h = .02  # step size in the mesh

        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])#, '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00'])#, '#0000FF'])

        corpus, feature_label = self.pp.sort_corpus(class1)
        corpus, feature_label = self.pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)
        train_set, test_set = self.pp.data_divider(corpus, feature_label)

        idfmtx = self.tfidf.fit_transform(train_set[0])
        idfmtx_test = self.tfidf.transform(test_set[0])

        xtrain = self.svd.fit_transform(idfmtx)
        xtest = self.svd.transform(idfmtx_test)
        print xtrain.shape
        print xtest.shape
        self.k.fit(xtrain, train_set[1])

        pred = self.k.predict(xtest)
        print(accuracy_score(test_set[1], pred))
        print(classification_report(test_set[1], pred, target_names=class_names))

        #for weights in ['uniform', 'distance']:

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = xtest[:, 0].min() - 1, xtest[:, 0].max() + 1
        y_min, y_max = xtest[:, 1].min() - 1, xtest[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = self.k.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(xtest[:, 0], xtest[:, 1], c=test_set[1], cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("2-Class classification (k = %i, weights = '%s')"
                  % (10, "distance"))

        plt.show()

###########################################################
## KNN with error rate, plot
    def knn_err_plot(self, weight_model):
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

        #idfmtx = self.sbf.fit_transform(idfmtx, train_set[1])
        #idfmtx_test = self.sbf.transform(idfmtx_test)

        #   train
        self.k.fit(idfmtx, train_set[1])


        #print(clf.predict(idfmtx_test))
        #print(accuracy_score(test_set[1], clfGNB.predict(idfmtx_test.toarray())))
        scores = cross_val_score(self.k, idfmtx, train_set[1], cv=100)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        pred = self.k.predict(idfmtx_test)
        print(accuracy_score(test_set[1], pred))
        print(classification_report(test_set[1], pred, target_names=class_names))

        cnf_mtx = confusion_matrix(test_set[1], pred)
        plt.figure()
        self.pp.plot_confusion_matrix(cnf_mtx, classes=class_names, title="confusion matrix, KNN classifier")
        plt.show()



if __name__ == "__main__":

    clf = KNN(binary=False, svd_features=2, n_neighbors=10, stop_words=False, stemmer=False, entity_remover=False, lowercase=False)
    #clf = KNN(n_neighbors=5)
    clf.knn_err_plot("tfidf")