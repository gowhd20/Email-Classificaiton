"""
DATA AND TEXT MINING COURSE WORK PART 1,
SVM
1. SVC (linear SVM)
2. PCA + SVM
3. LDA classifier 
"""

print(__doc__)

import math
import nltk
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer   # needs to be tested

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from stop_words import get_stop_words
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

from PreProcessing import PreProcessing, DenseTransformer

class1 = 'spam'
class2 = 'ham'
class_names = ['ham', 'spam']
itera = 10
features = 100000

class SVM(object):
    def __init__(self, binary=False, kernel="linear", lda_features=5, svd_features=5, pca_features=5, stop_words=True, stemmer=True, entity_remover=True, lowercase=True):
        if stop_words:
            self.en_stop = get_stop_words('en')
        else:
            self.en_stop = None
        self.pp = PreProcessing()
        self.cnt_vec = CountVectorizer(min_df=1, stop_words=self.en_stop, binary=binary,
            tokenizer=lambda text: self.pp.entity_stem_tokner(text=text, stemmer=stemmer, entity_remover=entity_remover), decode_error='replace', lowercase=lowercase)
        self.tfidf = TfidfVectorizer(min_df=1, stop_words=self.en_stop, 
            tokenizer=lambda text: self.pp.entity_stem_tokner(text=text, stemmer=stemmer, entity_remover=entity_remover), decode_error='replace', lowercase=lowercase)
        self.hv = HashingVectorizer(stop_words=self.en_stop, 
            tokenizer=lambda text: self.pp.entity_stem_tokner(text=text, stemmer=stemmer, entity_remover=entity_remover), decode_error='replace', lowercase=lowercase)
        self.tsne = TSNE(n_components=2, verbose=1, random_state=0)
        self.sbf = SelectKBest(chi2, k=features)

        self.clf = svm.SVC(kernel=kernel)
        self.svd = TruncatedSVD(n_components=svd_features, random_state=0)
        self.pca = PCA(n_components=pca_features)
        self.lda = LinearDiscriminantAnalysis(n_components=lda_features)
        self.to_dense = DenseTransformer()

        self.rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=1)
        self.poly_svc = svm.SVC(kernel='poly', degree=3, C=1)
        self.lin_svc = svm.LinearSVC(C=1)

        super(SVM, self).__init__()



###########################################################
## LDA + partitioned data set -- see 'test_ratio' variable

    def lda(self):
        print "running lda classifier"
        corpus, feature_label = self.pp.sort_corpus(class1)
        corpus, feature_label = self.pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)

        train_set, test_set = self.pp.data_dividor(corpus, feature_label)

        #   define idf matrix
        idfmtx = self.tfidf.fit_transform(corpus)
        #   train
        lda.fit(idfmtx.toarray(), feature_label)

        idfmtx_test = self.tfidf.transform(test_set[0])

        #print lda.score(idfmtx_test, test_set[1])
        pred = lda.predict(idfmtx_test)
        print(accuracy_score(test_set[1], pred))

        print(classification_report(test_set[1], pred, target_names=class_names))

###########################################################
## LDA + partitioned data set -- see 'test_ratio' variable

    def lda_pipeline(self):
        print "running lda with pipeline"
        pipeline = Pipeline([
            ('tfidf', self.tfidf),
            ('to_dense', self.to_dense),
            ('lda', self.lda),
            ('clf', self.clf)
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
## PCA aself.pplied +  partitioned data set -- see 'test_ratio' variable
    def svc_pca(self):
        print "running SVM with PCA"
        corpus, feature_label = self.pp.sort_corpus(class1)
        corpus, feature_label = self.pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)

        train_set, test_set = self.pp.data_divider(corpus, feature_label)
        self.pp.cnt_avg_mail_length(train_set[0])
        print len(train_set[0])
        print len(test_set[0])
        #   define idf matrix
        idfmtx = self.tfidf.fit_transform(train_set[0])
        print idfmtx.shape
        #   train
        xtrain = self.pca.fit_transform(idfmtx.toarray())
        print xtrain[0]
        #xtrain = self.pca.transform(idfmtx.toarray())

        idfmtx_test = self.tfidf.transform(test_set[0])
        xtest = self.pca.transform(idfmtx_test.toarray())

        self.clf.fit(xtrain, train_set[1])

        pred = self.clf.predict(xtest)
        print(accuracy_score(test_set[1], pred))
        print(classification_report(test_set[1], pred, target_names=class_names))

###########################################################
## SVD aself.pplied +  partitioned data set -- see 'test_ratio' variable
    def svc_svd(self):
        print "running SVM with SVD"
        corpus, feature_label = self.pp.sort_corpus(class1)
        corpus, feature_label = self.pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)

        train_set, test_set = self.pp.data_divider(corpus, feature_label)
        self.pp.cnt_avg_mail_length(train_set[0])
        print len(train_set[0])
        print len(test_set[0])
        #   define idf matrix
        idfmtx = self.tfidf.fit_transform(train_set[0])
        print idfmtx.shape
        #   train
        xtrain = self.svd.fit_transform(idfmtx.toarray())
        print xtrain.shape
        print xtrain[0]
        #xtrain = self.pca.transform(idfmtx.toarray())

        idfmtx_test = self.tfidf.transform(test_set[0])
        xtest = self.svd.transform(idfmtx_test.toarray())

        self.clf.fit(xtrain, train_set[1])

        pred = self.clf.predict(xtest)
        print(accuracy_score(test_set[1], pred))
        print(classification_report(test_set[1], pred, target_names=class_names))


###########################################################
## SVD + plot
    def svc_svd_plot_comp(self):
        print "running SVM with SVD and plot the result"
        h = .02  # step size in the mesh
        # title for the plots
        titles = ['SVC with linear kernel',
                  'LinearSVC (linear kernel)',
                  'SVC with RBF kernel',
                  'SVC with polynomial (degree 3) kernel']

        corpus, feature_label = self.pp.sort_corpus(class1)
        corpus, feature_label = self.pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)
        train_set, test_set = self.pp.data_divider(corpus, feature_label)

        #   define idf matrix
        idfmtx = self.tfidf.fit_transform(train_set[0])
        idfmtx_test = self.tfidf.transform(test_set[0])

        #   train
        xtrain = self.svd.fit_transform(idfmtx)
        xtest = self.svd.transform(idfmtx_test)


        self.rbf_svc.fit(xtrain, train_set[1])
        self.poly_svc.fit(xtrain, train_set[1])
        self.lin_svc.fit(xtrain, train_set[1])
        self.clf.fit(xtrain, train_set[1])

        # create a mesh to plot in
        x_min, x_max = xtest[:, 0].min() - 1, xtest[:, 0].max() + 1
        y_min, y_max = xtest[:, 1].min() - 1, xtest[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        titles = ['svc', 'lin', 'rbf', 'poly']

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        for i, clf in enumerate((self.clf, self.lin_svc, self.rbf_svc, self.poly_svc)):
            print "accuracy %s" % titles[i-1]

            pred = clf.predict(xtest)
            print(accuracy_score(test_set[1], pred))
            print(classification_report(test_set[1], pred, target_names=class_names))

            plt.subplot(2, 2, i + 1)
            plt.subplots_adjust(wspace=0.4, hspace=0.4)

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

            # Plot also the training points
            plt.scatter(xtest[:, 0], xtest[:, 1], c=test_set[1], cmap=plt.cm.coolwarm)
            plt.xlabel('Sepal length')
            plt.ylabel('Sepal width')
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.xticks(())
            plt.yticks(())
            plt.title(titles[i])

        plt.show()

###########################################################
## PCA + pipeline
    def svc_pca_pipeline(self):
        print "running SVM with PCA"
        pipeline = Pipeline([
            ('tfidf', self.tfidf),
            ('to_dense', self.to_dense),
            ('pca', self.pca),
            ('clf', self.clf),
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
## SVD + pipeline
    def svc_svd_pipeline(self):
        print "running SVM with SVD"
        pipeline = Pipeline([
            ('tfidf', self.tfidf),
            ('svd', self.svd),
            ('clf', self.clf),
        ])

        corpus, feature_label = self.pp.sort_corpus(class1)
        corpus, feature_label = self.pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)

        results = []
        for c in range(itera):

            train_set, test_set = self.pp.data_divider(corpus, feature_label)
            self.pp.cnt_avg_mail_length(train_set[0])
            #   train
            pipeline.fit(train_set[0], train_set[1])

            pred = pipeline.predict(test_set[0])
            results.append(accuracy_score(test_set[1], pred))
            print "iteration %s" % len(results)

        print sum(i for i in results)/itera



###########################################################
## SVC(linear=True) using pipeline
    def svc_pipelined(self):
        print "running SVC with pipeline"
        pipeline = Pipeline([
            ('tfidf', self.tfidf),
            ('clf', self.clf),
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
## SVC(linear=polynominal) using pipeline
    def svc_model_comp(self):
        print "running SVM models for comparison"
        corpus, feature_label = self.pp.sort_corpus(class1)
        corpus, feature_label = self.pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)

        titles = ['svc', 'lin', 'rbf', 'poly']

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        for i, clf in enumerate((self.clf, self.lin_svc, self.rbf_svc, self.poly_svc)):
            results = []
            for c in range(itera):
                train_set, test_set = self.pp.data_divider(corpus, feature_label)
                #   define idf matrix
                idfmtx = self.tfidf.fit_transform(train_set[0])
                idfmtx_test = self.tfidf.transform(test_set[0])

                #   train
                clf.fit(idfmtx, train_set[1])
                pred = clf.predict(idfmtx_test)
                results.append(accuracy_score(test_set[1], pred))

                print "iteration %s" % len(results)

            print "model: %s" % titles[i-1]
            print sum(i for i in results)/itera
        #print(accuracy_score(test_set[1], pred))
        #print(classification_report(test_set[1], pred, target_names=class_names))


###########################################################
## SVC(linear=True) + partitioned data set -- see 'test_ratio' variable

    def svc(self):
        print "running SVC"
        corpus, feature_label = self.pp.sort_corpus(class1)
        corpus, feature_label = self.pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)

        results = []
        for c in range(itera):
            train_set, test_set = self.pp.data_divider(corpus, feature_label)

            #   define idf matrix
            idfmtx = self.cnt_vec.fit_transform(train_set[0])

            #   train
            self.clf.fit(idfmtx, train_set[1])

            #   transform normal documents 
            idfmtx_test = self.cnt_vec.transform(test_set[0])
            pred = self.clf.predict(idfmtx_test)
            results.append(accuracy_score(test_set[1], pred))
            print cross_val_score(self.clf, idfmtx_test, test_set[1], cv=5)
            print "iteration %s" % len(results)

        print sum(i for i in results)/itera
        #cnf_mtx = confusion_matrix(test_set[1],pred)
        #plt.figure()
        #self.pp.plot_confusion_matrix(cnf_mtx,classes=class_names, title="confusion matrix, without preprocessing")
        #plt.show()
        #print(accuracy_score(test_set[1], pred))
        #print(classification_report(test_set[1], pred, target_names=class_names))

###########################################################
## SVC(linear=True) + partitioned data set -- see 'test_ratio' variable
    def svc_cross_val(self, weight_model=None):
        corpus, feature_label = self.pp.sort_corpus(class1)
        corpus, feature_label = self.pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)
        train_set, test_set = self.pp.data_divider(corpus, feature_label)

        if weight_model == 'tfidf':
            print "vector model"
            model = self.tfidf
        elif weight_model == 'count':
            print "count model"
            model = self.cnt_vec
        else:
            print "hash model"
            model = self.hv
            
        #   define idf matrix
        idfmtx = model.fit_transform(train_set[0])
        idfmtx_test = model.transform(test_set[0])
        print idfmtx[0].shape
        print idfmtx[1].shape

        idfmtx = self.sbf.fit_transform(idfmtx, train_set[1])
        idfmtx_test = self.sbf.transform(idfmtx_test)
        #   train
        self.clf.fit(idfmtx, train_set[1])


        #print(clf.predict(idfmtx_test))
        #print(accuracy_score(test_set[1], clfGNB.predict(idfmtx_test.toarray())))
        scores = cross_val_score(self.clf, idfmtx, train_set[1], cv=100)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        pred = self.clf.predict(idfmtx_test)
        print(accuracy_score(test_set[1], pred))
        print(classification_report(test_set[1], pred, target_names=class_names))

        cnf_mtx = confusion_matrix(test_set[1], pred)
        plt.figure()
        self.pp.plot_confusion_matrix(cnf_mtx, classes=class_names, title="confusion matrix, SVM classifier")
        plt.show()


###########################################################
## SVC(linear=True) + t-SNE dimensionality reduction technique

    def svc_tSNE(self):
        print "running SVC with t-SNE"
        #np.set_printoptions(suself.ppress=True)

        corpus, feature_label = self.pp.sort_corpus(class1)
        corpus, feature_label = self.pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)

        train_set, test_set = self.pp.data_dividor(corpus, feature_label)

        #   define idf matrix
        idfmtx = self.tfidf.fit_transform(train_set[0])
        print idfmtx.shape
        modelv = tsne.fit_transform(idfmtx.toarray())
        print modelv.shape

        #   train
        self.clf.fit(modelv, train_set[1])

        #   transform normal documents 
        #idfmtx_test = self.tfidf.transform(test_set[0])
        idfmtx_test = self.tfidf.transform(test_set[0])
        modelv_test = self.tsne.fit_transform(idfmtx_test.toarray())

        #print(self.clf.predict(idfmtx_test))
        #cnf_matrix = confusion_matrix(test_set[1], self.clf.predict(idfmtx_test.toarray()))
        pred = self.clf.predict(modelv_test)
        print(self.clf.score(modelv_test, test_set[1]))
        print(classification_report(test_set[1], pred, target_names=class_names))


if __name__ == "__main__":

    clf = SVM(binary=False, stop_words=False, stemmer=False, entity_remover=False, lowercase=False)
    print "no preprocessing, tfidf"
    print features
    #clf = SVM()
    #clf.lda_pipeline()
    clf.svc_cross_val("tfidf")


###########################################################
## test
"""
corpus, feature_label = self.pp.sort_corpus(class1)
corpus, feature_label = self.pp.sort_corpus_custom(class2, corpus=corpus, feature_label=feature_label)

train_set, test_set = self.pp.data_dividor(corpus, feature_label)

#   define idf matrix
idfmtx = self.tfidf.fit_transform(train_set[0])
#   train
self.clf.fit(idfmtx, train_set[1])

feature_label = []

#   transform normal documents 
idfmtx_test = self.tfidf.transform(test_set[0])

class_names = ['ham', 'spam']
pred = self.clf.predict(idfmtx_test)

print test_set[1]
print(accuracy_score(test_set[1], pred))
print(classification_report(test_set[1], pred, target_names=class_names))
corpus_test = []
corpus = []
feature_label = []"""
