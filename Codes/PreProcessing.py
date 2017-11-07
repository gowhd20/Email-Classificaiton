import math
import nltk
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import random

from nltk.stem.porter import *

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook
from stop_words import get_stop_words
from sklearn.base import TransformerMixin
#from sklearn.model_selection import train_test_split  # alternative for dividing data into training and testing

test_ratio = .5
ham_count = 1000#12952#10540#12952
spam_count = 1000#8619#8138#8619

class PreProcessing(object):
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.en_stop = get_stop_words('en')
        super(PreProcessing, self).__init__()

    def plot_confusion_matrix(self, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


    def entity_stem_tokner(self, text, stemmer=True, entity_remover=True):
        tokens = nltk.word_tokenize(text)

        if entity_remover:
            tagged = nltk.pos_tag(tokens)
            for item in tagged:
                if item[1] == 'NNP' or item[1] == 'NNPS':
                    tokens.remove(item[0])
        if stemmer:
            stemmed = []
            for item in tokens:
                stemmed.append(self.stemmer.stem(item))
            tokens=stemmed
        return tokens


    def tokenizer(self, sentences):
        return list(nltk.word_tokenize(unicode(m, 'ISO-8859-1')) for m in sentences)


    def remove_stopwords(self, tksentences):
        emails = []
        for e in tksentences:
            emails.append(list(w for w in e if w not in self.en_stop))
        return emails


    def data_divider(self, full_list, full_label):
        full_list, full_label = self.data_shuffler(full_list, full_label)
        size = int(len(full_list)* test_ratio)
        return [full_list[size:], full_label[size:]], [full_list[:size], full_label[:size]] #   [train data, lables], [test data, labels]


    def data_shuffler(self, dlist, label):
        c = list(zip(dlist, label))
        random.shuffle(c)
        dlist, label = zip(*c)
        return list(dlist), list(label)


    def sort_corpus(self, etype, count=None, corpus=None, feature_label=None):
        if corpus == None:
            corpus = []
        if feature_label == None:
            feature_label = []

        if count == None:
            if etype == 'spam':
                count = spam_count
            else:
                count = ham_count

        for e in range(count):
            with os.fdopen(os.open("./dataset_combined/data_part_2/"+etype+"/"+str(e).zfill(5)+".txt", os.O_RDWR),'w+') as outfile:
                if etype == 'spam':
                    f = 1
                else:
                    f = 0 
                feature_label.append(f)
                corpus.append(outfile.read())
                outfile.close()
            
        return corpus, feature_label 


    def dialogue_act_features(self, idx, etype):
        with os.fdopen(os.open("./"+etype+"/"+str(idx).zfill(3)+".txt", os.O_RDWR|os.O_CREAT),'w+') as outfile:
            tokens = nltk.word_tokenize(outfile.read())

            features = {}
            for t in tokens:
                features['contains(%s)' % t.lower()] = True
            outfile.close()

            return features

    def file_rename(self, src, dst):
        os.rename(src, dst)


    def cnt_avg_mail_length(self, emails):
        largest = 0
        smallest = 15
        mails_len = self.tokenizer(emails)          # tokenize emails
        #mails_len = self.remove_stopwords(mails_len)    # removes stopwords
        cnt = len(mails_len)
        total = 0
        for c in mails_len:
            clen = len(c)
            total = clen + total
            if clen > largest:
                largest = clen
            elif clen < smallest:
                smallest = clen

        print "largest %s" % largest
        print "smallest %s" % smallest 
        return int(total/cnt)


class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

"""
import glob

import re
regex = re.compile('spm.')
#basePath = "./dataset_combined/data_part_2/lingspam_public/lemm/part10/"
basePath = "./dataset_combined/data_part_1/ham/"
toPath = "./dataset_combined/data_part_2/ham/"

pp = PreProcessing()
cnt = 8619

for e in range(cnt):
    path = "./dataset_combined/data_part_1/spam/"+str(e).zfill(4)+".txt"
    to = "./dataset_combined/data_part_1/spam/"+str(e).zfill(5)+".txt"
    pp.file_rename(path, to)


count=0
for dirpath, dnames, fnames in os.walk(basePath):
    for f in fnames:
        path = basePath+f
        to = basePath+str(count).zfill(5)+".txt"
        count = 1+count
        pp.file_rename(path, to)


count=500
for dirpath, dnames, fnames in os.walk(basePath):
    for f in fnames:
        if len(f) == 8:
            os.remove(basePath+f)
            
"""
