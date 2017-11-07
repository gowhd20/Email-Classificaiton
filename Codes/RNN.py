import numpy
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words

from PreProcessing import PreProcessing

from sklearn.feature_selection import SelectKBest, chi2

class1 = 'spam'
class2 = 'ham'
max_features = 10000
max_review_length = 250
pca_features = 3
#best_length = 50
pp = PreProcessing()
tk = Tokenizer(nb_words=max_features, lower=True)
en_stop = get_stop_words('en')
#tfidf = TfidfVectorizer(min_df=1, stop_words=en_stop, tokenizer=pp.entity_stem_tokner, decode_error='replace')

print('Loading data...')
corpus, feature_label = pp.sort_corpus(class1)
corpus, feature_label = pp.sort_corpus(class2, corpus=corpus, feature_label=feature_label)
train_set, test_set = pp.data_divider(corpus, feature_label)
x_train = train_set[0]
y_train = train_set[1]
x_test = test_set[0]
y_test = test_set[1]

print(len(x_train), 'train')
print(len(y_train), 'test')
#x_train = tfidf.fit_transform(train_set[0])
#x_test = tfidf.transform(test_set[0])

"""
tk.fit_on_texts(x_train)
tk.fit_on_texts(x_test)
x_train = tk.texts_to_sequences(x_train)#, mode='tfidf')
x_test = tk.texts_to_sequences(x_test)#, mode='tfidf')

"""

from sklearn.decomposition import PCA

pca = PCA(n_components=pca_features)

tk.fit_on_texts(x_train)
tk.fit_on_texts(x_test)
x_train = tk.texts_to_matrix(x_train, mode='tfidf')
x_test = tk.texts_to_matrix(x_test, mode='tfidf')

new_x_train = pca.fit_transform(x_train)

print x_train.shape
print new_x_train.shape
print x_test.shape
print new_x_train[0]
new_x_train = numpy.reshape(new_x_train[0], 1, new_x_train[1])
x_test = numpy.reshape(x_test[0], 1, x_test[1])

#x_train = SelectKBest(chi2, k=best_length).fit_transform(x_train, y_train) # pick lowest (unique) k number of vectors
print new_x_train.shape
print x_test.shape
"""
# truncate and pad input sequences
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

print(x_train.shape, 'train shape')
print(x_test.shape, 'test shape')
"""

# create the model
embedding_vecor_length = 32 	# length of vector for describing a word
batch_size = 64
model = Sequential()

#model.add(Embedding(max_features, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100, input_dim=3, input_length=1000))
#model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit([[[123]]], y_train, validation_data=(x_test, y_test), nb_epoch=5, batch_size=batch_size)

scores = model.evaluate(x_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

