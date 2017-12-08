import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.optimizers import SGD

train = pd.read_csv('train.csv')
targets = ['EAP', 'HPL', 'MWS']

x_text = train['text'].values
y_text = train['author'].values
y = np.zeros((19579, 3), dtype=np.bool)
for index, author in enumerate(y_text):
    y[index][targets.index(author)] = 1

vocab_size = 20000

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x_text)
x = tokenizer.texts_to_matrix(x_text, mode='tfidf')

model = Sequential()

model.add(Dense(32, input_shape=(vocab_size,), use_bias=True))
model.add(Dropout(0.2))
model.add(Activation('softmax'))

model.add(Dense(3, activation='softmax'))

sgd = SGD()
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x, y, validation_split=0.1, verbose=2, epochs=100)

model.save('linear_model.h5')
