import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.optimizers import SGD

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
targets = ['EAP', 'HPL', 'MWS']

probs_train = []
probs = []
sources = ['char_based', 'bayes', 'linear']

for source in sources:
    probs_train.append(np.load(source + '_scores_training.npy'))
    probs.append(np.load(source + '_scores.npy'))

x_train = np.hstack(*probs_train)
x = np.hstack(*probs)

y_train_text = train['author'].values
y_train = np.zeros((train.shape[0], train.shape[1]), dtype=np.bool)
for index, author in enumerate(y_train_text):
    y_train[index][targets.index(author)] = 1

y_text = test['author'].values
y = np.zeros((test.shape[0], test.shape[1]), dtype=np.bool)
for index, author in enumerate(y_text):
    y[index][targets.index(author)] = 1

model = Sequential()

model.add(Dense(32, input_shape=(9,), use_bias=True))
model.add(Dropout(0.2))
model.add(Activation('softmax'))

model.add(Dense(3, activation='softmax'))

sgd = SGD()
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x, y, validation_split=0.1, verbose=2, epochs=100)

predicted_proba = model.predict(x)

output = pd.DataFrame(data={
    'id': test['id'],
    'EAP': predicted_proba[:,0],
    'HPL': predicted_proba[:,1],
    'MWS': predicted_proba[:,2]
})

output = output[['id', 'EAP', 'HPL', 'MWS']]

output.to_csv('submission.csv', index=False)
