import sys
import numpy as np
import tensorflow as tf

np.random.seed(29)
tf.set_random_seed(29)

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import pickle

from preprocessing import load_training_file

def main():
    authors = ['eap', 'mws', 'hpl']

    if len(sys.argv) < 2:
        author = ''
    else:
        author = sys.argv[1].lower()

    print('Training on author:', author)

    SEQUENCE_LENGTH, chars, char_indices, indices_char, text, X, y = load_training_file(author + '_train.txt')

    # -MODEL PARAMETERS-
    # single LSTM layers with 128 neurons which accepts input of shape (40, 61)
    # a fully connected layer (for our output) is added after that which has 61 neurons
    # and uses softmax as its activation function
    model = Sequential()
    model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    # our model is trained for 5 epochs using the RMSProp optimizer
    # and uses 10% of the data for testing/validation
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(X, y, validation_split=0.10, batch_size=128, epochs=30, shuffle=True).history

    # it took a lot of time to train our model, so we saved our progress on a HDF5 file
    model.save(author + '_model.h5')
    pickle.dump(history, open(author + '_history.p', "wb"))

if __name__ == "__main__":
    main()
