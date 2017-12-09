# Spooky Author Identification

## Important Node

The character-based RNN code was committed to the repository by amclees, but it was authored by PolyCarboKnight.

Some code comes from other sources; this code is accompanied by licenses where required (in a separate file or as a header).

Specifically, the following elements were developed as follows:
* The Naive Bayes code was loosely based on the Naive Bayes pipeline code from the CS 4660 lecture on Naive Bayes
* The text generation code for the character-based LSTM was based on [an example provided by the official Keras GitHub](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py).
* The word-based LSTM was based on a separate tutorial, license information may be found inside the `rnn_word_vec` folder
* The word2vec standalone model and visualization is a slightly modified version of sample code from [this TensorFlow tutorial](https://www.tensorflow.org/tutorials/word2vec)

## Instructions

### Naive Bayes

* Open `bayes.ipynb` and run all cells. Predictions will be exported to `submission.csv`

### Linear Model in Keras

* Run `linear_model.py` to train; the trained model will be stored in `linear_model.h5`.
* Run `linear_model_predict.py` to generate a new `submission.csv`
* Run `linear_model_visualization.py` to generate a visualization of the model in `linear_model.png`

### Character-based LSTM

* Run `rnn/training.py <author initials>` to train the model
* Run `rnn/testing.py <author initials>` to obtain graphs of the model loss and accuracy during training and sample text
* Run `rnn/predict.py` to predict characters in each sentence of the provided CSV

### Word-based LSTM

* Open and run `rnn_word_vec/Train_rnn.ipynb` to train the model
* Open and run `rnn_word_vec/Generate_text.ipynb` to generate text

## Team Deutneuronomy

* amclees
* DesignsByBenji
* ellipsclamation
* PolyCarboKnight

