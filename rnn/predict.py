import numpy as np
import pandas as pd
from keras.models import load_model
import sys
import heapq

from preprocessing import load_training_file

authors = ['eap', 'mws', 'hpl']

test = pd.read_csv('test.csv')

x = test['text'].values
scores = np.zeros((x.size, 3), dtype=np.bool)

for index, author in enumerate(authors):
    SEQUENCE_LENGTH, chars, char_indices, indices_char, text, X, y = load_training_file(author + '_train.txt')

    # we load our model back up to make sure it works
    model = load_model(author + '_model.h5')

    # this function prepares input text
    def prepare_input(text):
        x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
        for t, char in enumerate(text):
            if char in char_indices:
                x[0, t, char_indices[char]] = 1.0
        return x

    # this function allows us to ask our model what the next n most probable characters are
    def sample_guess(predictions, top_n=3):
        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions)
        exp_predictions = np.exp(predictions)
        predictions = exp_predictions / np.sum(exp_predictions)
        return heapq.nlargest(top_n, range(len(predictions)), predictions.take)

    # this function predicts the next character until a space is predicted
    def predict_completion(text):
        original_text = text
        completion = ''
        while True:
            x = prepare_input(text)
            predictions = model.predict(x, verbose=0)[0]
            next_index = sample_guess(predictions, top_n=1)[0]
            next_char = indices_char[next_index]
            text = text[1:] + next_char
            completion += next_char
            if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
                return completion

    # this function ties everything together and allows us to predict multiple completions:
    def predict_completions(text, n=3):
        x = prepare_input(text)
        predictions = model.predict(x, verbose=0)[0]
        next_indices = sample_guess(predictions, n)
        return [indices_char[index] + predict_completion(text[1:] + indices_char[index]) for index in next_indices]

    # this function selects an index from a probability array
    def sample(predictions, temperature=1.0):
        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions) / temperature
        exp_predictions = np.exp(predictions)
        predictions = exp_predictions / np.sum(exp_predictions)
        probabilities = np.random.multinomial(1, predictions, 1)
        return np.argmax(probabilities)

    def score_completions(text):
        if len(text) <= SEQUENCE_LENGTH:
            return 0

        score = 0
        for i in range(0, len(text) - SEQUENCE_LENGTH - 1):
            precut = text[i:]
            x = prepare_input(precut[:40])
            predictions = model.predict(x, verbose=0)[0]
            next_index = sample_guess(predictions, top_n=1)[0]
            next_char = indices_char[next_index]
            if next_char == text[i + SEQUENCE_LENGTH]:
                score += 1

        return score

    for i, text in enumerate(x):
        scores[i][index] = score_completions(text)

np.save('char_based_scores.npy', scores)
