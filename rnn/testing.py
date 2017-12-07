import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import pickle
import sys
import heapq
import seaborn as sns
from pylab import rcParams

from preprocessing import load_training_file

authors = ['eap', 'mws', 'hpl']

if len(sys.argv) < 2:
    author = ''
else:
    author = sys.argv[1].lower()

if author not in authors:
    print('Invalid author, please try one of', ', '.join(authors))
    sys.exit()

print('Testing author:', author)

SEQUENCE_LENGTH, chars, char_indices, indices_char, text, X, y = load_training_file(author + '_train.txt')

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 12, 5

# we load our model back up to make sure it works
model = load_model(author + '_model.h5')
history = pickle.load(open(author + '_history.p', 'rb'))

# we can now plot how our accuracy and loss change over training epochs
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# this function prepares input text
def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(text):
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

# first quote is by Mary Shelley, second is by HP Lovecraft, the last is by Edgar Allen Poe
quotes = [
    "Life and death appeared to me ideal bounds, which I should first break through, and pour a torrent of light into our dark world.",
    "The oldest and strongest emotion of mankind is fear, and the oldest and strongest kind of fear is fear of the unknown.",
    "Deep into that darkness peering, long I stood there, wondering, fearing, doubting, dreaming dreams no mortal ever dared to dream before."
]

# model predictions
for q in quotes:
    seq = q[:40]
    print(q)
    print(seq)
    print(predict_completions(seq, 5))
    print()

# this function selects an index from a probability array
def sample(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)

start_index = np.random.randint(0, len(text) - SEQUENCE_LENGTH - 1)

for diversity in [0.2, 0.5, 1.0, 1.2]:
    print()
    print('----- diversity:', diversity, '------')

    generated = ''
    sentence = text[start_index: start_index + SEQUENCE_LENGTH]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '" -----')
    sys.stdout.write(generated)

    for i in range(400):
        x_pred = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
