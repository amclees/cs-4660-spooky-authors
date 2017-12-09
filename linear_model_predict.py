import numpy as np
import pandas as pd

from keras.models import load_model
from keras.preprocessing.text import Tokenizer

model = load_model('linear_model.h5')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

x_text = test['text'].values

vocab_size = 20000

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train['text'].values)
x = tokenizer.texts_to_matrix(x_text, mode='tfidf')
x_train = tokenizer.texts_to_matrix(train['text'].values)

predicted_proba = model.predict(x)
predicted_proba_train = model.predict(x_train)

np.save('linear_scores.npy', predicted_proba)
np.save('linear_scores_training.npy', predicted_proba_train)

output = pd.DataFrame(data={
    'id': test['id'],
    'EAP': predicted_proba[:,0],
    'HPL': predicted_proba[:,1],
    'MWS': predicted_proba[:,2]
})

output = output[['id', 'EAP', 'HPL', 'MWS']]

output.to_csv('submission.csv', index=False)
