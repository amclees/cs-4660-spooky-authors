import numpy as np

def load_training_file(filename):
    text = open(filename, encoding='utf-8').read()

    # find all unique chars in corpus and create char to index and index to char maps
    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # let's cut the corpus into chucks of 40 characters, spacing the sequences by 3 characters
    # we will store the next character (the one we need to predict) for every sequence
    SEQUENCE_LENGTH = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - SEQUENCE_LENGTH, step):
        sentences.append(text[i: i + SEQUENCE_LENGTH])
        next_chars.append(text[i + SEQUENCE_LENGTH])

    # we will use the previously generated sequences and characters that need to be
    # predicted to create one-hot encoded vectors using the char_indices map
    X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    return SEQUENCE_LENGTH, chars, char_indices, indices_char, text, X, y

