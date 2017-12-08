import numpy as np
import pandas as pd

from keras.models import load_model
from keras.utils import plot_model

model = load_model('linear_model.h5')

plot_model(model, to_file='linear_model.png')
