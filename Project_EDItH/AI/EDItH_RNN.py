import time
from collections import namedtuple
import numpy as np
import tensorflow as tf
import pickle

data = pickle.load(open('kako_dialog.pickle', 'rb'))
text = ''.join(data)