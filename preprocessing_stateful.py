from keras.models import Model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import LSTM, Dropout, Dense, BatchNormalization, Activation, Input, TimeDistributed
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU
from keras.optimizers import RMSprop, Adam, Adadelta, Adagrad
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from time import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import binarize

from assignment.helpers import datapreparation as prep

# Note on specifying the initial state of RNNs
# Seems as though you can reset state (probably stateful=True), and then pass an array of initial states that can be used
# in the RNN - so initialize based on composer :D
# https://keras.io/layers/recurrent/

fs5_dirpath = "./assignment/datasets/training/piano_roll_fs5"
fs1_dirpath = "./assignment/datasets/training/piano_roll_fs1"

datasets = prep.load_all_dataset(fs1_dirpath)
dataset_names = prep.load_all_dataset_names(fs1_dirpath)

datasets = [dataset[:, 1:] for dataset in datasets] # Remove the headers

dataset_id_names = dict(zip(np.arange(len(dataset_names)), dataset_names))
longest_song = max(datasets[i].shape[1] for i in range(len(datasets)))
sequence_length = 10 #40
#b_size = 71 # 4047=3*19*71, 0.2 split ==> 3237 = 3*13*83
number_of_batches = longest_song//sequence_length + 1
num_keys = len(datasets[0])
pad_length = number_of_batches*sequence_length

def preprocess_for_stateful_with_padding(dataset, num_songs, sequence_length, num_batches, num_keys, pad_length):
    big_ass_ndarray = [[[] for a in range(num_songs)] for b in range(num_batches)]
    songs_padded = pad_sequences(dataset, maxlen=pad_length, padding="post", value=np.array([-1.0 for _ in range(num_keys)]))
    for i in range(num_batches):
        for j in range(num_songs):
            big_ass_ndarray[i][j] = songs_padded[j, i*sequence_length:(i+1)*sequence_length]
    return np.array(big_ass_ndarray)




datasets = np.array([dataset.T for dataset in datasets])
xs = preprocess_for_stateful_with_padding(datasets, len(datasets), sequence_length, number_of_batches, num_keys, pad_length)
datasets_labels = np.array([np.append(dataset[1:,:], np.array([np.ones(num_keys)]), axis=0) for dataset in datasets])
ys = preprocess_for_stateful_with_padding(datasets, len(datasets), sequence_length, number_of_batches, num_keys, pad_length)
print("done")