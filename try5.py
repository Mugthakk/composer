
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import LSTM, Dropout, Dense, BatchNormalization, Activation, Input, TimeDistributed
from keras.regularizers import l1, l2
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU
from keras.optimizers import RMSprop, Adam, Adadelta, Adagrad
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from time import time
from matplotlib import pyplot as plt
import random as rn
import tensorflow as tf
from assignment.helpers import datapreparation as prep

# Seed session
np.random.seed(123456)
rn.seed(123456)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=0,
                              inter_op_parallelism_threads=0)
from keras import backend as K

tf.set_random_seed(123456)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
##

fs1_dirpath = "./assignment/datasets/training/piano_roll_fs1"
fs5_dirpath = "./assignment/datasets/training/piano_roll_fs5"

# Load initial data
datasets = prep.load_all_dataset(fs5_dirpath)
dataset_names = prep.load_all_dataset_names(fs5_dirpath)
unique_names = set()
for name in dataset_names:  # Make sure the same names get the same encoding each run
    unique_names.add(name)
unique_names = list(unique_names)
name_to_int = dict([(unique_names[i], i) for i in range(len(unique_names))])
int_to_name = dict([(i, unique_names[i]) for i in range(len(unique_names))])
dataset_names = to_categorical([name_to_int[name] for name in dataset_names])  # one-hot encode the composers
datasets = [dataset[:, 1:] for dataset in datasets]  # Remove the headers

# Setting initial parameters
dataset_id_names = dict(zip(np.arange(len(dataset_names)), dataset_names))
longest_song = max(datasets[i].shape[1] for i in range(len(datasets)))
longest_song = 4173  # 4155 // 43 != good, 4173 // 43 = 97
print(longest_song)
sequence_length = 43
length = longest_song // sequence_length + 1
num_keys = len(datasets[0])
parts_per_song = int(longest_song / sequence_length)
composer_encoding_len = len(dataset_names[0])  # 4 composers


# Makes several datasets from this first one with differing intervals between to capture the "gaps" between two sequences
# Add each subsequence of each song with differing offsets ([0:10], [1:11], [2:12], ...) to retain information.
# Unable to implement stateful, so try to retain as much information between subsequences as possible.
# Also a way of dataset augmentation (regularization) by increasing the size of the dataset

def pepper_for_generator(dataset_names, datasets, num_keys, maxlen):
    zs = []
    songs = np.array([dataset.T for dataset in datasets])
    targets = np.array([np.append(song[:-1], np.array([np.ones(num_keys)]), 0) for song in songs])
    songs = pad_sequences(songs, maxlen=maxlen, padding="post", value=np.array([np.zeros(128)]))
    targets = pad_sequences(targets, maxlen=maxlen, padding="post", value=np.array([np.zeros(128)]))
    song_composers = np.array([dataset_names[i] for i in range(len(datasets))])
    return songs, targets, song_composers


def data_generator(xs, ys, composers, b_size, sequence_length, number_steps):
    prev_index = 0

    while True:

        xs_batch, composer_batch, ys_batch = [], [], []

        for i in range(b_size):
            xs_batch.append(xs[i][prev_index * sequence_length:(prev_index + 1) * sequence_length])
            ys_batch.append(ys[i][prev_index * sequence_length:(prev_index + 1) * sequence_length])
            composer_batch.append(composer_batch[i])

        prev_index += 1

        yield [np.array(xs_batch), np.array(composer_batch)], np.array(ys_batch)


#datasets = pad_sequences(datasets, maxlen=longest_song, padding='post', value=np.zeros(num_keys))
train_xs, train_ys, train_composers = pepper_for_generator(dataset_names, datasets, num_keys, longest_song)

generator = data_generator(train_xs, train_ys, train_composers, len(train_xs), sequence_length,
                           longest_song // sequence_length)