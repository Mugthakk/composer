from keras.models import Model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import LSTM, Dropout, Dense, BatchNormalization, Activation, Input
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU
from keras.optimizers import RMSprop, Adam, Adadelta, Adagrad
import numpy as np
from time import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import binarize

from assignment.helpers import datapreparation as prep


fs5_dirpath = "./assignment/datasets/training/piano_roll_fs5"

datasets = prep.load_all_dataset(fs5_dirpath)
dataset_names = prep.load_all_dataset_names(fs5_dirpath)

datasets = [dataset[:, 1:] for dataset in datasets] # Remove the headers

dataset_id_names = dict(zip(np.arange(len(dataset_names)), dataset_names))
longest_song = max(datasets[i].shape[1] for i in range(len(datasets)))
sequence_length = 17
length = longest_song//sequence_length + 1
num_keys = len(datasets[0])
parts_per_song = int(longest_song/sequence_length)


def transpose_and_label(datasets, num_keys):
    xs, ys = [], []
    datasets_transposed = np.array([dataset.T for dataset in datasets])
    for song in datasets_transposed:
        for i in range(0, len(song)//sequence_length):
            xs.append(song[i*sequence_length:(i+1)*sequence_length])
            if i == len(song)//sequence_length - 1:
                ys.append(np.append(song[i*sequence_length+1:(i+1)*sequence_length], np.array([np.ones(num_keys)]), 0))
            else:
                ys.append(song[i*sequence_length+1:(i+1)*sequence_length+1])
        print("hello")
    return xs, ys

# Makes several datasets from this first one with differing intervals between to capture the "gaps" between two sequences
def transpose_and_label_more(datasets, num_keys):
    zs = []
    datasets_transposed = np.array([dataset.T for dataset in datasets])
    for song in datasets_transposed:
        for offset in range(0, sequence_length):
            for i in range(0, len(song)//sequence_length):
                x = song[offset+i*sequence_length:offset+(i+1)*sequence_length]
                if i == len(song)//sequence_length - 1: # Add the EOF marker if last seq of song
                    y = np.append(song[offset+i*sequence_length+1:offset+(i+1)*sequence_length], np.array([np.ones(num_keys)]), 0)
                else:
                    y = song[offset+i*sequence_length+1:offset+(i+1)*sequence_length+1]
                zs.append((x,y))
    zs = np.array(zs)
    np.random.shuffle(zs)
    return zs[:, 0], zs[:, 1]

xs, ys = transpose_and_label_more(datasets, num_keys) # TODO shuffle


xs_concat, ys_concat = np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

#print(xs_concat.shape) # 69122 = 2*17*19*107

xs_split = np.array([xs_concat[i*sequence_length:(i+1)*sequence_length] for i in range(len(xs_concat)//sequence_length)])
ys_split = np.array([ys_concat[i*sequence_length:(i+1)*sequence_length] for i in range(len(ys_concat)//sequence_length)])

ys_labels = ys_split[:, 0, :]

inputs = Input(shape=(sequence_length, num_keys))
# Units = units per timestep LSTM block, i.e. output dimensionality (128 here since input and output 128 keys)
lstm = LSTM(num_keys,
               activation='relu',
               return_sequences=True,
               dropout=0.0, #0.25,
               recurrent_dropout=0.0, #0.25,
               kernel_regularizer=None, #l2(0.0001),
               recurrent_regularizer=None, #l2(0.0001),
               bias_regularizer=None,
               activity_regularizer=None, #l2(0.0001),
               )(inputs)
#lstm = LeakyReLU(alpha=.001)(lstm) # Workaround for getting leakyrelu as activation in lstm
lstm = BatchNormalization()(lstm)
#lstm = Dropout(0.25)(lstm)
lstm = Dense(128)(lstm)
outputs = Activation("sigmoid")(lstm) # Sigmoid keeps the probabilities independent of each other, while softmax does not!

model = Model(inputs=inputs, outputs=outputs)

rmsprop = RMSprop(lr=0.001)
adagrad =  Adagrad(lr=0.001)
adam = Adam(lr=0.001)
adadelta = Adadelta(lr=1.0)

# Want to penalize each output node independantly. So we pick a binary loss
# and model the output of the network as a independent bernoulli distributions per label.

model.compile(loss='binary_crossentropy',
              optimizer=adam, # consider changing this one for others
              metrics=['categorical_accuracy'])
print(model.summary())

tensorboard = TensorBoard(log_dir="./logs/{}".format(time()))
early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=3, verbose=0, mode="auto")
model.fit(xs_split, ys_split,
          epochs=2000, # Train harder more for more things was too bad train man :(
          validation_split=0.2,
          batch_size=64,
          shuffle=True,
          callbacks=[tensorboard],
          )

model.save("./models/latest.h5f")
# When LSTM is done can use trainable=False to freeze it while training the other one
