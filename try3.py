from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dropout, Dense, BatchNormalization, Lambda, Activation
from keras.backend import max as kmax
from keras.backend import greater_equal
from keras.backend import cast, floatx
from keras.backend import round as kround
from keras.regularizers import l1, l2
from keras.activations import softmax
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
sequence_length = 19
length = longest_song//sequence_length + 1
num_keys = len(datasets[0])
parts_per_song = int(longest_song/sequence_length)


def transpose_and_label(datasets, num_keys):
    datasets_transposed = np.array([dataset.T for dataset in datasets])
    datasets_transposed_padded_labels = np.array([np.append(dataset[1:, :], np.array([np.ones(num_keys)]), 0) for dataset in datasets_transposed])
    return datasets_transposed, datasets_transposed_padded_labels


xs, ys = transpose_and_label(datasets, num_keys)

xs_concat, ys_concat = np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

#print(xs_concat.shape) # 69122 = 2*17*19*107

xs_split = np.array([xs_concat[i*sequence_length:(i+1)*sequence_length] for i in range(len(xs_concat)//sequence_length)])
ys_split = np.array([ys_concat[i*sequence_length:(i+1)*sequence_length] for i in range(len(ys_concat)//sequence_length)])

model = Sequential()

model.add(LSTM(128,
               activation='relu',
               return_sequences=True,
               dropout=0.5,
               input_shape=(sequence_length, num_keys),
               kernel_regularizer=l2(0.001),
               recurrent_regularizer=l2(0.001),
               bias_regularizer=None,
               activity_regularizer=l2(0.001),
               )
          )
model.add(Dropout(0.5))

#model.add(BatchNormalization())
#model.add(Dense(units=128, activation=Lambda(lambda z: kround(softmax(z)))))

model.add(BatchNormalization())
model.add(Activation("softmax"))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print(model.summary())

tensorboard = TensorBoard(log_dir="./logs/{}".format(time()))
model.fit(xs_split, ys_split,
          epochs=100,
          validation_split=0.2,
          shuffle=True,
          callbacks=[tensorboard],
          )

model.save("./models/latest.h5f")
# When LSTM is done can use trainable=False to freeze it while training the other one


def make_song_from_predict(model, initial_data, limit):
    song = []
    keep_producing = True
    prev_data = initial_data
    while keep_producing and len(song) < limit:
        predictions = model.predict(np.array([prev_data]))[0]
        predictions = binarize(predictions, threshold=0.5*np.max(predictions[-1]))
        last_output = predictions[-1]
        keep_producing = np.sum(last_output) != len(last_output)
        song.append(last_output)
        prev_data = predictions
    return np.array(song)


song = make_song_from_predict(model, xs_split[4], 1000)
prep.visualize_piano_roll(song.T)
prep.visualize_piano_roll(xs_split[4].T)
