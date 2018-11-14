# %matplotlib inline


from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dropout, Dense, BatchNormalization, Lambda
from keras.backend import max as kmax
from keras.backend import greater_equal
from keras.backend import cast, floatx, round
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

print(xs_concat.shape) # 69122 = 2*17*19*107

xs_split = np.array([xs_concat[i*sequence_length:(i+1)*sequence_length] for i in range(len(xs_concat)//sequence_length)])
ys_split = np.array([ys_concat[i*sequence_length:(i+1)*sequence_length] for i in range(len(ys_concat)//sequence_length)])

model = Sequential()

model.add(LSTM(sequence_length,
               activation='relu',
               return_sequences=True,
               input_shape=(sequence_length, num_keys)
               )
          )
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(128,
                activation="softmax"
                )
          )

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print(model.summary())

tensorboard = TensorBoard(log_dir="./logs/{}".format(time()))
model.fit(xs_split, ys_split,
          epochs=25,
          validation_split=0.2,
          shuffle=True,
          callbacks=[tensorboard],
          )

model.save("./models/latest.h5f")
# When LSTM is done can use trainable=False to freeze it while training the other one
a = model.predict(xs_split, verbose=True)
b = np.max(a[1][-1])
a2 = binarize(a[1], threshold=0.5*np.max(a[1][-1]))
plt.plot(a2[-1])
plt.show()
prep.visualize_piano_roll(a[1].T)
prep.visualize_piano_roll(a2.T)
start_i, stop_i = 500, 510
first_song = np.concatenate(np.array([binarize(c, threshold=0.5*np.max(c[-1])) for c in a[start_i:stop_i,:]]), axis=0)
prep.visualize_piano_roll(first_song.T)
prep.embed_play_v1(first_song.T)
prep.visualize_piano_roll(np.concatenate(xs_split[start_i:stop_i,:], axis=0).T)
prep.embed_play_v1(np.concatenate(xs_split[start_i:stop_i,:], axis=0).T)