
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Masking, Dropout, Dense, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from time import time

from assignment.helpers import datapreparation as prep


fs5_dirpath = "./assignment/datasets/training/piano_roll_fs5"

datasets = prep.load_all_dataset(fs5_dirpath)
dataset_names = prep.load_all_dataset_names(fs5_dirpath)

datasets = [dataset[:, 1:] for dataset in datasets] # Remove the headers

dataset_id_names = dict(zip(np.arange(len(dataset_names)), dataset_names))
longest_song = max(datasets[i].shape[1] for i in range(len(datasets)))
sequence_length = 50
length = longest_song//sequence_length + 1  # make each song multiple of 1000
num_keys = len(datasets[0])
parts_per_song = int(longest_song/sequence_length)


def transpose_and_label(datasets, num_keys, length):
    datasets_transposed = np.array([dataset.T for dataset in datasets])
    datasets_transposed_padded = pad_sequences(datasets_transposed, maxlen=sequence_length*length, padding="post", value=[1.0 for _ in range(num_keys)])
    datasets_transposed_padded_labels = np.array([np.append(dataset[1:, :], np.array([np.ones(num_keys)]), 0) for dataset in datasets_transposed_padded])
    return datasets_transposed_padded, datasets_transposed_padded_labels


# lists of the notes at each time padded for each song (43x4155x128) and one step ahead for ys
xs, ys = transpose_and_label(datasets, num_keys, length)

xs_split = np.array([song[i:i+sequence_length, :] for song in xs for i in range(parts_per_song)])
ys_split = np.array([song[i:i+sequence_length, :] for song in ys for i in range(parts_per_song)])

model = Sequential()


model.add(Masking(mask_value=2.0,
                  input_shape=(sequence_length, num_keys)
                  )
          )
model.add(LSTM(sequence_length,
               activation='relu',
               return_sequences=True,
              # input_shape=(sequence_length, num_keys)
               )
          )
#model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(128,
                activation="softmax"
                )
          )

tensorboard = TensorBoard(log_dir="./logs/{}".format(time()))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print(model.summary())

model.fit(xs_split, ys_split,
          epochs=10,
          validation_split=0.2,
          shuffle=True,
          callbacks=[tensorboard],
          )

model.save("./models/latest.h5f")
# When LSTM is done can use trainable=False to freeze it while training the other one

a = model.predict(xs_split, verbose=True)
print(a)
print(a.max)
prep.visualize_piano_roll(xs_split[0].T)
prep.visualize_piano_roll(a[0].T)
print("done")
# need to ignore the first timestep because guy messed up row numbers lol