from keras.models import Model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import LSTM, Dropout, Dense, BatchNormalization, Activation, Input
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
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

inputs = Input(shape=(sequence_length, num_keys))

# Units = units per timestep LSTM block, i.e. output dimensionality (128 here since input and output 128 keys)

lstm = LSTM(num_keys,
               activation='linear',
               return_sequences=True,
               dropout=0.25,
               recurrent_dropout=0.25,
               kernel_regularizer=None,
               recurrent_regularizer=l2(0.001),
               bias_regularizer=None,
               activity_regularizer=l2(0.001),
               )(inputs)
lstm = LeakyReLU(alpha=.001)(lstm) # Workaround for getting leakyrelu as activation in lstm
lstm = Dropout(0.25)(lstm)
lstm = BatchNormalization()(lstm)
lstm = Dense(128)(lstm)
outputs = Activation("softmax")(lstm)

model = Model(inputs=inputs, outputs=outputs)

rmsprop = RMSprop(lr=0.001)

model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])
print(model.summary())

tensorboard = TensorBoard(log_dir="./logs/{}".format(time()))
early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=2, verbose=0, mode="auto")
model.fit(xs_split, ys_split,
          epochs=1000,
          validation_split=0.2,
          shuffle=True,
          callbacks=[tensorboard, early_stop],
          )

model.save("./models/latest.h5f")
# When LSTM is done can use trainable=False to freeze it while training the other one
