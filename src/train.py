"""
Author: Md Mostafziur Rahman
File: Training a CNN architecture using cifar-10 dataset

"""


import keras
import numpy as np

#project Modules
from .. import config
from . import my_model, preprocess
#Loding Data
x_train, y_train = preprocess.load_train_data()
print("Train Data shape: ", x_train.shape)
print("Train Label Shape: ", y_train.shape)

#Loding model
model = my_model.get_model()

#compile

model.compile(keras.optimizers.Adam(config.lr),
              keras.losses.categorical_crossentropy,
              metrics = ['accurancy'])

# Check point
model.cp = my_model.save_model_checkpoints()
early_stopping = my_model.set_early_stopping()

#model training
model.fit(x_train, y_train,
          batch_size = config.batch_size,
          epocs = config.nb_epocs,
          verbose=2,
          shuffle = True,
          callbacks = [early_stopping, model_cp],
          validation_split = 0.2)