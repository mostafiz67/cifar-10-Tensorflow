"""
Author: Md Mostafizur Rahman
File: Network Architecture for Cifar-10 dataset image
"""


import keras, os
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

# project modules
from .. import config

model_checkpoint_dir = os.path.join(config.checkpoint_path(), "baseline.h5")
save_model_dir = os.path.join(config.output_path(), "baseline.h5")

# defining CNN model
def get_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=config.img_shape))
    model.add(Activation("relu"))

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=config.img_shape))
    model.add(Activation("relu"))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate = 0.20))
    
    model.add(Conv2D(64, (3, 3), padding="same", input_shape=config.img_shape))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3), padding="same", input_shape=config.img_shape))
    model.add(Activation("relu"))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate = 0.30))
    
    model.add(Flatten())
    model.add(Dense(384, 
                    kernel_regularizer = keras.regularizers.L2(0.01)))
    model.add(Activation("relu"))
    model.add(Dropout(rate = 0.30))
    
    model.add(Dense(config.nb_classes))
    model.add(Activation("softmax"))

    return model



def read_model():
    model = load.model(save_model_dir)
    
def save_model_checkpint():
    return ModelCheckpoint(model_checkpoint_dir, 
                           monitor='val_loss', 
                           verbose=2, 
                           save_best_only=True, 
                           save_weights_only=False, 
                           mode='auto', 
                           period=1
                           )
    
def set_early_stopping():
    return EarlyStopping(monitor='val_loss', 
                         patience=15, 
                         verbose=2, 
                         mode='auto'
                         )

if __name__ == "__main__":
    m = get_model()
    m.summary()
