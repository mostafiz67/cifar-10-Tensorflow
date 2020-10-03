"""
Author: Md Mostafizur Rahman
File: Preprocessing cifar-10 train and test Images

"""

import numpy as np
import cv2, os
import pandas as pd

from sklearn.preprocessing import LabelBinarizer

# project package
from .. import config

x_train = np.ndarray(
    (config.nb_train_samples, config.img_size, config.img_size, config.img_channel),
    dtype=np.float32
)


def normalization(x):
    x = np.divide(x, 255)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2)

    return x


def load_train_data():
    train_data_dir = os.path.join(config.dataset_path(), "train")
    train_images = sorted(
        os.listdir(train_data_dir), key=lambda x: int(x.split(".")[0])
    )
    train_images = [os.path.join(train_data_dir, img_path) for img_path in train_images]

    # Loding image labels
    train_labels_df = pd.read_csv(
        os.path.join(config.dataset_path(), "trainLabels.csv")
    )
    train_labels = train_labels_df["label"].values

    # One hot encoding image labels
    encoder = LabelBinarizer()
    y_labels = encoder.fit_transform(train_labels)

    print("Loding Train Images ...")
    # loding images from absolute directory using openCV
    for i, img_dir in enumerate(train_images):
        img = cv2.imread(img_dir)
        img = normalization(img)
        # print(img)
        x_train[i] = img

    # print((x_train.shape))
    return x_train, y_labels



# Loding Test Data
def load_test_data():
    test_data_dir = os.path.join(config.dataset_path(), "test")
    test_images = sorted(
        os.listdir(test_data_dir), key=lambda x: int(x.split(".")[0])
    )
    test_images = [os.path.join(test_data_dir, img_path) for img_path in test_images]
    
    start = 0
    
    nb_images = 50000
    for part in range(0,6):
        if not (part == 0): start +=nb_images
        end = start + nb_images
        
        #print(start, "", end)
        x_test = np.ndarray((nb_images, 
                             config.img_size, 
                             config.img_size, 
                             config.img_channel),
                            dtype=np.float32
)
        
        

    print("Loding test images from ", start, "to ", end)
  
    # loding images from absolute directory using openCV
    for i, img_dir in enumerate(test_images[start:end]):
        img = cv2.imread(img_dir)
        img = normalization(img)
        x_test[i] = img
        
        np.save(os.path.join(config.output_path(), 
                             "x_test_" + str(part)), x_test)
    
    del x_test
return x_test

def get_test_data_by_part(part):
    print("Loding test images part ", part, " from numpy array")
    np.save(os.path.join(config.output_path(), 
                             "x_test_" + str(part) + ".npy"))
    


if __name__ == "__main__":
    x, y = load_test_data()
    #print(x.shape)
    #print(y.shape)