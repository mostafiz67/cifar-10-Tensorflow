""""

Author: Md MOstafizur Rahman
File: Configaration file for kaggle cifar-10 project
"""

import os

nb_train_samples = 50000
nb_test_samples = 300000
nb_classes = 10

img_size = 32
img_channel = 3
img_shape = (img_size, img_size, img_channel)
lr = 0.001
batch_size = 2
nb_epochs = 3


def root_path():
    return os.path.dirname(__file__)


def checkpoint_path():
    return os.path.join(root_path(), "checkpoints")


def dataset_path():
    return os.path.join(root_path(), "dataset")


def submission_path():
    return os.path.join(root_path(), "submission")


def src_path():
    return os.path.join(root_path(), "src")

def output_path():
    return os.path.join(root_path(), "output")