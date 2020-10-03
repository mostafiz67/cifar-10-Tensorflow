""""
Author: Am Mostafizur Rahman
File: Testing and submission kaggle cifar-10 test images
"""

import numpy as np
import pandas as pd
import os

#project modules
from ..import config
from . import preprocess, my_model

#loading model
model = my_model.read_model()

#loding test data
resylt = []
for part in range(0, 6):
    x_test = preprocess.get_test_data_by_part(part)
    #predicting results
    print("Predicting result")
    predictions = model.predict(x_test,
                                batch_size = config.batch_sise,
                                verbose = 2)
    
    #print(len(predictions))
    label_pred = np.argmx(predictions, axes = 1)
    #print("label_pred")
    
    