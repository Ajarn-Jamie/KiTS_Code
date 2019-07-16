from time import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
from keras import backend as K

def weighted_binary_crossentropy(zero_weight, one_weight):

    def loss(y_true, y_pred):

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return loss

weights = np.ones((2,256,256,1))
weights[0,:,:,:] *= 0.5
weights[1,:,:,:] *= 10.0
loss = weighted_binary_crossentropy(weights[0], weights[1])

model = load_model('unet_model_Binary_99.h5',
                  custom_objects={'loss': loss})

cases = [0]
for i in cases:
##for i in range(0):
    start_time = time()
    print('case_00%.3i'%i)
    # Load processed data
    with open('D:\DATA_KITS2019\kits19\prep_data2\case_00%.3i.pickle'%i, 'rb') as f:
        x_prep, y_prep, restore_dat = pickle.load(f)
    rest_dat = np.array(restore_dat)

    x, y = [], []

    for i in range(x_prep.shape[0]):
        im = cv2.resize(x_prep[i],(256,256),interpolation=cv2.INTER_LINEAR)
        im = np.expand_dims(im,0)
        im = np.expand_dims(im,4)
        tar = cv2.resize(y_prep[i],(256,256),interpolation=cv2.INTER_LINEAR)
        tar = np.expand_dims(tar,0)
        tar = np.expand_dims(tar,4)
        x.append(im)
        y.append(tar)

    x = np.vstack(x)
    y_true = np.vstack(y)

    y_pred = model.predict(x)
