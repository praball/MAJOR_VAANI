import math
import os
import pathlib

import cv2
import matplotlib.image as img
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import PIL.Image as Image
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from keras import layers, models
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

dataset_dir = 'E:\Programming\Major project trial\Image'
dataset_dir = pathlib.Path(dataset_dir)
print(dataset_dir)

characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              '1', '2', '3', '4', '5', '6', '7', '8', '9']

dict = {}

for char in characters:
    dict[char] = list(dataset_dir.glob(f'{char}/*'))[:400] #:400 or :600  

print(img.imread(dict['A'][0]).shape)

X,Y=[],[]
for label in dict.keys():
     for image in dict[label]:
        X.append(img.imread(image)/255.00)
        Y.append(label)

X=np.array(X)
Y=np.array(Y)

labels={'A': 0 ,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'H': 7,
        'I': 8,'J': 9,'K': 10,'L': 11,'M': 12,'N': 13,'O': 14,'P': 15,
        'Q': 16,'R': 17,'S': 18,'T': 19,'U': 20,'V': 21,'W': 22,'X': 23,'Y': 24,'Z': 25,
        '1': 26,'2': 27,'3': 28,'4': 29,'5': 30,'6': 31,'7': 32,'8': 33,'9': 34}

def func(i):
    return labels[i]

func=np.vectorize(func)
Y=func(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.3)

y_train=to_categorical(y_train,35)