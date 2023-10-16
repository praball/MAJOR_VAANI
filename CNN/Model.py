import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# spark
# CNN model
cnn = keras.models.Sequential()
cnn.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(128, 128, 3)))
cnn.add(keras.layers.MaxPool2D(pool_size=2))
cnn.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
cnn.add(keras.layers.MaxPool2D(pool_size=2))
cnn.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
cnn.add(keras.layers.MaxPool2D(pool_size=2))
cnn.add(keras.layers.Flatten())
cnn.add(keras.layers.Dropout(rate=0.2, seed=100))
cnn.add(keras.layers.Dense(units=256, activation='relu'))
cnn.add(keras.layers.Dense(units=128, activation='relu'))
cnn.add(keras.layers.Dense(units=35, activation='softmax'))

cnn.compile(optimizer='nadam', metrics=['accuracy'], loss='categorical_crossentropy')

cnn.fit(datagen.flow(X_train, y_train, batch_size=128), epochs=10, validation_data=(X_test, y_test), callbacks=[early_stopping])