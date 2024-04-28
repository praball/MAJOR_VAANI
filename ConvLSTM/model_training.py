import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (ConvLSTM2D, Dense, Dropout, Flatten,
                                     MaxPooling3D, TimeDistributed)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

features = np.load("features.npy")
labels = np.load("labels.npy")

one_hot_encoded_labels = to_categorical(labels)
features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels,
                                                                            test_size=0.20, shuffle=True,
                                                                            random_state=50)

def create_convlstm_model():
    model = Sequential()
    model.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=True, input_shape=(SEQUENCE_LENGTH,
                                                                                     IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(ConvLSTM2D(filters=14, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(Flatten())
    model.add(Dense(len(CLASSES_LIST), activation="softmax"))
    model.summary()
    return model

convlstm_model = create_convlstm_model()
convlstm_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

convlstm_model_training_history = convlstm_model.fit(x=features_train, y=labels_train, epochs=200, batch_size=4,
                                                     shuffle=True, validation_split=0.2,
                                                     callbacks=[early_stopping_callback])

model_file_name = 'convlstm_model.h5'
convlstm_model.save(model_file_name)
