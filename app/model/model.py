import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, TimeDistributed, Flatten
import os

def load_model():
    model = Sequential()
    model.add(Conv3D(128, 3, activation='relu', padding='same', input_shape=(75,46,140,1)))
    model.add(MaxPool3D((1,2,2)))
    model.add(Conv3D(256, 3, activation='relu', padding='same'))
    model.add(MaxPool3D((1,2,2)))
    model.add(Conv3D(75, 3, activation='relu', padding='same'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))
    
    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))
    
    model.load_weights(os.path.join('..','model','checkpoint'))
    
    return model