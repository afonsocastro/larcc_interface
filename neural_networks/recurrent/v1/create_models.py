#!/usr/bin/env python3

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout # create model


def create_lstm_v1_0():
    model = Sequential()
    model.add(LSTM(16, input_shape=(20, 12)))
    model.add(Dense(4, activation="softmax"))
    return model, "lstm_v1_0"


def create_lstm_v1_1():
    model = Sequential()
    model.add(LSTM(16, input_shape=(20, 12), return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(4, activation="softmax"))
    return model, "lstm_v1_1"


def create_lstm_v1_2():
    model = Sequential()
    model.add(LSTM(16, input_shape=(20, 12), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(16))
    model.add(Dense(4, activation="softmax"))
    return model, "lstm_v1_2"


def create_lstm_v1_3():
    model = Sequential()
    model.add(LSTM(64, input_shape=(20, 12)))
    model.add(Dense(4, activation="softmax"))
    return model, "lstm_v1_3"


def create_lstm_v1_4():
    model = Sequential()
    model.add(LSTM(64, input_shape=(20, 12), return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(4, activation="softmax"))
    return model, "lstm_v1_4"


def create_lstm_v1_5():
    model = Sequential()
    model.add(LSTM(64, input_shape=(20, 12), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dense(4, activation="softmax"))
    return model, "lstm_v1_5"

