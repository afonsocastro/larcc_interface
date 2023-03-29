#!/usr/bin/env python3

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM


if __name__ == '__main__':

    params = 12
    time_steps = 20
    batch_size = 64
    epochs = 150
    validation_split = 0.7
    time_window = 2
    labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
    n_labels = len(labels)

    # Define the model architecture
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(time_steps, params)))
    model.add(Dense(n_labels, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

