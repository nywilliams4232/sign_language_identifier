from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Activation
from tensorflow.keras.models import Sequential
from numpy import argmax

class SLNN:
    def __init__(self):
        self.model = Sequential([
            #1 Convolution One (captures lower level features)
            Conv2D(64, kernel_size=(3,3), input_shape=(28,28, 1)), #collects relevent info
            Activation('relu'),
            MaxPool2D(pool_size=(2,2)), #Gathers together all the maximum values in 'pools' to reduce the number of dimensions

            #2 Convlution layer Two (captures higher level features)
            Conv2D(64, kernel_size=(3, 3)),
            Activation('relu'),
            MaxPool2D(pool_size=(2, 2)),

            Flatten(), #Basically turns final convolved image into 1D vector

            #3 hidden layers
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),

            #Output layer
            Dense(24, activation='softmax')
        ])

        self.model.compile(loss="categorical_crossentropy",
                           optimizer='adam',
                           metrics=['mean_squared_error', 'accuracy'])

    def fit(self, X, y, batch_size=100, validation_split=0.1, epochs=10):
        self.model.fit(X, y, batch_size=batch_size, validation_split=validation_split, epochs=epochs)

    def predict(self, X):
        return argmax(self.model.predict(X)) #Returns the highest possible solution

    def save_weights(self, file_path="weights_slnn.w"):
        self.model.save_weights(file_path)

    def load_weights(self, file_path="weights_slnn.w"):
        self.model.load_weights(file_path)
