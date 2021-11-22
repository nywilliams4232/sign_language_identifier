from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Activation, Dropout
from tensorflow.keras.models import Sequential
from numpy import argmax

class SLNN:
    def __init__(self):
        self.model = Sequential([
            #1 Convolution One (captures lower level features)
            Conv2D(32, kernel_size=(3, 3), input_shape=(28,28,1)), #collects relevent info
            Activation('relu'),
            MaxPool2D(pool_size=(2,2), strides=(2, 2)), #Gathers together all the maximum values in 'pools' to reduce the number of dimensions

            #2 Convlution layer Two (captures higher level features)
            Conv2D(32, kernel_size=(3, 3)),
            Activation('relu'),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

            Flatten(), #Basically turns final convolved image into 1D vector

            #2 hidden layers
            Dense(128, activation='relu'),
            Dropout(.2),
            Dense(64, activation='relu'),
            Dropout(.2),

            #Output layer
            Dense(26, activation='softmax')
        ])

        self.model.compile(loss="sparse_categorical_crossentropy",
                           optimizer='adam',
                           metrics=['accuracy'])

    def fit(self, X, y, batch_size=100, epochs=10, validation_data=None):
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_data=validation_data)

    def predict(self, X):
        # Returns the highest possible solution
        return self.model.predict([X])


    def save_weights(self, file_path="weights_slnn.w"):
        self.model.save_weights(file_path)

    def load_weights(self, file_path="weights_slnn.w"):
        self.model.load_weights(file_path)
