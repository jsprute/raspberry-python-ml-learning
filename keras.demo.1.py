
# Import required Keras libraries
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# Set a random seed
seed = 42
np.random.seed(seed)

# Load the MNIST dataset into training and test datasets.
(X_trains, y_train), (X_test, y_test) = mnist.load_data()

# Flatten the 28 x 28 image into a 784 element input data vector
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0],num_pixels).astype('float32')

# Normalize data input values from 0 - 255 to 0 - 1.0
X_train = X_train / 255.0
X_test = X_test / 255.0

# One hot encoding of the categorical outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test[1]

def baseline_model():
    # Create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer="normal", activation="relu"))
    model.add(Dense(num_classes, kernel_initializer="normal", activation="softmax"))
    #Compile model
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

# Run the demo
model = baseline_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print('Basline error: %.2f%%'%(100-scores[1]*100))