
# Import required Keras libraries
import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

#K.set_image_dim_ordering('th')
K.image_data_format()
# Set a random seed
seed = 42
np.random.seed(seed)

# import data
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()

### result
# 0 - T-shirt/top
# 1 - Trouser
# 2 - Pullover
# 3 - Dress
# 4 - Coat
# 5 - Sandal
# 6 - Shirt
# 7 - Sneaker
# 8 - Bag
# 9 - Ankle boot

# Flatten all of the 28 x 28 images into 784 element numpty input
# data vectors.
pixelNum = train_images.shape[1] * train_images.shape[2]
train_images = train_images.reshape(train_images.shape[0],1,28,28).astype('float32')
test_images = test_images.reshape(test_images.shape[0],1,28,28).astype('float32')

# Normalize inputs from 0-255 to 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

# One hot encoding
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_lables)
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot'] 

# Model definition
def cnn_model():
    # create model
    model = Sequential()
    # Convolutional layer
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation="relu"))
    # Pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout layer
    model.ad(Dropout(0.2))
    # Flatten layer
    model.add(Flatten())
    # Full connection layer
    model.add(Dense(128, activation="relu"))
    # Output layer
    model.add(Dense(numClass, activation="softmax"))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

# Run the demo and evaluate it
model = cnn_model()
model.fit(train_images,train_labels, validation_data=(test_images, test_labels), epochs=10, batch_size=200, verbose=2)

# final evaluation
scores = model.evaluate(test_images, test_labels, verbose=0)
print(scores[1])

