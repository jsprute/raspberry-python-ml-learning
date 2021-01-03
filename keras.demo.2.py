
# Import required Keras libraries
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# import data
from keras.datasets import fashion_mnist
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

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot'] 


# Set a random seed
seed = 42
np.random.seed(seed)

print('Number of training records and size of each training record: ', train_images.shape)
print()
print('Number of training labels: ',len(train_labels))
print()
print('Training label: ', train_labels)
print()
print('Number of test records and size of each test record: ', test_images.shape)
print()
print('Number of test labels: ', len(test_labels))
print()