import numpy as np
import matplotlib.pyplot as plt

dataFile = open('./data/mnist_train_100.csv')
dataList = dataFile.readlines()
dataFile.close()

print('Enter record number to view: ', end = ' ')
num = input()

record = dataList[int(num)].split(',')

imageArray = np.asfarray(record[1:]).reshape(28,28)

plt.imshow(imageArray, cmap="Greys", interpolation="None")
plt.show()
