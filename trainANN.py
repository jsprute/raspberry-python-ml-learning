# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from ann import ANN

# Set up the ANN configuration
inode = 785
hnode = 100
onode = 10

# set the learning rate
lr = 0.2

# Instantiate an ANN object names ann
ann = ANN(inode, hnode, onode, lr)

# Create the training list data
dataFile = open('./data/mnist_train_100.csv')
dataList = dataFile.readlines()
dataFile.close()

# Create the test list data
testDataFile = open('./data/mnist_test_10.csv')
testDataList = testDataFile.readlines()
testDataFile.close()

# Train the ANN using all the records in the list
for record in dataList:
    recordx = record.split(',')
    inputT = (np.asfarray(recordx[:])/255.0*0.99) + 0.01
    train = np.zeros(onode) + 0.01
    train[int(recordx[0])] = 0.99
    # Training begins here
    ann.trainNet(inputT, train)

# Iterate through all 10 test records and display output
# data vectors
match = 0
no_match = 0
for record in testDataList:
    recordz = record.split(',')
    # Determine record's
    labelz = int(recordz[0])
    # Adjust record values for ANN
    inputz = (np.asfarray(recordz[1:])/255.0*0.99)+0.01
    outputz = ann.testNet(inputz)
    #print('output for label = ', labelz)
    #print(outputz)
    max_value = np.argmax(outputz)
    if max_value == labelz:
        match = match + 1
    else:
        no_match = no_match + 1
    print('Expected {0} and received {1}'.format(labelz, max_value))
    success = float(match) / float(match + no_match)
print('success rate = {0}'.format(success))



