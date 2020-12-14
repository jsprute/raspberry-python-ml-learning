
# Import libs
from ann import  ANN

# Create input data vector
inputT = [0.2, 0.9, 0.4]

# Display it
print('Input data vector')
print(inputT)
print()

# Train for 1 iteration
train = inputT
ann = ANN(3,3,3,0.3)
output = ann.testNet(inputT)

#Display output
print('After one iteration')
print(output)
print()

matrixList = ann.getMatrices()
print('wtgih matrix')
print(matrixList[0])
print()
print('wtgho matrix')
print(matrixList[1])
print()

#Train for 499 interations
for i in range(499):
    ann.trainNet(inputT, train)

output = ann.testNet(inputT)

# Display output
print('After 500 iterations')
print(output)
print()

matrixList = ann.getMatrices()
print('wtgih matrix')
print(matrixList[0])
print()
print('wtgho matrix')
print(matrixList[1])
print()


#Train for 500 interations
for i in range(500):
    ann.trainNet(inputT, train)

output = ann.testNet(inputT)

# Display output
print('After 1000 iterations')
print(output)
print()

matrixList = ann.getMatrices()
print('wtgih matrix')
print(matrixList[0])
print()
print('wtgho matrix')
print(matrixList[1])
print()
