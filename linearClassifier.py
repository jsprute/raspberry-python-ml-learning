#import libs
import numpy as np
import cv2


# Use y = mx + b equation
# m is slope, b is y-intercept
def computeErrorForLineGivenPoints(b, m, points): 
    totalError = 0
    for i in range(0, len(points)):
        totalError += (points[i].y - (m * points[i].x + b)) ** 2
    return totalError / float(len(points))


def stepGradient(b_current, m_current, points, learningRate): 
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        b_gradient += -(2/N) * (points[i].y - ((m_current*points[i].x) + b_current))
        m_gradient += -(2/N) * points[i].x * (points[i].y - ((m_current * points[i].x) + b_current))
        new_b = b_current - (learningRate * b_gradient)
        new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


#init class labels and seed random
labels = ['dog','cat','squirrel']
np.random.speed(1)

#randomly initialize the weighted matrix and bias vector
W = np.random.randn(3,3072)
b = np.random.randn(3)

#set the font to draw the label
font = cv2.FONT_HERSHEY_SIMPLEX

#load the image and resize it.  Image is taken from dataset
orig = cv2.imread('dog.png')
image = cv2.resize(orig, (32,32)).flatten()

# Compute output scores
scores = W.dot(image) + b

# Loop over the scores and labels
for (label, score) in zip(labels, scores):
    print('[INFO] {}: {:.2f}'.format(label, score))

#Get the class label for the highest scoring class
classLabel = labels[np.argmax(scores)]

#Draw the predicted label on the original image
cv2.putText(orig, classLabel, (10,30), font, 0.9, (255,0,0), 2)

#Display the image
cv2.imshow('Image', orig)
cv2.waitKey(0)