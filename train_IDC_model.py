# Set the matplotlib backend so figures cna be saved in the 
# background
import matplotlib
matplotlib.use("Agg")

# Import the required libraries
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adagrad
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from cancernet import CancerNet
import config_IDC as config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png", 
    help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# Initialize the number of epochs, initial learning rate, and
# batch size.
NUM_EPOCHS = 4
INIT_LR = 1e-2
BS = 32

# Determine the total number of image paths in training,
# validation, and the testing directories.
trainPaths = list(paths.list_images(config.TRAIN_PATH))
totalTrain = len(trainPaths)
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))

# Account for skew in the labeled data
trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = np_utils.to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# Initialize the training data augmentation object
trainAug = ImageDataGenerator(
    rescale = 1/255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest")

# Initialize the validation (and testing) data augmentation
# object.
valAug = ImageDataGenerator(rescale=1 / 255.0)

# Initialize the training generator.
trainGen = trainAug.flow_from_directory(
    config.TRAIN_PATH,
    class_mode="categorical",
    target_size=(48, 48),
    color_mode="rgb",
    shuffle=True,
    batch_size=BS)

# Initialize the validation generator.
valGen = valAug.flow_from_directory(
    config.VAL_PATH,
    class_mode="categorical",
    target_size=(48, 48),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)

# Initialize the testing generator.
testGen = valAug.flow_from_direcotry(
    config.TEST_PATH,
    class_mode="categorical",
    target_size=(48,48),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)

# Initialize the CancerNet model and compile it.
model = CancerNet.build(width=48, height=48, depth=3, classes=2)
opt = Adagrad(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Fit the model.
H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // BS,
    validation_data=valGen,
    validation_steps = totalVal // BS,
    class_weight=classWeight,
    epochs=NUM_EPOCHS)

# Reset the testing generator and then use the trained model to
# make predictions on the data.
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen, setps=(totalTest // BS) + 1)

# For each image in the testing set find the index of the
# label with corresponding larget predicted probability.
predIdxs = np.argmax(predIdxs, axis=1)

# Show a nicely formatted classification report.
print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys()))

# Compute the confusion matrix and use it to derive the raw
# accuracy, sensitivity, and specificity.
cm = confusion_matrix(testGen.classes, predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# Show the confusion matrix, accuracy, sensitivity, and 
# specificty.
print(cm)
print("acc: {:.4f}".format(acc))
print("specificity: {:.4f}".format(specificity))

# Plot the training loss and accuracy.
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()

plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])


