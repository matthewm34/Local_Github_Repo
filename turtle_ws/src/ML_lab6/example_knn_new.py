#!/usr/bin/env python3
import cv2
import sys
import csv
import time
import math
import numpy as np
import random

import matplotlib as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


### Load training images and labels
# ---------------------------------------------- Train Test Split Data----------------------------------------------
imageDirectory = './2023Fimgs/'

with open(imageDirectory + 'labels.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

#Train Test Split
#Randomly choose train and test data (70/30 split).
random.shuffle(lines)
train_lines = lines[:math.floor(len(lines)*.7)][:]
test_lines = lines[math.floor(len(lines)*.7):][:]

# this line reads in all images listed in the file in color, and resizes them to 25x33 pixels
# train_data = np.array([np.array(cv2.resize(cv2.imread(imageDirectory+train_lines[i][0]+".jpg"),(25,33))) for i in range(len(train_lines))])

# this line reads in all images listed in the file in color, leave images as original size pixels: 308 x 410
img_height = 308
img_width = 410
train_data = np.array([np.array(cv2.imread(imageDirectory+train_lines[i][0]+".jpg")) for i in range(len(train_lines))])
test_data = np.array([np.array(cv2.imread(imageDirectory+test_lines[i][0]+".jpg")) for i in range(len(test_lines))])

# here we reshape each image into a long vector and ensure the data type is a float (which is what KNN wants), note the *3 is due to 3 channels of color.
# train_data = train.flatten().reshape(len(train_lines), 33*25*3)
# train_data = train_data.astype(np.float32)

# read in training labels
train_labels = np.array([np.int32(train_lines[i][1]) for i in range(len(train_lines))])
# train_labels = np.char.mod('%d', train_labels)
test_labels = np.array([np.int32(test_lines[i][1]) for i in range(len(test_lines))])
# test_labels = np.char.mod('%d', test_labels)

# add validation data
val_data = train_data[math.floor(len(train_lines)*.8):]
train_data = train_data[:math.floor(len(train_lines)*.8)]
val_labels = train_labels[math.floor(len(train_lines)*.8):]
train_labels = train_labels[:math.floor(len(train_lines)*.8)]

num_classes = len(np.unique(train_labels)) # number of different classification classes

# ---------------------------------------------- Model Architecture ----------------------------------------------
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)), 
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activation="softmax")
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# ----------------------------------------------Train Model----------------------------------------------
epochs = 15
# train_history = model.fit(train_data, train_labels, epochs=epochs, validation_data =(test_data, test_labels))

# model.save('CNN_v6_15epochs.h5')
load_model('CNN_v6_15epochs.h5')

# ----------------------------------------------Test Model----------------------------------------------
predictions = model.predict(test_data)
test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
# score = tf.nn.softmax(predictions[0])

predictions_values = []
for i in range(predictions.shape[0]):
     cur_prediction_vec = predictions[i]
     index = cur_prediction_vec.argmax(axis=0)
     predictions_values.append(index)

predicitons_values = np.concatenate(predictions_values, axis=0)


#Create confusion matrix and normalizes it over predicted (columns)
result = confusion_matrix(test_data, predictions , normalize='pred')





test = 0

######




### Train classifier
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

if(__debug__):
	Title_images = 'Original Image'
	Title_resized = 'Image Resized'
	cv2.namedWindow( Title_images, cv2.WINDOW_AUTOSIZE )

correct = 0.0
confusion_matrix = np.zeros((6,6))

k = 7

for i in range(len(test_lines)):
    original_img = cv2.imread(imageDirectory+test_lines[i][0]+".jpg")
    test_img = np.array(cv2.resize(cv2.imread(imageDirectory+test_lines[i][0]+".jpg"),(25,33)))
    if(__debug__):
        cv2.imshow(Title_images, original_img)
        cv2.imshow(Title_resized, test_img)
        key = cv2.waitKey()
        if key==27:    # Esc key to stop
            break
    test_img = test_img.flatten().reshape(1, 33*25*3)
    test_img = test_img.astype(np.float32)

    test_label = np.int32(test_lines[i][1])

    ret, results, neighbours, dist = knn.findNearest(test_img, k)

    if test_label == ret:
        print(str(lines[i][0]) + " Correct, " + str(ret))
        correct += 1
        confusion_matrix[np.int32(ret)][np.int32(ret)] += 1
    else:
        confusion_matrix[test_label][np.int32(ret)] += 1
        
        print(str(test_lines[i][0]) + " Wrong, " + str(test_label) + " classified as " + str(ret))
        print("\tneighbours: " + str(neighbours))
        print("\tdistances: " + str(dist))



print("\n\nTotal accuracy: " + str(correct/len(test_lines)))
print(confusion_matrix)
