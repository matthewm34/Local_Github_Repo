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

# imageDirectory = './2022Fimgs/'
# with open(imageDirectory + 'labels.txt', 'r') as f:
#     reader = csv.reader(f)
#     lines = list(reader)


#Train Test Split
#Randomly choose train and test data (80/20 split).
random.shuffle(lines)
train_lines = lines[:math.floor(len(lines)*.8)][:]
test_lines = lines[math.floor(len(lines)*.8):][:]

# this line reads in all images listed in the file in color, and resizes them to 25x33 pixels
# train_data = np.array([np.array(cv2.resize(cv2.imread(imageDirectory+train_lines[i][0]+".jpg"),(25,33))) for i in range(len(train_lines))])

# this line reads in all images listed in the file in color, leave images as original size pixels: 308 x 410
img_height = 308
img_width = 410
train_data = np.array([np.array(cv2.imread(imageDirectory+train_lines[i][0]+".jpg")) for i in range(len(train_lines))])
test_data = np.array([np.array(cv2.imread(imageDirectory+test_lines[i][0]+".jpg")) for i in range(len(test_lines))])

#currently not resizing the images for training, original size 
  # train_data = train_data.astype(np.float32)

# read in training labels
train_labels = np.array([np.int32(train_lines[i][1]) for i in range(len(train_lines))])
# train_labels = np.char.mod('%d', train_labels)
test_labels = np.array([np.int32(test_lines[i][1]) for i in range(len(test_lines))])
# test_labels = np.char.mod('%d', test_labels)

# add validation data set if necessary
# val_data = train_data[math.floor(len(train_lines)*.8):]
# train_data = train_data[:math.floor(len(train_lines)*.8)]
# val_labels = train_labels[math.floor(len(train_lines)*.8):]
# train_labels = train_labels[:math.floor(len(train_lines)*.8)]

num_classes = len(np.unique(train_labels)) # number of different classification classes

# ---------------------------------------------- Model Architecture ----------------------------------------------
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)), #this layer normalized the pixel values
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2), # add dropout layer
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activation="softmax") #output classification for the 6 classes
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# ----------------------------------------------Train Model----------------------------------------------
print('Training Model...')
epochs = 15
# train_history = model.fit(train_data, train_labels, epochs=epochs, validation_data =(test_data, test_labels))
train_history = model.fit(train_data, train_labels, epochs=epochs)

# model.save('CNN_v7_12epochs.h5')
# load_model('CNN_v7_12epochs.h5')

# ----------------------------------------------Test Model----------------------------------------------
predictions = model.predict(test_data)
test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
print('Classification Accuracy: ' +str(test_acc))
# score = tf.nn.softmax(predictions[0])

# just manually double checking that the accuracy it is saying is correct
# predictions_values = []
# for i in range(predictions.shape[0]):
#      cur_prediction_vec = predictions[i]
#      index = cur_prediction_vec.argmax(axis=0)
#      predictions_values.append(index)
# test = np.array(predictions_values)
# comparison = test == test_labels #mask for which True is redicted same as test data and False is wrong
# accuracy = sum(comparison)/len(test_labels)
# print('Accuracy:' + str(accuracy))

#Create confusion matrix and normalizes it over predicted (columns)
# result = confusion_matrix(test_data, predictions , normalize='pred')


