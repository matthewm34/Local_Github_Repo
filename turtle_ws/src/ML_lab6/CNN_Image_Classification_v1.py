#!/usr/bin/env python3
import cv2
import sys
import csv
import time
import math
import numpy as np
import random

import matplotlib as plt
import matplotlib.pyplot as pyplot
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

#try preprocessing the training data with grayscaale 
# weights = np.array([0.2989, 0.5870, 0.1140])
# train_data = np.dot(train_data[...,:3], weights)

num_classes = len(np.unique(train_labels)) # number of different classification classes

# ----------------------------------------------- Preprocessing -----------------------------------------------
#  
img_train_stack = []
for i in range(train_data.shape[0]): # for each image
  cur_image = train_data[i,:,:,:] #get image of shape 308, 410, 3
  
  # cv2.imshow('OG Image', cur_image) # show current image
  hsv_cur_image = cv2.cvtColor(cur_image, cv2.COLOR_RGB2HSV) #convert rgb to hsv
  # COLOR_RGB2HSV
  #define thresholds $ Hue (H): 0 to 179, Saturation (S): 0 to 255, Value (V): 0 to 255
  lower_green = np.array([36, 70, 70])
  upper_green = np.array([74, 255, 255])

  #define thresholds
  lower_blue = np.array([0, 50, 50])
  upper_blue = np.array([30, 255, 255])

  #define thresholds
  lower_orange = np.array([115, 100, 100])
  upper_orange = np.array([145, 255, 255])

  #define thresholds
  lower_white = np.array([0, 0, 100])
  upper_white = np.array([255, 100, 255])

  #mask
  mask_green = cv2.inRange(hsv_cur_image, lower_green, upper_green)
  mask_blue = cv2.inRange(hsv_cur_image, lower_blue, upper_blue)
  mask_orange = cv2.inRange(hsv_cur_image, lower_orange, upper_orange)
  white_mask = cv2.inRange(hsv_cur_image, lower_white, upper_white)

  # cv2.imshow('Green Parts Only', mask_green)
  # cv2.imshow('Blue Parts Only', mask_blue)
  # cv2.imshow('Orange Parts Only', mask_orange)
  # cv2.imshow('White Parts Only', white_mask)

  cur_stack = np.stack((mask_green,mask_blue,mask_orange),axis=2)
  img_train_stack.append(cur_stack)
  #find anything that is blue green or orange
  test = 0

train_data = np.stack(img_train_stack,axis=0)

img_test_stack = []
for i in range(test_data.shape[0]): # for each image
  cur_image = test_data[i,:,:,:] #get image of shape 308, 410, 3
  
  # cv2.imshow('OG Image', cur_image) # show current image
  hsv_cur_image = cv2.cvtColor(cur_image, cv2.COLOR_RGB2HSV) #convert rgb to hsv
  # COLOR_RGB2HSV
  #define thresholds $ Hue (H): 0 to 179, Saturation (S): 0 to 255, Value (V): 0 to 255
  lower_green = np.array([36, 70, 70])
  upper_green = np.array([74, 255, 255])

  #define thresholds
  lower_blue = np.array([0, 50, 50])
  upper_blue = np.array([30, 255, 255])

  #define thresholds
  lower_orange = np.array([115, 100, 100])
  upper_orange = np.array([145, 255, 255])

  #define thresholds
  lower_white = np.array([0, 0, 100])
  upper_white = np.array([255, 100, 255])

  #mask
  mask_green = cv2.inRange(hsv_cur_image, lower_green, upper_green)
  mask_blue = cv2.inRange(hsv_cur_image, lower_blue, upper_blue)
  mask_orange = cv2.inRange(hsv_cur_image, lower_orange, upper_orange)
  white_mask = cv2.inRange(hsv_cur_image, lower_white, upper_white)

  # cv2.imshow('Green Parts Only', mask_green)
  # cv2.imshow('Blue Parts Only', mask_blue)
  # cv2.imshow('Orange Parts Only', mask_orange)
  # cv2.imshow('White Parts Only', white_mask)

  cur_stack = np.stack((mask_green,mask_blue,mask_orange),axis=2)
  img_test_stack.append(cur_stack)
  #find anything that is blue green or orange
  test = 0

test_data = np.stack(img_test_stack,axis=0)



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
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# ----------------------------------------------Train Model----------------------------------------------
print('Training Model...')
epochs = 7
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
predictions_values = []
for i in range(predictions.shape[0]):
     cur_prediction_vec = predictions[i]
     index = cur_prediction_vec.argmax(axis=0)
     predictions_values.append(index)
test_predictions = np.array(predictions_values)
comparison = test_predictions == test_labels #mask for which True is redicted same as test data and False is wrong
accuracy = sum(comparison)/len(test_labels)
# print('Accuracy:' + str(accuracy))
test = 0

#show confusion matrix
ConfusionMatrixDisplay.from_predictions(test_labels, test_predictions)

test = 0
#Create confusion matrix and normalizes it over predicted (columns)
# result = confusion_matrix(test_data, predictions , normalize='pred')

