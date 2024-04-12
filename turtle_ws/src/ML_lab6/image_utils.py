import math
import random
import PIL
import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.callbacks import EarlyStopping
from keras.regularizers import l2


def find_largest_contour(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour
    return None


def crop_to_largest_cluster(image, mask):
    # Find the largest contour in the mask
    largest_contour = find_largest_contour(mask)
    if largest_contour is not None:
        # Compute the bounding rectangle for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Crop the image using the bounding rectangle
        return image[y:y+h, x:x+w]
    return image  # Return original if no contours found


def filter_img(img, color):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if color == 'r':
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        # Create masks for red color
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
        mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif color == 'b':
        lower_blue = np.array([110, 30, 30])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
    else:
        lower_green = np.array([40, 25, 25])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(img_hsv, lower_green, upper_green)

    cropped_img = crop_to_largest_cluster(img, mask)
    filtered_img = cv2.bitwise_and(img, img, mask=mask)

    return cropped_img


def get_dataset(imageDir):
    data_list = []
    
    with open(imageDir + 'labels.txt', 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
        
    for i, label in lines:
        # read in image
        img = np.array(cv2.imread(f'{imageDir}{i}.png'))
        if img.shape == ():
            img = np.array(cv2.imread(f'{imageDir}{i}.jpg'))

        red_img = filter_img(img, 'r')
        blue_img = filter_img(img, 'b')
        green_img = filter_img(img, 'g')

        # concatenated_image = cv2.hconcat([img, red_img, blue_img, green_img])
        # cv2.imshow('OG image', concatenated_image)
        cv2.imshow('red',red_img)
        cv2.imshow('blue',blue_img)
        cv2.imshow('green',green_img)
        
        cv2.waitKey(0)
            
        # img = cv2.resize(img, (25,33))
        # data_list.append((img, int(label)))
            
    # random.shuffle(data_list)
    
    # train_ind = range(len(data_list))
    # test_ind = random.sample(train_ind, int(len(train_ind)*0.2))
    # train_ind = [item for item in train_ind if item not in test_ind]
    # val_ind = random.sample(train_ind, int(len(train_ind)*0.2))
    # train_ind = [item for item in train_ind if item not in val_ind]

    # test_trials = np.array(data_list,dtype=object)[test_ind]
    # val_trials = np.array(data_list,dtype=object)[val_ind]
    # train_trials = np.array(data_list,dtype=object)[train_ind]
    
    # return test_trials, val_trials, train_trials

# imageDirs = ['2024Simgs/','2023Fimgs/','S2023_imgs/','2022Fimgs/','2022Fheldout/']
imageDirs = ['2024Simgs/']
             
for imdir in imageDirs:
    get_dataset(imdir)

