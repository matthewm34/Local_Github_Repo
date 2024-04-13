import math
import random
import PIL
import csv
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

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
    """
    Return the largest contour in a binary mask.

    Args:
        mask (np.ndarray): Binary mask.

    Returns:
        cv2.contour: largest contour in mask. or None if no contours.
    """
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


def get_cropped_image(image, mask, min_area=100, min_size=64, verbose=True):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if no contours are present in the image, return None
    if not contours:
        if verbose: print('no contours found.')
        return None 
    
    # Step 1 - filter by size: 
    # Remove any contours that are smaller than min_area
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if verbose: print(f'after removing by size: {len(valid_contours)} valid contours')

    # Step 2 - filter by centroid vertical location:
    # Remove contours with centroid that are on the top or bottom of image. Most likely just background contours.
    # if there is only one contour left, it might just be very big on the screen, widen the centroid limits
    if len(valid_contours) == 1:
        upper_limit = image.shape[0]*.10
        lower_limit = image.shape[0]*.90
    else:
        upper_limit = image.shape[0]*.25
        lower_limit = image.shape[0]*.75

    valid_contours_centroid = []
    for contour in valid_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cy = int(M["m01"] / M["m00"])
            if upper_limit < cy < lower_limit:
                valid_contours_centroid.append(contour)
    valid_contours = valid_contours_centroid
    if verbose:  print(f'after removing by vertical centroid: {len(valid_contours)} valid contours')

    # Step 3 - filter by size again:
    # remove again anything that is too small
    valid_contours = [contour for contour in valid_contours if cv2.contourArea(contour) >= min_area*2]
    if verbose:  print(f'after removing second size restriction: {len(valid_contours)} valid contours')

    # if we removed the "noisy" contours that are present on the top and bottom of the image due to background color
    # and we still have multiple contours in the center, there may be too many signs present and we should classify as 0.
    if len(valid_contours) > 1:
        if verbose:  print('after removing vertical centroids, too many contours are left')
        return None

    # Step 4 - filter by centroid horizontal location:
    # If centroid is too far on the sides of the screen, remove it.
    valid_contours_centroid = []
    left_limit = image.shape[1]*.05
    right_limit = image.shape[1]*.95
    
    for contour in valid_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            if left_limit < cx < right_limit:
                valid_contours_centroid.append(contour)

    valid_contours = valid_contours_centroid
    if verbose: print(f'after removing by horizontal centroid: {len(valid_contours)} valid contours')

    # if there are too many contours left, it can mean there are too many signs on the screen
    # and therefore we can classify as just the wall
    if len(valid_contours) > 3:
        if verbose: print(f'too many contours found: {len(valid_contours)}')
        return None

    # if there are no valid contours, we are at a wall
    if not valid_contours:
        if verbose: print('no valid contours found.')
        return None

    # Get the largest contour based on area
    largest_contour = max(valid_contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    # if the contour we found is sort of tilted (we are seeing it from an angle) don't count it
    if w < 30 or w < h*.7:
        if verbose: print('contour width too small')
        return None
    
    # Ensure the cropped area is at least the provided min_size x min_size
    if w < min_size or h < min_size:
        # Calculate how much to add to width and height
        add_w = max(0, min_size - w) // 2
        add_h = max(0, min_size - h) // 2

        # Adjust x, y, w, h to maintain the minimum size and try to center the contour
        x = max(0, x - add_w)
        y = max(0, y - add_h)
        w = w + 2 * add_w
        h = h + 2 * add_h

        # Check boundaries so we don't go out of the image limits
        x = min(x, image.shape[1] - min_size)
        y = min(y, image.shape[0] - min_size)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)

    # Crop the image to desired size
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image


def filter_img(img):
    """
    Filter our original image into just the region of interest.
    Crops down to a sign if present, or just a resizing of original image if no signs present.

    Args:
        img (np.ndarray): Original image.

    Returns:
        np.ndarray: Cropped image. In size 64x64x3.
    """

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # red mask
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # blue mask
    lower_blue = np.array([110, 30, 30])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

    # green mask
    lower_green = np.array([40, 25, 25])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(img_hsv, lower_green, upper_green)

    # combine all of the three masks together.
    mask = cv2.bitwise_or(red_mask, blue_mask)
    mask = cv2.bitwise_or(mask, green_mask)

    # crop the image to sign of interest
    cropped_img = get_cropped_image(img, mask, verbose=False)

    if cropped_img is None:
        cropped_img = cv2.resize(img, (64,64))
    else:
        cropped_img = cv2.resize(cropped_img, (64,64))

    return cropped_img


def check_extension(imageDir):
    """
    Checks the extension of the image files for the image directory.

    Args:
        imageDir (str): Directory to images

    Returns:
        str: extension of the image files in folder.
    """
    for file in os.listdir(imageDir):
        if file[-3:] == 'jpg':
            return 'jpg'
        elif file[-3:] == 'png':
            return 'png'
        else:
            continue


def get_dataset(imageDir):
    data_list = []
    
    with open(imageDir + 'labels.txt', 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)

    ext = check_extension(imageDir)
        
    for i, label in lines:
        # read in image and filter out to region of interest
        img = np.array(cv2.imread(f'{imageDir}{i}.{ext}'))
        filtered_img = filter_img(img) # either a 64x64x3 image of sign or just the original image
        data_list.append((filtered_img, int(label)))
            
    random.shuffle(data_list)

    # divide data into test, val, and training set (20 / 16 / 64 split)
    train_ind = range(len(data_list))
    test_ind = random.sample(train_ind, int(len(train_ind)*0.2))
    train_ind = [item for item in train_ind if item not in test_ind]
    val_ind = random.sample(train_ind, int(len(train_ind)*0.2))
    train_ind = [item for item in train_ind if item not in val_ind]

    # convert from list into numpy array
    test_trials = np.array(data_list,dtype=object)[test_ind]
    val_trials = np.array(data_list,dtype=object)[val_ind]
    train_trials = np.array(data_list,dtype=object)[train_ind]
    
    return test_trials, val_trials, train_trials


def create_cnn(input_shape, num_classes, lr=0.00001):
#     model = Sequential([
#         layers.Rescaling(1./255, input_shape=input_shape), #this layer normalized the pixel values
# #         layers.Conv2D(16, 3, input_shape=input_shape, padding='same', 
# #                       activation='relu', kernel_regularizer=l2(0.01)),
# #         layers.MaxPooling2D((2,2)),
# #         layers.Dropout(0.3),
#         layers.Conv2D(32, 3, padding='same', activation='relu',
#                       kernel_regularizer=l2(0.01)),
#         layers.MaxPooling2D((2,2)),
#         layers.Dropout(0.4),
#         layers.Conv2D(64, 3, padding='same', activation='relu'),
#         layers.MaxPooling2D(),
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dropout(0.3),
#         layers.Dense(num_classes) #output classification for the 6 classes
#     ])
    
    model = Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),
        layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape,
                            kernel_regularizer=l2(0.01)),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),
        layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape,
                            kernel_regularizer=l2(0.01)),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes)
    ])
    
    
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    model.summary()
    
    return model


def train_cnn(train_data, val_data, plot=True):
    input_shape = train_data[0][0].shape
    num_classes = len(np.unique(train_data[:,1]))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    model = create_cnn(input_shape, num_classes, lr=0.0001)

    x_train = []
    y_train = []
    x_val = []
    y_val = []

    for data, label in train_data:
        x_train.append(data)
        y_train.append(label)

    for data, label in val_data:
        x_val.append(data)
        y_val.append(label)    

    x_train = np.array(x_train, dtype=object).astype(np.float32)
    y_train = np.array(y_train, dtype=object).astype(np.float32)
    x_val = np.array(x_val, dtype=object).astype(np.float32)
    y_val = np.array(y_val, dtype=object).astype(np.float32)

    history = model.fit(x_train, y_train, epochs=500, batch_size=32, validation_data=(x_val, y_val), 
                        callbacks=[es])
    
    if plot:
        plt.plot(history.history['loss'],label='training_loss')
        plt.plot(history.history['val_loss'],label='validation_loss')
        plt.legend()

        plt.figure()
        plt.plot(history.history['accuracy'],label='training_acc')
        plt.plot(history.history['val_accuracy'],label='val_acc')
        plt.legend()
        plt.show()

    return model


def test_model(test_data, model):
    x_test = []
    y_test = []

    for data, label in test_data:
        x_test.append(data)
        y_test.append(label)

    x_test = np.array(x_test)
    y_test = np.array(y_test)


    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=-1)    

    test_acc = len(np.where(pred == y_test)[0])/len(y_test)
    result = confusion_matrix(y_test, pred)

    print(f'Total accuracy: {test_acc}')
    print(result)