import random
import csv
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
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


def get_cropped_image(image, mask, min_area=50, min_size=32, verbose=True):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if no contours are present in the image, return None
    if not contours:
        if verbose: print('no contours found.')
        return None 
    
    # Step 1 - filter by centroid vertical location:
    # Remove contours with centroid that are on the top or bottom of image. Most likely just background contours.
    # if there is only one contour left, it might just be very big on the screen, widen the centroid limits
    if len(contours) == 1:
        upper_limit = image.shape[0]*.10
        lower_limit = image.shape[0]*.90
    else:
        upper_limit = image.shape[0]*.25
        lower_limit = image.shape[0]*.75

    valid_contours_centroid = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cy = int(M["m01"] / M["m00"])
            if upper_limit < cy < lower_limit:
                valid_contours_centroid.append(contour)
    valid_contours = valid_contours_centroid
    if verbose:  print(f'after removing by vertical centroid: {len(valid_contours)} valid contours')

    # mask1 = np.zeros(mask.shape, dtype=np.uint8)
    # cv2.drawContours(mask1, valid_contours, -1, (255), thickness=cv2.FILLED)

    # Step 2 - filter by size:
    # remove again anything that is too small
    valid_contours = [contour for contour in valid_contours if cv2.contourArea(contour) >= min_area]
    if verbose:  print(f'after removing second size restriction: {len(valid_contours)} valid contours')

    # mask2 = np.zeros(mask.shape, dtype=np.uint8)
    # cv2.drawContours(mask2, valid_contours, -1, (255), thickness=cv2.FILLED)

    # if there are no valid contours, we are at a wall
    if not valid_contours:
        if verbose: print('no valid contours found.')
        return None
    
    # Step 3 - filter by centroid horizontal location:
    # If centroid is too far on the sides of the screen, remove it.
    valid_contours_centroid = []
    if len(valid_contours) == 1:
        left_limit = image.shape[1]*.1
        right_limit = image.shape[1]*.9
    else:
        left_limit = image.shape[1]*.2
        right_limit = image.shape[1]*.8
    
    for contour in valid_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            if left_limit < cx < right_limit:
                valid_contours_centroid.append(contour)
    valid_contours = valid_contours_centroid
    if verbose: print(f'after removing by horizontal centroid: {len(valid_contours)} valid contours')

    if len(valid_contours) > 1:
        sorted_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        area_diff = cv2.contourArea(sorted_contours[1]) / cv2.contourArea(sorted_contours[0])

        # check the area of the two largest contours
        if area_diff > 0.4:
            if verbose:  print('the contours left are too similar in size')
            return None

    # mask3 = np.zeros(mask.shape, dtype=np.uint8)
    # cv2.drawContours(mask3, valid_contours, -1, (255), thickness=cv2.FILLED)

    if not valid_contours:
        if verbose: print('no valid contours found.')
        return None

    largest_contour = max(valid_contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 230:
        if verbose: print('final contour is too small or too big')
        return None

    x, y, w, h = cv2.boundingRect(largest_contour)

    # if the contour we found is sort of tilted (we are seeing it from an angle) don't count it
    if w < h*.7:
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

    # concat_img = cv2.hconcat([mask,mask1,mask2,mask3,mask4])
    # cv2.imshow('og / size / vert_cent / size compare / horiz_cent', concat_img)

    return cropped_image


def filter_img(img, train=False):
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
        no_change_flag = True
    else:
        cropped_img = cv2.resize(cropped_img, (64,64))
        no_change_flag = False

    # cv2.imshow('original',img)
    # cv2.imshow('cropped',cropped_img)
    # cv2.waitKey(0)

    return cropped_img, no_change_flag


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


def get_dataset(imageDir, train=False, grayscale=False):
    data_list = []
    
    with open(imageDir + 'labels.txt', 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)

    ext = check_extension(imageDir)
    skip_count = 0
    mislabel_count = 0
        
    for i, label in lines:
        # read in image and filter out to region of interest
        img = np.array(cv2.imread(f'{imageDir}{i}.{ext}'))
        if train:
            filtered_img, no_change_flag = filter_img(img)
            if (no_change_flag and int(label) != 0):
                # if we didn't detect a sign, but the label is a sign, don't include in dataset (case 1)
                skip_count += 1
                continue
            if (not no_change_flag and int(label) == 0):
                # if we detected a sign, but the label is not a sign, don't include in dataset (case 2)
                filtered_img = cv2.resize(img, (64,64))
                mislabel_count += 1
                continue
        else:
            filtered_img, _ = filter_img(img) # either a 64x64x3 image of sign or just the original image

        if grayscale:
            filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)

        data_list.append((filtered_img, int(label)))
    
    if train:
        random.shuffle(data_list)

    print(f'Total of {skip_count}/{len(lines)} images were skipped due to imperfect cropping.')
    print(f'Total of {mislabel_count}/{len(lines)} images were relabeled due to imperfect cropping.')

    return data_list


def get_train_data(imgDir, val_split=False, grayscale=False):

    data_list = get_dataset(imgDir, train=True, grayscale=grayscale)

    if val_split:
    # divide data into test, val, and training set (20 / 16 / 64 split)
        train_ind = range(len(data_list))
        val_ind = random.sample(train_ind, int(len(train_ind)*0.2))
        train_ind = [item for item in train_ind if item not in val_ind]

        # convert from list into numpy array
        val_trials = np.array(data_list,dtype=object)[val_ind]
        train_trials = np.array(data_list,dtype=object)[train_ind]
        
        return train_trials, val_trials
    else:
        return np.array(data_list,dtype=object)


def get_test_data(imgDir, grayscale=False):
    with open(imgDir + 'labels.txt', 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
    ext = check_extension(imgDir)

    data_list = get_dataset(imgDir, grayscale=grayscale)
    return np.array(data_list,dtype=object), lines, ext


def create_cnn(input_shape, num_classes, lr=0.00001, grayscale=False):
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
    
    if grayscale:
        model = Sequential([
            layers.Rescaling(1./255, input_shape=input_shape),
            layers.Conv1D(32, 3, padding='same', activation='relu', input_shape=input_shape,
                                kernel_regularizer=l2(0.01)),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            layers.Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape,
                                kernel_regularizer=l2(0.01)),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            # layers.Dropout(0.3),
            layers.Dense(num_classes)
        ])
    else:
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
            # layers.Dropout(0.3),
            layers.Dense(num_classes)
        ])
    
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    model.summary()
    
    return model


def train_cnn(train_data, patience=10, plot=True, save_model=True, model_name='CNN_model', grayscale=False):
    input_shape = train_data[0][0].shape
    num_classes = len(np.unique(train_data[:,1]))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    model = create_cnn(input_shape, num_classes, lr=0.0001, grayscale=grayscale)

    x_train = []
    y_train = []

    for data, label in train_data:
        x_train.append(data)
        y_train.append(label)

    x_train = np.array(x_train, dtype=object).astype(np.float32)
    y_train = np.array(y_train, dtype=object).astype(np.float32)

    history = model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.2, 
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

    if save_model:
        print(f'Model saved as {model_name}.h5')
        model.save(f'{model_name}.h5')

    return model


def test_model(lines, imageDir, ext, test_data, model, visualize=False, only_false=False):
    x_test = []
    y_test = []

    for data, label in test_data:
        x_test.append(data)
        y_test.append(label)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=-1)

    if visualize:
        for i, img in enumerate(x_test):
            if only_false and y_test[i] == pred[i]:
                continue
            og_img = np.array(cv2.imread(f'{imageDir}{lines[i][0]}.{ext}'))
            cv2.imshow("Original image", og_img)
            cv2.imshow("Cropped image", img)
            print('--------------------------')
            print(f'True label: {y_test[i]}')
            print(f'Prediction: {pred[i]}')
            print(f'Result: {y_test[i] == pred[i]}')
            cv2.waitKey(0)

    test_acc = len(np.where(pred == y_test)[0])/len(y_test)
    result = confusion_matrix(y_test, pred)

    print(f'Total accuracy: {test_acc}')
    print(result)