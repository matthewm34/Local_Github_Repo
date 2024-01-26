# Spring 2024 ME 7785 Lab 1
# authors: Jeongwoo Cho, Matthew McKenna

import numpy as np
import cv2


def filter_by_color(img_hsv, lower_bound, upper_bound):
    """
    Returns array of contours that fall within a HSV bound given an image. 

    Args:
        img_hsv (_type_): input image in HSV format
        lower_bound (_type_): lower HSV bound
        upper_bound (_type_): upper HSV bound

    Returns:
        _type_: array of contours
    """

    # apply the color filter and create a mask
    color_mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
    kernel = np.ones((10,10),np.uint8)
    mask_open = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

    # initial contours found based on color
    contours, _ = cv2.findContours(mask_open, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def find_largest_contour(contours, min_contour_size=5000):
    """
    Given list of contours, finds the contour that is greater than a minimum size, and is the largest.

    Args:
        contours (_type_): list of contours
        min_contour_size (int, optional): minimum size of contour. Defaults to 5000.

    Returns:
        _type_: _description_
    """
    # find largest contour that meets minimum contour size
    max_contour_size = min_contour_size
    final_contour = None

    for contour in contours:
        if cv2.contourArea(contour) > max_contour_size:
            max_contour_size = cv2.contourArea(contour)
            final_contour = contour

    return final_contour


def create_bounding_box(img, final_contour):
    """
    Draws bounding box around contour on an image and also writes/prints the centroid of the contour.

    Args:
        img (_type_): image in RGB format
        final_contour (_type_): contour that is to be drawn
    """
    # initialize empty mask and populate with largest contour found
    final_mask = np.zeros(img.shape[:2], dtype=img.dtype)
    if final_contour is not None:
        cv2.drawContours(final_mask, [final_contour], 0, (255), -1)

        # draw bounding box around largest contour
        x,y,w,h = cv2.boundingRect(final_contour)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        # find centroid
        M = cv2.moments(final_contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # draw centroid on image and print
        cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(img, f"({cX}. {cY})", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        print(f"({cX}. {cY})")


cap = cv2.VideoCapture(0)

while(True):
    ret, img = cap.read()

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv_blurred = cv2.GaussianBlur(img_hsv, (5, 5), 0)

    # lower and higher bound for filtering by color
    lower_hsv = np.array([0, 175, 20])
    higher_hsv = np.array([10, 255, 255])

    # find largest contour based on color, draw box around it, and print centroid
    contours = filter_by_color(img_hsv_blurred, lower_hsv, higher_hsv)
    final_contour = find_largest_contour(contours, 5000)
    create_bounding_box(img, final_contour)
    
    # Display the resulting frame
    # cv2.imshow('mask', final_mask)
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

