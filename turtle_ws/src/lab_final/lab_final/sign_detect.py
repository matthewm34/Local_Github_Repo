import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Point
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
import cv2
from cv_bridge import CvBridge

from tensorflow.keras.models import load_model

# from image_utils import *


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
    if cv2.contourArea(largest_contour) < 260:
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
        no_change_flag = True
    else:
        cropped_img = cv2.resize(cropped_img, (64,64))
        no_change_flag = False

    # cv2.imshow('original',img)
    # cv2.imshow('cropped',cropped_img)
    # cv2.waitKey(0)

    return cropped_img, no_change_flag


class DetectSign(Node):
    def __init__(self):
        super().__init__("detect_sign")

        #Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

        self._image_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_label_callback,
            image_qos_profile
            )
        
        self.sign_publisher = self.create_publisher(Point, '/sign_detect', 10) # does image_qos_profile change anything for a Point?
        self.model = load_model('/home/jwcho/Turtlebot_Labs/turtle_ws/src/lab_final/lab_final/CNN_model.h5')

    
    def image_label_callback(self, msg):
        br = CvBridge()
        img_raw = br.compressed_imgmsg_to_cv2(msg, "bgr8")

        cv2.imshow('image',img_raw)
        cv2.waitKey(1)


        img = np.array(filter_img(img_raw))
        pred = self.model.predict(img)
        print(pred)

        ## ======= PUBLISHING ========
        msg_coord = Point()
        msg_coord.x = float(pred)
        self.coord_publisher.publish(msg_coord)


def main():
    print('Running sign_detect...')

    rclpy.init()
    sign_detect= DetectSign()

    while rclpy.ok():
        rclpy.spin_once(sign_detect)

    sign_detect.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()