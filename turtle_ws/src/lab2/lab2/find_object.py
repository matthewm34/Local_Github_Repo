import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
import cv2
from cv_bridge import CvBridge


class FindObject(Node):

    def __init__(self):
        super().__init__("find_object")

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
    
        self.img_publisher = self.create_publisher(CompressedImage, '/find_object/labeled_img', 10)
        self.coord_publisher = self.create_publisher(Point, '/find_object/coord', 10) # does image_qos_profile change anything for a Point?


    def image_label_callback(self, msg):
        br = CvBridge()
        img_raw = br.compressed_imgmsg_to_cv2(msg, "bgr8")

        img_hsv = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)
        img_hsv_blurred = cv2.GaussianBlur(img_hsv, (5, 5), 0)

        # lower and higher bound for filtering by color
        #lower_hsv = np.array([0, 175, 20])
        #higher_hsv = np.array([10, 255, 255])
        lower_hsv = np.array([74, 40, 100]) #Matthew: just changed the color thresholding from red to green
        higher_hsv = np.array([95, 255, 255])

        # find largest contour based on color, draw box around it, and print centroid
        contours = self.filter_by_color(img_hsv_blurred, lower_hsv, higher_hsv)
        final_contour = self.find_largest_contour(contours, 5000)
        cX, cY = self.create_bounding_box(img_raw, final_contour)

        ## ======= PUBLISHING ========
        # compress the labeled image
        labeled_compressed_img = br.cv2_to_compressed_imgmsg(img_raw)

        # publish labeled image at '/find_object/labeled_img' topic
        msg_img = CompressedImage()
        msg_img.data = labeled_compressed_img
        self.img_publisher.publish(msg_img)

        # publish coordinate of contour at '/find_object/coord' topic
        msg_coord = Point()
        msg_coord.x = float(cX)
        msg_coord.y = float(cY)
        msg_coord.z = float(img_raw.shape[1])       # kind of cheating here and sending the image width as z. should technically be a custom msg
        self.coord_publisher.publish(msg_coord)


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
        Returns:
            cX: X coordinate of the centroid of contour
            cY: Y coordinate of the centroid of contour
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
            # print(f"({cX}. {cY})")

        return cX, cY


def main():
    print('Running find_object...')

    rclpy.init()
    object_finder = FindObject()

    while rclpy.ok():
        rclpy.spin_once(object_finder)

    object_finder.destroy_node()
    rclpy.shutdown()

    print('node camera_debugger shutdown')


if __name__ == '__main__':
    main()
