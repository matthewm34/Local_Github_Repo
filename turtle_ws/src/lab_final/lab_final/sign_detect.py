import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Point
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
import cv2
from cv_bridge import CvBridge

from tensorflow.keras.models import load_model

from image_utils import *



class DetectSign(Node):
    def __init__(self):
        super().__init__("detect_sign")

        self._image_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_label_callback)
        
        self.sign_publisher = self.create_publisher(Point, '/sign_detect', 10) # does image_qos_profile change anything for a Point?
        self.model = load_model('CNN_model.h5')

    
    def image_label_callback(self, msg):
        br = CvBridge()
        img_raw = br.compressed_imgmsg_to_cv2(msg, "bgr8")

        img = np.array(filter_img(img_raw))
        pred = self.model.predict(img)

        ## ======= PUBLISHING ========
        msg_coord = Point()
        msg_coord.x = float(pred)
        self.coord_publisher.publish(msg_coord)
