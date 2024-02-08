import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
import cv2
from cv_bridge import CvBridge


class CameraDebugger(Node):

    def __init__(self):
        super().__init__("camera_debugger")

        #Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

        self._image_subscriber = self.create_subscription(
            Image,
            '/find_object/labeled_img',
            self._image_callback,
            image_qos_profile
        )
    
    def _image_callback(self, msg):
        br = CvBridge()
        self._img_BGR = br.compressed_imgmsg_to_cv2(msg)

        cv2.imshow('debug image', self._img_BGR)
        self._user_input = cv2.waitKey(1)


def main():
    print('Running node camera_debugger...')

    rclpy.init()
    camera_debugger = CameraDebugger()

    while rclpy.ok():
        rclpy.spin_once(camera_debugger)
        if camera_debugger._user_input == ord('q'):
            cv2.destroyAllWindows()
            break

    camera_debugger.destroy_node()
    rclpy.shutdown()

    print('node camera_debugger shutdown')


if __name__ == '__main__':
    main()
