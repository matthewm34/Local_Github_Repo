
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist, Vector3
from sensor_msgs.msg import LaserScan #make sure to import laserscan like this
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np


class LidarDetect(Node):

    def __init__(self):
        super().__init__("lidar_detect")

        self.lidar_data = []
        self.lidar_angles = []
        self.lidar_deg_inc = None

        #Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)
        
        #Subscriber to LIDAR scan
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            image_qos_profile
        )

        self.dist_publisher = self.create_publisher(Point, '/lidar_dist', 10) # create publiher for the object distance


    def scan_callback(self, msg):
        # read in the coordinate message from /scan'
        # Notes: the lidar scans CCW starting from the robots heading
        lidar_range_raw = msg.ranges #get LIDAR values
        angle_increment = msg.angle_increment
        angle_min = msg.angle_min  
        angle_max = msg.angle_max  

        # print("\n\n\LIDAR RANGE\n\n\n\n" + str(lidar_range_raw))
        lidar_range_data = np.array(lidar_range_raw)
        lidar_angles = np.arange(angle_min, angle_max, angle_increment) * 180/np.pi


        angle_increment_deg = angle_increment * 180/np.pi
        self.lidar_deg_inc = angle_increment_deg
        ind_window = int(np.floor(31.1 / angle_increment_deg))

        # actual LIDAR data segmented
        lidar_left = lidar_range_data[ind_window:0:-1]
        lidar_right = lidar_range_data[:-ind_window:-1]

        # the angles associated with the above LIDAR data 
        lidar_angles_left = lidar_angles[ind_window:0:-1]
        lidar_angles_right = lidar_angles[:-ind_window:-1]-360

        # combine the segmented out LIDAR data together
        masked_lidar = np.concatenate((lidar_left,[lidar_range_data[0]],lidar_right),axis=0)
        masked_lidar_angles = np.concatenate((lidar_angles_left,[lidar_angles[0]],lidar_angles_right),axis=0)

        self.lidar_data = masked_lidar
        self.lidar_angles = masked_lidar_angles

        ## DISTANCE CALCULATION
        # find the index closest to our error angle
        closest_dist = np.min(self.lidar_data)

        msg_dist = Point()
        msg_dist.z = float(closest_dist)
        self.dist_publisher.publish(msg_dist)


def main():
    print('Running lidar_detect...')

    rclpy.init()
    robot_range= LidarDetect()

    while rclpy.ok():
        rclpy.spin_once(robot_range)

    robot_range.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()