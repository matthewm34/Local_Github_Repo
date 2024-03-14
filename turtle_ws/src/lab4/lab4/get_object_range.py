# ME 7785 Lab 4
# Authors: Jeongwoo Cho, Matthew McKenna
'''
get_object_range: This node should detect the ranges and orientation of obstacles. 
It should subscribe to the scan node and publish the vector pointing from the robot to the nearest point on
the object.
'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist, Vector3
from sensor_msgs.msg import LaserScan #make sure to import laserscan like this
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np


class GetObjectRange(Node):

    def __init__(self):
        super().__init__("get_object_range")

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

        #Publisher: vector pointing from  robot to nearest point on object
        self.pos_publisher = self.create_publisher(Point, '/object_range/range', 10) # create publiher for the object distance

    def scan_callback(self, msg):
        # read in the coordinate message from /scan'
        # Notes: the lidar scans CCW starting from the robots heading
        lidar_range_raw = msg.ranges #get LIDAR values
        lidar_range_min = msg.range_min
        lidar_range_max = msg.range_max
        angle_increment = msg.angle_increment
        angle_min = msg.angle_min
        angle_max = msg.angle_max

        # print("\n\n\LIDAR RANGE\n\n\n\n" + str(lidar_range_raw))
        lidar_range_data = np.array(lidar_range_raw)
        lidar_angles = np.arange(angle_min, angle_max, angle_increment)

        self.lidar_deg_inc = angle_increment
        ind_window = int(np.floor(90*np.pi/180 / angle_increment))

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

        ## filter lidar values
        masked_lidar[masked_lidar > 0.15] = 3

        if min(masked_lidar) == 3:
            ind_max = 100
        else:  
            ind_max = np.where(masked_lidar == min(masked_lidar))[0][0]

        msg_pos = Point()
        msg_pos.x = float(ind_max)
        msg_pos.y = float(len(masked_lidar))

        self.pos_publisher.publish(msg_pos)
    

def main():
    print('Running get_object_range...')

    rclpy.init()
    robot_range= GetObjectRange()

    while rclpy.ok():
        rclpy.spin_once(robot_range)

    robot_range.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()