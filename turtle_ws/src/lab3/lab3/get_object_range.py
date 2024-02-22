# ME 7785 Lab 3
# Authors: Jeongwoo Cho, Matthew McKenna

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
        
        self.coord_subscriber = self.create_subscription(
            Point,
            '/find_object/coord',
            self.coord_callback,
            image_qos_profile
        )

        self.pos_publisher = self.create_publisher(Point, '/get_object_range/object_position', 10) # create publiher for the object distance
        # self.ang_publisher = self.create_publisher(Point, '/get_object_range/object_angular_position', 10) # create publiher for the object angle


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
        ind_window = int(np.floor(31.1*np.pi/180 / angle_increment))

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
    

    def coord_callback(self, msg):
        # read in the pixel coordinate message from /find_object/coord
        x = msg.x
        y = msg.y
        width = msg.z


        # the angle error in radians
        theta_error_rad = (width/2-x) * (62.2/width) * ((np.pi)/180)
        # Here the error is + on the RHS from robot's POV and - for LHS

        msg_pos = Point()

        if x == 999:
            theta_error_rad = 0
        msg_pos.z = float(theta_error_rad)

        ## DISTANCE CALCULATION
        # find the index closest to our error angle
        try:
            closest_ind = np.argmin(np.abs(self.lidar_angles - theta_error_rad))
            distance = self.lidar_data[closest_ind]
            
            if x == 999:
                distance = 0.5
                
            msg_pos.x = float(distance)
            self.pos_publisher.publish(msg_pos)
        except:
            pass


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