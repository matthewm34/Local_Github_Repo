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
        

        #Subscriber to LIDAR scan
        self.coord_subscriber = self.create_subscription(
            Point,
            '/find_object/coord',
            self.coord_callback,
            image_qos_profile
        )

        self.dist_publisher = self.create_publisher(Point, '/get_object_range/object_distance', 10) # create publiher for the object distance
        self.ang_publisher = self.create_publisher(Point, '/get_object_range/object_angular_position', 10) # create publiher for the object angle


    def scan_callback(self, msg):
        # read in the coordinate message from /scan'
        # Notes: the lidar scans CCW starting from the robots heading
        lidar_range_raw = msg.ranges #get LIDAR values

        # print("\n\n\LIDAR RANGE\n\n\n\n" + str(lidar_range_raw))
        lidar_range_data = np.array(lidar_range_raw)

        lidar_range_min = msg.range_min
        lidar_range_max= msg.range_max
        
        angle_increment = msg.angle_increment  
        angle_min = msg.angle_min  
        angle_max = msg.angle_max  
        lidar_angle_data_rad = np.arange(angle_min, angle_max, angle_increment)

        # get the lhs and rhs robot lidar data for angle
        lidar_lhs_robot_mask = lidar_angle_data_rad < 31.1*np.pi/180
        angle_lhs_robot = lidar_angle_data_rad[lidar_lhs_robot_mask]
        angle_lhs_robot = np.flip(angle_lhs_robot)

        lidar_rhs_robot_mask = lidar_angle_data_rad > (360-31.1)*np.pi/180
        angle_rhs_robot = lidar_angle_data_rad[lidar_rhs_robot_mask]
        angle_rhs_robot = np.flip(angle_rhs_robot)

        angle_robot_rad = np.append(angle_lhs_robot, angle_rhs_robot)

        # get the lhs and rhs robot lidar data for distance
        dist_lhs_robot = lidar_range_data[lidar_lhs_robot_mask]
        dist_lhs_robot = np.flip(dist_lhs_robot)

        dist_rhs_robot = lidar_range_data[lidar_rhs_robot_mask]
        dist_rhs_robot = np.flip(dist_lhs_robot)

        dist_robot = np.append(dist_lhs_robot, dist_rhs_robot)

        #filter out values NAN; also filter above and below the designated LIDAR distances thresholds
        lidar_mask = np.logical_and(lidar_range_data > lidar_range_min, lidar_range_data < lidar_range_max)
       
        lidar_range_data_masked = lidar_range_data[lidar_mask] 
        lidar_radians_vec_masked = lidar_angle_data_rad[lidar_mask] 

        lidar_angle_data_rad = lidar_range_data_masked
        lidar_range_data = lidar_radians_vec_masked


        # print(len(angle_robot_rad), len(dist_robot))
        # print(len(angle_lhs_robot), len(angle_rhs_robot), len(dist_lhs_robot), len(dist_rhs_robot))
        # print(any(np.isnan(angle_lhs_robot)), any(np.isnan(angle_rhs_robot)), any(np.isnan(dist_lhs_robot)), any(np.isnan(dist_rhs_robot)))
    


    def coord_callback(self, msg):
        # read in the pixel coordinate message from /find_object/coord
        x = msg.x
        y = msg.y
        width = msg.z

        #the angle error in radians
        theta_error_rad = (160-x) * (62.2/320) * ((np.pi)/180)
        # Here the error is + on the RHS from robot's POV and - for LHS

        # print(theta_error_rad, (160-x) * (62.2/320))




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