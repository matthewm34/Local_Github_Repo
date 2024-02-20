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
        
        self.dist_publisher = self.create_publisher(Point, '/get_object_range/object_distance', 10) # create publiher for the object distance
        self.ang_publisher = self.create_publisher(Point, '/get_object_range/object_angular_position', 10) # create publiher for the object angle



    def scan_callback(self, msg):
        # read in the coordinate message from /scan'
        print('Running Callback...')
        print(msg)

    #     # TODO: create a PID to convert coordinate to rotation
    #     # for now we're just gonna rotate a specific speed
    #     # print(f'{x}, {y}')
    #     angular_vel = self.get_rotation(x,width)
    #     ang_msg = Vector3()
    #     ang_msg.z = float(angular_vel)

    #     # publish motor commands
    #     msg_twist = Twist()
    #     msg_twist.angular = ang_msg
    #     self.motor_publisher.publish(msg_twist)


    # def get_rotation(self, x, width):
    #     # object is on the right
    #     if x - width/2 > 20:
    #         print('object on right')
    #         return -0.5
    #     # object is on the left
    #     elif x - width/2 < -20:
    #         print('object on left')
    #         return +0.5
    #     else: 
    #         return 0 

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