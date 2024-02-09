# ME 7785 Lab 2
# Authors: Jeongwoo Cho, Matthew McKenna

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist, Vector3
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np


class RobotRotate(Node):

    def __init__(self):
        super().__init__("rotate_robot")

        #Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

        self.coord_subscriber = self.create_subscription(
            Point,
            '/find_object/coord',
            self.coord_callback,
            image_qos_profile
        )

        self.motor_publisher = self.create_publisher(Twist, '/cmd_vel', 10)


    def coord_callback(self, msg):
        # read in the coordinate message from /find_object/coord
        x = msg.x
        y = msg.y
        width = msg.z

        # TODO: create a PID to convert coordinate to rotation
        # for now we're just gonna rotate a specific speed
        # print(f'{x}, {y}')
        angular_vel = self.get_rotation(x,width)
        ang_msg = Vector3()
        ang_msg.z = float(angular_vel)

        # publish motor commands
        msg_twist = Twist()
        msg_twist.angular = ang_msg
        self.motor_publisher.publish(msg_twist)


    def get_rotation(self, x, width):
        # object is on the right
        if x - width/2 > 20:
            print('object on right')
            return -0.5
        # object is on the left
        elif x - width/2 < -20:
            print('object on left')
            return +0.5
        else: 
            return 0 

def main():
    print('Running rotate_robot...')

    rclpy.init()
    robot_rotator = RobotRotate()

    while rclpy.ok():
        rclpy.spin_once(robot_rotator)

    robot_rotator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
