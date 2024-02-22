# ME 7785 Lab 2
# Authors: Jeongwoo Cho, Matthew McKenna

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist, Vector3
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import time

import numpy as np


class PID():
        def __init__(self, kp, ki, kd, setpoint, output_limits):
            self.setpoint = setpoint
            self.kp = kp
            self.ki = ki
            self.kd = kd

            self.e_prev = 0

            self.time_prev = 0

            self.integral_window = []

            self.min_output = output_limits[0]
            self.max_output = output_limits[1]
        

        def measure(self, measurement, time):
            e = measurement - self.setpoint
            t_diff = time - self.time_prev
            self.time_prev = time

            derv_err = (e - self.e_prev) / t_diff

            self.integral_window.append(e*t_diff)
            if len(self.integral_window) > 20:
                self.integral_window.pop(0)

            int_err = np.sum(self.integral_window)

            # cap the error
            total = self.kp*e + self.ki*int_err + self.kd*derv_err
            total = max(min(total, self.max_output),self.min_output)

            return total


class ChaseObject(Node):

    def __init__(self):
        super().__init__("chase_object")

        self.ang_pid = PID(1, 0, 0, setpoint=0, output_limits=(-1.4, 1.4))
        self.dist_pid = PID(1, 0, 0, setpoint=0.5, output_limits=(-0.1, 0.1))

        #Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)

        self.pos_subscriber = self.create_subscription(
            Point,
            '/get_object_range/object_position',
            self.pos_callback,
            image_qos_profile
        )

        self.motor_publisher = self.create_publisher(Twist, '/cmd_vel', 10)


    def pos_callback(self, msg):
        curr_time = time.time()

        # read in the coordinate message from /find_object/coord
        dist = msg.x
        ang = msg.z

        # TODO: create a PID to convert coordinate to rotation
        # for now we're just gonna rotate a specific speed
        # print(f'{x}, {y}')

        dist_output = self.dist_pid.measure(dist, curr_time)
        ang_output = self.ang_pid.measure(ang, curr_time)

        print(f"distance: {dist}\nangle: {ang}\nPIDdist: {dist_output}\nPIDang: {ang_output}")
        print('-------------------------------')

        ang_msg = Vector3()
        ang_msg.z = float(ang_output)

        dist_msg = Vector3()
        dist_msg.x = float(dist_output)*-1


        # publish motor commands
        msg_twist = Twist()
        msg_twist.linear = dist_msg
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
    print('Running chase_robot...')

    rclpy.init()
    robot_chaser = ChaseObject()

    while rclpy.ok():
        rclpy.spin_once(robot_chaser)

    robot_chaser.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
