# ME 7785 Lab 4
# Authors: Jeongwoo Cho, Matthew McKenna
'''
get_object_range: This node should detect the ranges and orientation of obstacles. 
It should subscribe to the scan node and publish the vector pointing from the robot to the nearest point on
the object.
'''

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Twist, Vector3, Quaternion
from sensor_msgs.msg import LaserScan #make sure to import laserscan like this
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
# from print_fixed_odometry import update_Odometry

import numpy as np
import math


#Sean's Code to initialize the starting frame of the robot and update the odometry data
class print_transformed_odom(Node):
    def __init__(self):
        super().__init__('print_fixed_odom')
        # State (for the update_Odometry code)
        self.Init = True
        self.Init_pos = Point()
        self.Init_pos.x = 0.0
        self.Init_pos.y = 0.0
        self.Init_ang = 0.0
        self.globalPos = Point()

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            1)
        self.odom_sub  # prevent unused variable warning

    def odom_callback(self, data):
        self.update_Odometry(data)

    def update_Odometry(self,Odom):
        position = Odom.pose.pose.position
        
        #Orientation uses the quaternion aprametrization.
        #To get the angular position along the z-axis, the following equation is required.
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

        if self.Init:
            #The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
            self.Init_pos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Init_pos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Init_pos.z = position.z
        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        

        #We subtract the initial values
        self.globalPos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos.x
        self.globalPos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos.y
        self.globalAng = orientation - self.Init_ang
    
        self.get_logger().info('Transformed global pose is x:{}, y:{}, a:{}'.format(self.globalPos.x,self.globalPos.y,self.globalAng))
        print('Transformed global pose is x:{}, y:{}, a:{}'.format(self.globalPos.x,self.globalPos.y,self.globalAng))
    





class GoToGoal(Node):

    def __init__(self):
        super().__init__("go_to_goal")

        # self.lidar_data = []
        # self.lidar_angles = []
        # self.lidar_deg_inc = None

        #Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(
		    reliability=QoSReliabilityPolicy.BEST_EFFORT,
		    history=QoSHistoryPolicy.KEEP_LAST,
		    durability=QoSDurabilityPolicy.VOLATILE,
		    depth=1
		)
        
        #Subscriber to object_range from get_object_range
        self.range_subscriber = self.create_subscription(
            Point,
            '/object_range/range',
            self.range_callback,
            image_qos_profile
        )
        
       
        # create publiher for the Twist motor command
        self.vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10) 



    def range_callback(self, msg):
        print(str(msg))

        # read in the coordinate message from /find_object/coord
        dist = msg.x
        ang = msg.z

        waypoint_loc = np.array([[1.5, 0],
                                [1.5, 1.4],
                                [0, 1.4]
                                ])

        msg_pos = Point()

        
    

    # def coord_callback(self, msg):

        '''
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
        '''



# def main(args=None):
#     rclpy.init(args=args)
#     print_odom = print_transformed_odom()
#     rclpy.spin(print_odom)
#     print_odom.destroy_node()
#     rclpy.shutdown()


def main():
    print('Running go_to_goal..')

    rclpy.init()
    robot_goal= GoToGoal()
    robot_odom= print_transformed_odom()

    rclpy.spin(robot_goal)
    rclpy.spin(robot_odom)
    

    # while rclpy.ok():
    #     rclpy.spin_once(robot_goal)
    #     rclpy.spin_once(robot_odom)

    robot_goal.destroy_node()
    robot_odom.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()