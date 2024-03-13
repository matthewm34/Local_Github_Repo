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
import time
import numpy as np
import math


#define PID for moving between waypoints
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



class GoToGoal(Node):

    def __init__(self):

        self.count = 1

        self.ang_pid = PID(1.02, 0, 0.02, setpoint=0, output_limits=(-1.4, 1.4))
        self.dist_pid = PID(1.02, 0, 0.02, setpoint=0.5, output_limits=(-0.1, 0.1))

        super().__init__("go_to_goal")
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
        self.motor_publisher = self.create_publisher(Twist, '/cmd_vel', 10) 


    def odom_callback(self, data):
        curr_time = time.time()

        waypoint_loc = np.array([[1.5, 0],
                        [1.5, 1.4],
                        [0, 1.4]
                        ])

        cur_pos_x, cur_pos_y, cur_angle_rad = self.update_Odometry(data)

        print(cur_pos_x, cur_pos_y, cur_angle_rad)

        if self.count == 1:
            checkpoint_x, checkpoint_y = waypoint_loc[0,:]
        elif self.count == 2:
            checkpoint_x, checkpoint_y = waypoint_loc[1,:]
        elif self.count == 3:
            checkpoint_x, checkpoint_y = waypoint_loc[2,:]


        distance_error = np.sqrt((checkpoint_x - cur_pos_x)**2 + (checkpoint_x - cur_pos_y)**2)


        dist_output = self.dist_pid.measure(distance_error, curr_time)
        # ang_output = self.ang_pid.measure(ang, curr_time)

        # print(f"distance: {dist}\nangle: {ang}\nPIDdist: {dist_output}\nPIDang: {ang_output}")
        ang = 0
        print(f"distance: {distance_error}\nangle: {ang}\nPIDdist: {dist_output}")

        print('-------------------------------')

        ang_msg = Vector3()
        # ang_msg.z = float(ang_output)
        ang_msg.z = float(0)

        dist_msg = Vector3()
        dist_msg.x = float(dist_output)

        if checkpoint_x - cur_pos_x < .1:
            dist_msg = Vector3()
            dist_msg.x = float(0)
            dist_msg.y = float(0)
            msg_twist = Twist()
            msg_twist.linear = dist_msg
            self.motor_publisher.publish(msg_twist)
            
        else:
            # publish motor commands
            msg_twist = Twist()
            msg_twist.linear = dist_msg
            msg_twist.angular = ang_msg
            self.motor_publisher.publish(msg_twist)
            test = 0



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
    
        # self.get_logger().info('Transformed global pose is x:{}, y:{}, a:{}'.format(self.globalPos.x,self.globalPos.y,self.globalAng))
        # print('Transformed global pose is x:{}, y:{}, a:{}'.format(self.globalPos.x,self.globalPos.y,self.globalAng))
        cur_pos_x, cur_pos_y, cur_angle_rad = self.globalPos.x, self.globalPos.y, self.globalAng
        return cur_pos_x, cur_pos_y, cur_angle_rad #all of type float


  
    def range_callback(self, msg):
        test = 0


    

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

    rclpy.spin(robot_goal)    

    # while rclpy.ok():
    #     rclpy.spin_once(robot_goal)
    #     rclpy.spin_once(robot_odom)

    robot_goal.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()