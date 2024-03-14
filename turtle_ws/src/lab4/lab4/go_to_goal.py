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

        self.count = 1 # initialize the goal counter as 1st cehckpoint
        self.first_iteration = True 

        self.GoGoal = True # flag for converting between go to goal and avoid obstacle
        self.turnDir = None # direction to turn when in avoid obstacle state

        self.ang_pid = PID(1, 0, 0, setpoint=0, output_limits=(-1.4, 1.4))
        self.dist_pid = PID(1, 0, 0, setpoint=0, output_limits=(-0.2, 0.2))

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

        curr_time = time.time() # get current time

        waypoint_global_loc = np.array([[1.5, 0, 1], # waypoint locations constant
                                    [1.5, 1.4,  1],
                                    [ 0, 1.4, 1]
                                    ])        
        
        cur_pos_x, cur_pos_y, cur_angle_rad = self.update_Odometry(data) # update the odometry data
        print(cur_pos_x, cur_pos_y, cur_angle_rad)
        
        # wrap this? 
        transformation_matrix = np.matrix([[np.cos(cur_angle_rad), -np.sin(cur_angle_rad), cur_pos_x],
                                        [np.sin(cur_angle_rad), np.cos(cur_angle_rad), cur_pos_y],
                                        [0,                             0,                  1  ]])


        print('-------------------------------')
        print(f'Count: {self.count}')
        print(f'Go to goal check: {self.GoGoal}')

        if self.count == 1:
            local_checkpoint_vec = transformation_matrix.I @ waypoint_global_loc[0,:].reshape(-1,1)
            local_checkpoint_dist_x = local_checkpoint_vec[0]
            local_checkpoint_dist_y = local_checkpoint_vec[1]
            desired_angle = np.pi/2 #straight ahead
        elif self.count == 2:
            local_checkpoint_vec = transformation_matrix.I @ waypoint_global_loc[1,:].reshape(-1,1)
            local_checkpoint_dist_x = local_checkpoint_vec[0]
            local_checkpoint_dist_y = local_checkpoint_vec[1]
            desired_angle = np.pi#90 degrees to the left
        elif self.count == 3:
            local_checkpoint_vec = transformation_matrix.I @ waypoint_global_loc[2,:].reshape(-1,1)
            local_checkpoint_dist_x = local_checkpoint_vec[0]
            local_checkpoint_dist_y = local_checkpoint_vec[1]
            desired_angle = 3*np.pi/2 #another 90 degrees to the left
        else:    
            test = 0

            #terminate code -> made it to the end

        
        if not self.GoGoal:

            print("in obstacle avoidance state")
            if self.turnDir is None:
                # move forward in twist
                linSpeed = 0.1
                turnSpeed = 0

            else:
                linSpeed = 0
                # angular velocity 
                if self.turnDir == 'CW':
                    turnSpeed = -0.7
                else:
                    turnSpeed = 0.7

            
            dist_msg = Vector3()
            dist_msg.x, dist_msg.y  = float(linSpeed), float(0) # make sure linear velocity is zero
            ang_msg = Vector3()
            ang_msg.z = float(turnSpeed)
            msg_twist = Twist()
            msg_twist.linear = dist_msg
            msg_twist.angular = ang_msg
            self.motor_publisher.publish(msg_twist)

        else:
        
            distance_error = local_checkpoint_dist_x #determine distance between robot and checkpoint

            local_goal_direction = np.arctan2(local_checkpoint_dist_y,local_checkpoint_dist_x)
            theta_error = local_goal_direction - cur_angle_rad #determine distance between robot and checkpoint
            theta_error = local_goal_direction

            #TODO do i have to wrap theta error???

            dist_output = self.dist_pid.measure(distance_error, curr_time) #PID for distance, approaching checkpoint
            ang_output = self.ang_pid.measure(theta_error, curr_time) #PID for rotating 90 degrees, once reached checkpoint
            
            print(f"distance: {distance_error}\nangle: {theta_error}\nPIDdist: {dist_output}")
            # go to goal state ----------------------------------------------------------
            # if  theta_error > np.pi/180 * 5: # if the heading of the robot is greater than 5 degrees away from the goal direction
            if theta_error > np.pi/180 * 5:
                # rotate robot 90 degrees ccw since its at checkpoint
                dist_msg = Vector3()
                dist_msg.x, dist_msg.y  = float(0), float(0) # make sure linear velocity is zero
                ang_msg = Vector3()
                ang_msg.z = float(ang_output)# positive for turning ccw
                msg_twist = Twist()
                msg_twist.angular = ang_msg
                self.motor_publisher.publish(msg_twist)

                # elif local_goal_direction < -np.pi/180 * 5:
            elif theta_error < -np.pi/180 * 5:
                dist_msg = Vector3()
                dist_msg.x, dist_msg.y  = float(0), float(0) # make sure linear velocity is zero
                ang_msg = Vector3()
                ang_msg.z = float(ang_output) # negative for turning ccw
                msg_twist = Twist()
                msg_twist.angular = ang_msg
                self.motor_publisher.publish(msg_twist)  

                #reached the checkpoint
            elif distance_error < .025: # made it to checkpoint -> set vel = 0
                dist_msg = Vector3()
                dist_msg.x = float(0)
                dist_msg.y = float(0)
                msg_twist = Twist()
                msg_twist.linear = dist_msg
                self.motor_publisher.publish(msg_twist)

                # if self.first_iteration == True: # wait at checkpoint
                #stop at the checkpoint for 10 seconds
                print('Waiting at Checkpoint 5 seconds')
                time.sleep(5)
                # self.first_iteration = False
                self.count = self.count + 1 # set count to move toward next checkpoint

            else: #if not at checkpoint -> move forward 
                # publish motor commands
                dist_msg = Vector3()
                dist_msg.x = float(dist_output)

                msg_twist = Twist()
                msg_twist.linear = dist_msg
                self.motor_publisher.publish(msg_twist)
                self.first_iteration = True


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
        center = msg.x # index of the range vector at which the minimum distane of the "obstacle" is at
        range = msg.y # length of the lidar vector from -90 to 90 degrees

        if center == 900:
            # set state to go to goal 
            self.GoGoal = True

        else:
            self.GoGoal = False

            if center < range/2:    # object is on the left side
                if center < 10:
                    self.turnDir = None
                else:
                    # turn clockwise until center is below 10
                    self.turnDir = 'CW'
            else:
                if center > range-10:
                    # move forward
                    self.turnDir = None
                else:
                    # turn counter clockwise until center is below 10
                    self.turnDir = 'CCW'


    

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