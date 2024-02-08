from launch import LaunchDescription
from launch_ros.actions import Node, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('~/dontTouch_ws/turtlebot3/turtlebot3_bringup'),
                         'launch/camera_robot.launch.py')
        ),
        Node(
            package='lab2',
            executable='find_object',
        ),
        Node(
            package='lab2',
            executable='rotate_robot',
        )
    ])