# ME 7785 Lab 2
# Authors: Jeongwoo Cho, Matthew McKenna

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('turtlebot3_bringup'),
                            'launch/camera_robot.launch.py')
            )
        ),
        Node(
            package='lab3',
            executable='find_object',
        ),
        Node(
            package='lab3',
            executable='get_object_range',
        ),
    ])