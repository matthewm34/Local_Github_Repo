from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lab2',
            executable='find_object',
        ),
        Node(
            package='lab2',
            executable='rotate_tobot',
        )
    ])