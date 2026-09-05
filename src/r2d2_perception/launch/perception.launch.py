#!/usr/bin/env python3
"""VLA perception and the memory bridge.

    ros2 launch r2d2_perception perception.launch.py

Started separately from the navigation stack on purpose. Both nodes here depend
on things outside the robot - a cloud VLA endpoint and a NATS server - and a
robot that has lost its network should still be able to drive. Nothing in
locomotion, localisation or navigation subscribes to anything this launch file
starts.

Environment expected by the VLA node:

    R2D2_VLA_BASE_URL   default https://integrate.api.nvidia.com/v1
    R2D2_VLA_MODEL      default nvidia/cosmos-reason2-8b
    R2D2_NVIDIA_API_KEY required for the hosted endpoint, not for self-hosted
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    sim = {'use_sim_time': use_sim_time}

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('use_vla', default_value='true'),
        DeclareLaunchArgument('use_memory', default_value='true'),
        DeclareLaunchArgument(
            'vla_interval', default_value='6.0',
            description='Minimum seconds between automatic VLA frames. Raise '
                        'it to cut inference cost, lower it for denser mapping.'),
        DeclareLaunchArgument('nats_url', default_value='nats://127.0.0.1:4222'),

        Node(
            package='r2d2_perception',
            executable='vla_node',
            name='vla_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_vla')),
            parameters=[sim, {
                'min_interval': LaunchConfiguration('vla_interval'),
            }],
        ),
        Node(
            package='r2d2_memory',
            executable='memory_node',
            name='memory_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_memory')),
            parameters=[sim, {
                'nats_url': LaunchConfiguration('nats_url'),
            }],
        ),
    ])
