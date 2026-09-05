#!/usr/bin/env python3
"""Drive controller, terrain monitor and climb FSM.

    ros2 launch r2d2_locomotion locomotion.launch.py

Command topology, which matters because two nodes publish velocity:

    Nav2 / teleop ──> /cmd_vel_nav ──> climb_fsm ──> /cmd_vel ──> tristar_controller

climb_fsm is a gate, not a bypass. In IDLE it republishes /cmd_vel_nav verbatim;
in every other state it substitutes its own command and takes the transmission
into tumbling mode. Nothing else may publish /cmd_vel or /locomotion/mode.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('r2d2_locomotion'), 'config', 'locomotion.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time')

    common = [config, {'use_sim_time': use_sim_time}]

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'autonomous_climb', default_value='false',
            description='Let the FSM enter a climb on its own ToF evidence, '
                        'instead of only on an explicit navigation request.'),

        Node(package='r2d2_locomotion', executable='tristar_controller',
             name='tristar_controller', parameters=common, output='screen'),
        Node(package='r2d2_locomotion', executable='terrain_monitor',
             name='terrain_monitor', parameters=common, output='screen'),
        Node(package='r2d2_locomotion', executable='climb_fsm',
             name='climb_fsm', output='screen',
             parameters=common + [{
                 'autonomous_entry': LaunchConfiguration('autonomous_climb'),
             }]),
    ])
