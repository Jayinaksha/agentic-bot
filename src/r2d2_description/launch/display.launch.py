#!/usr/bin/env python3
"""Visualise the tri-star rover in RViz without starting the simulator.

Useful for sanity-checking cluster geometry and sensor frames after editing
config/robot_params.yaml.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('r2d2_description')
    xacro_file = PathJoinSubstitution([pkg, 'urdf', 'r2d2_tristar.urdf.xacro'])
    robot_description = Command(['xacro ', xacro_file, ' use_sim:=false'])

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true',
                              description='Show the joint_state_publisher slider GUI.'),
        DeclareLaunchArgument('rviz', default_value='true'),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen',
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            condition=IfCondition(LaunchConfiguration('gui')),
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            condition=IfCondition(LaunchConfiguration('rviz')),
            output='screen',
        ),
    ])
