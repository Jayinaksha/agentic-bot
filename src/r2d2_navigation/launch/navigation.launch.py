#!/usr/bin/env python3
"""Nav2 bring-up for one storey, plus the precision docking servo.

    ros2 launch r2d2_navigation navigation.launch.py

Velocity command chain, end to end:

    controller_server -> /cmd_vel_smoothed  (velocity_smoother)
                      -> /cmd_vel_nav       (collision_monitor)
                      -> climb_fsm          (gates, may substitute its own)
                      -> /cmd_vel           (tristar_controller)

Nav2 therefore never reaches the wheels directly. That is deliberate: during a
stair transition the FSM must be able to take the platform away from Nav2
entirely, and a single writer to /cmd_vel is the only way to guarantee it.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


NAV2_NODES = [
    ('nav2_controller', 'controller_server', 'controller_server'),
    ('nav2_smoother', 'smoother_server', 'smoother_server'),
    ('nav2_planner', 'planner_server', 'planner_server'),
    ('nav2_behaviors', 'behavior_server', 'behavior_server'),
    ('nav2_bt_navigator', 'bt_navigator', 'bt_navigator'),
    ('nav2_velocity_smoother', 'velocity_smoother', 'velocity_smoother'),
    ('nav2_collision_monitor', 'collision_monitor', 'collision_monitor'),
]


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('r2d2_navigation'), 'config', 'nav2_house.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time')
    sim = {'use_sim_time': use_sim_time}

    nav2 = [
        Node(package=pkg, executable=exe, name=name, output='screen',
             parameters=[params, sim],
             remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')])
        for pkg, exe, name in NAV2_NODES
    ]

    lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[sim, {
            'autostart': True,
            'node_names': [name for _, _, name in NAV2_NODES],
        }],
    )

    docking = Node(
        package='r2d2_navigation',
        executable='precise_docking',
        name='precise_docking',
        output='screen',
        parameters=[sim, {
            'standoff': LaunchConfiguration('dock_standoff'),
        }],
        condition=IfCondition(LaunchConfiguration('use_docking')),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('use_docking', default_value='true'),
        DeclareLaunchArgument(
            'dock_standoff', default_value='0.35',
            description='Metres from a surface at which a precise dock stops.'),
        *nav2,
        lifecycle,
        docking,
    ])
