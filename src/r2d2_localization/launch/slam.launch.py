#!/usr/bin/env python3
"""Map one floor and save it where floor_manager expects to find it.

    ros2 launch r2d2_localization slam.launch.py floor:=0
    # drive the robot around the floor, then:
    ros2 run nav2_map_server map_saver_cli -f ~/r2d2_maps/house_f0

Map each floor separately. A two-storey house cannot be one occupancy grid -
the bedroom sits directly above the kitchen and a single grid fuses them into
nonsense - so the stack keeps one map per floor and joins them with the
explicit transition nodes in floor_manager.
"""

import os

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            LogInfo)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    floor = LaunchConfiguration('floor')
    map_dir = LaunchConfiguration('map_directory')

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('r2d2_localization'), 'launch', 'localization.launch.py'])),
        launch_arguments={
            'mode': 'slam',
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'map_directory': map_dir,
            'floor': floor,
        }.items(),
    )

    locomotion = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('r2d2_locomotion'), 'launch', 'locomotion.launch.py'])),
        launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('floor', default_value='0'),
        DeclareLaunchArgument('map_directory',
                              default_value=os.path.expanduser('~/r2d2_maps')),
        locomotion,
        localization,
        LogInfo(msg=['Mapping floor ', floor, '. When the floor is covered, save with:\n',
                     '  ros2 run nav2_map_server map_saver_cli -f ',
                     map_dir, '/house_f', floor]),
    ])
