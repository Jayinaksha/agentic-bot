#!/usr/bin/env python3
"""The whole robot, in one command.

    ros2 launch r2d2_navigation house_stack.launch.py
    ros2 launch r2d2_navigation house_stack.launch.py sim:=false   # real robot
    ros2 launch r2d2_navigation house_stack.launch.py mode:=slam   # build a map

Brings up, in dependency order: simulator (optional), locomotion, localisation,
navigation. Perception, memory and the MCP agent are started separately - they
have cloud dependencies and their own failure modes, and a robot that cannot
reach the internet should still be able to drive.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _include(package, launch_file, arguments, condition=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare(package), 'launch', launch_file])),
        launch_arguments=arguments.items(),
        condition=condition,
    )


def generate_launch_description():
    sim = LaunchConfiguration('sim')
    use_sim_time = LaunchConfiguration('use_sim_time')
    mode = LaunchConfiguration('mode')
    map_dir = LaunchConfiguration('map_directory')

    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'mode', default_value='amcl',
            description='"slam" to build a map, "amcl" to navigate a saved one.'),
        DeclareLaunchArgument('map_directory', default_value='~/r2d2_maps'),
        DeclareLaunchArgument('floor', default_value='0'),
        DeclareLaunchArgument('autonomous_climb', default_value='false'),

        _include('r2d2_sim', 'sim.launch.py',
                 {'spawn_floor': LaunchConfiguration('floor')},
                 condition=IfCondition(sim)),

        _include('r2d2_locomotion', 'locomotion.launch.py', {
            'use_sim_time': use_sim_time,
            'autonomous_climb': LaunchConfiguration('autonomous_climb'),
        }),

        _include('r2d2_localization', 'localization.launch.py', {
            'use_sim_time': use_sim_time,
            'mode': mode,
            'map_directory': map_dir,
            'floor': LaunchConfiguration('floor'),
        }),

        _include('r2d2_navigation', 'navigation.launch.py', {
            'use_sim_time': use_sim_time,
        }),
    ])
