#!/usr/bin/env python3
"""Bring up Gazebo Sim with the two-storey house and the tri-star rover.

    ros2 launch r2d2_sim sim.launch.py
    ros2 launch r2d2_sim sim.launch.py spawn_floor:=1   # start upstairs
    ros2 launch r2d2_sim sim.launch.py headless:=true   # no GUI, for CI

This launch file owns simulation only: world, robot, bridge and controllers.
Localisation, navigation, perception and the agent are separate launch files so
each layer can be restarted without tearing the simulator down.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, RegisterEventHandler)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, LaunchConfiguration, PathJoinSubstitution,
                                  PythonExpression)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim_share = get_package_share_directory('r2d2_sim')
    desc_share = get_package_share_directory('r2d2_description')

    world_file = os.path.join(sim_share, 'worlds', 'house_two_floor.sdf')
    bridge_file = os.path.join(sim_share, 'config', 'house_bridge.yaml')
    xacro_file = os.path.join(desc_share, 'urdf', 'r2d2_tristar.urdf.xacro')

    headless = LaunchConfiguration('headless')
    spawn_floor = LaunchConfiguration('spawn_floor')
    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')

    robot_description = Command(['xacro ', xacro_file, ' use_sim:=true'])

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])),
        launch_arguments={
            # -r starts unpaused, -v3 keeps the log readable.
            'gz_args': ['-r -v3 ', world_file],
            'on_exit_shutdown': 'true',
        }.items(),
        condition=UnlessCondition(headless),
    )

    gz_sim_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])),
        launch_arguments={
            'gz_args': ['-r -s -v3 ', world_file],
            'on_exit_shutdown': 'true',
        }.items(),
        condition=IfCondition(headless),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
    )

    # Spawn height clears the sub-wheels so the platform drops onto its clusters
    # and settles, rather than starting interpenetrating the floor. Upstairs adds
    # the floor-to-floor rise (stairs.riser * stairs.steps = 1.80 m).
    spawn_z = PythonExpression(['"0.20" if ', spawn_floor, ' == 0 else "2.00"'])

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'r2d2_tristar',
            '-x', spawn_x,
            '-y', spawn_y,
            '-z', spawn_z,
            '-Y', '1.5708',
        ],
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        parameters=[{
            'config_file': bridge_file,
            'use_sim_time': True,
        }],
    )

    # ros2_control spawners must wait for the controller manager that
    # gz_ros2_control starts inside Gazebo, hence the chain off `spawn`.
    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    drive_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['tristar_velocity_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('spawn_floor', default_value='0',
                              description='0 = ground floor, 1 = upper floor.'),
        DeclareLaunchArgument('spawn_x', default_value='6.0'),
        DeclareLaunchArgument('spawn_y', default_value='1.0'),

        gz_sim,
        gz_sim_headless,
        robot_state_publisher,
        bridge,
        spawn,

        RegisterEventHandler(OnProcessExit(
            target_action=spawn,
            on_exit=[joint_state_broadcaster],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster,
            on_exit=[drive_controller],
        )),
    ])
