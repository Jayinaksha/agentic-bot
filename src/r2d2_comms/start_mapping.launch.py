import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Get the path to your parameter file
    config_file = os.path.join(
        get_package_share_directory('r2d2_comms'),
        'config',
        'mapper_config.yaml'
    )

    return LaunchDescription([
        # 1. Start the RPLIDAR driver
        Node(
            package='rplidar_ros',
            executable='rplidar_node',
            name='rplidar_node',
            parameters=[{'frame_id': 'laser'}]
        ),

        # 2. Start the Static Transform Publisher (base_link -> laser)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0.1', '0', '0.2', '0', '0', '0', 'base_link', 'laser'],
        ),

        # 3. Start the Laser Scan Matcher (provides odom -> base_link)
        Node(
            package='laser_scan_matcher',
            executable='laser_scan_matcher_node',
            name='laser_scan_matcher',
            parameters=[{
                'publish_tf': True,
                'use_sim_time': False,
                'base_frame': 'base_link',
                'odom_frame': 'odom',
            }]
        ),

        # 4. Start SLAM Toolbox with the correct configuration
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            parameters=[config_file]
        ),
    ])
