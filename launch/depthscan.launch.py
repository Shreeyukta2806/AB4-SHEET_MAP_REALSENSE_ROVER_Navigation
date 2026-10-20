"""
Depth→LaserScan Launch — CEAM GoKart Project, MAHE Bengaluru

Converts the RealSense depth image to a LaserScan message so Nav2 costmaps
and the collision_monitor can use it as an obstacle source.

Requires: depthimage_to_laserscan (apt install ros-jazzy-depthimage-to-laserscan)

QoS overrides are required because the RealSense node publishes with
BEST_EFFORT reliability, but depthimage_to_laserscan defaults to RELIABLE.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='depthimage_to_laserscan',
            executable='depthimage_to_laserscan_node',
            name='depthimage_to_laserscan',
            remappings=[
                ('depth', '/camera/camera/depth/image_rect_raw'),
                ('depth_camera_info', '/camera/camera/depth/camera_info'),
            ],
            parameters=[{
                'output_frame': 'base_link',
                'range_min': 0.3,
                'range_max': 4.0,
                'scan_height': 10,
            }],
            ros_arguments=[
                '--ros-args',
                '--param',
                'qos_overrides./camera/camera/depth/camera_info.subscription.reliability:=best_effort',
                '--param',
                'qos_overrides./camera/camera/depth/image_rect_raw.subscription.reliability:=best_effort',
            ]
        )
    ])
