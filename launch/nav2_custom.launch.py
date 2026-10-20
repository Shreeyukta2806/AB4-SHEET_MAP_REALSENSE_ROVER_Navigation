"""
Nav2 Custom Launch — CEAM GoKart Project, MAHE Bengaluru

Launches the full Nav2 navigation stack without map_server or AMCL.
Use alongside a static map→odom transform (no AMCL needed for flat indoor track):

    ros2 run tf2_ros static_transform_publisher \
        --frame-id map --child-frame-id odom \
        --x 0 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0

Params file: /home/rover1/nav2_params.yaml
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='nav2_controller',
            executable='controller_server',
            output='screen',
            parameters=['/home/rover1/nav2_params.yaml']
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            output='screen',
            parameters=['/home/rover1/nav2_params.yaml']
        ),
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            output='screen',
            parameters=['/home/rover1/nav2_params.yaml']
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            output='screen',
            parameters=['/home/rover1/nav2_params.yaml']
        ),
        Node(
            package='nav2_velocity_smoother',
            executable='velocity_smoother',
            output='screen',
            parameters=['/home/rover1/nav2_params.yaml']
        ),
        Node(
            package='nav2_collision_monitor',
            executable='collision_monitor',
            output='screen',
            parameters=['/home/rover1/nav2_params.yaml']
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'autostart': True,
                'node_names': [
                    'controller_server',
                    'planner_server',
                    'behavior_server',
                    'bt_navigator',
                    'velocity_smoother',
                    'collision_monitor',
                ]
            }]
        ),
    ])
