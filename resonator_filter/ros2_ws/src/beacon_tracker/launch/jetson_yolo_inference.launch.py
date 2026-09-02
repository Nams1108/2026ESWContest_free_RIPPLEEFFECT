#!/usr/bin/env python3
"""Launch only the Jetson YOLO/HTTP side of the split vision pipeline."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory("beacon_tracker"))
    params_file = LaunchConfiguration("params_file")
    default_params = str(package_share / "config" / "waffle.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params),
        Node(
            package="beacon_tracker",
            executable="jetson_yolo_inference_node",
            name="jetson_yolo_inference_node",
            output="screen",
            parameters=[params_file],
        ),
    ])
