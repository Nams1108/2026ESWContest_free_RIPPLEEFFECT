#!/usr/bin/env python3
"""Launch the map-independent coverage/evidence beacon search.

This replaces (rather than accompanies) ``beacon_tracker.launch.py`` for the
new algorithm.  It starts one owner for each microphone: Sipeed XYXY packet
validation continuously, and ReSpeaker DOA only after the search reaches the
selected room's open interior pose.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _validate_boolean_launch_arguments(context):
    """Fail before spawning hardware/search nodes on a mistyped boolean."""

    allowed = {"true", "1", "false", "0"}
    for name in ("search_enabled", "launch_yolo", "launch_diagnostics"):
        value = LaunchConfiguration(name).perform(context).strip().lower()
        if value not in allowed:
            raise RuntimeError(
                f"invalid {name}: {value!r}; use exactly true or false"
            )
    return []


def generate_launch_description():
    package_share = Path(get_package_share_directory("beacon_tracker"))
    params_file = LaunchConfiguration("params_file")
    search_enabled = LaunchConfiguration("search_enabled")
    launch_yolo = LaunchConfiguration("launch_yolo")
    launch_diagnostics = LaunchConfiguration("launch_diagnostics")
    default_params = str(package_share / "config" / "waffle.yaml")

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument(
            "search_enabled",
            default_value="false",
            description="Set true only after Nav2/map/localization are ready",
        ),
        DeclareLaunchArgument(
            "launch_yolo",
            default_value="true",
            description=(
                "Start NUC RealSense JPEG relay + local depth/TF person localizer. "
                "Run jetson_yolo_inference_node separately on Jetson."
            ),
        ),
        DeclareLaunchArgument(
            "launch_diagnostics",
            default_value="true",
            description=(
                "Write UWB, XYXY PASS/FAIL, Nav2 goals/paths, and internal "
                "map-evidence state snapshots until this launch receives Ctrl+C."
            ),
        ),
        # Validate every conditional argument before the first process is
        # spawned. Previously launch_yolo:=true~ failed late and left an
        # orphan map_evidence_search_node publishing competing Nav2 goals.
        OpaqueFunction(function=_validate_boolean_launch_arguments),
        Node(package="beacon_tracker", executable="uwb_range_node", name="uwb_range_node", output="screen", parameters=[params_file]),
        Node(package="beacon_tracker", executable="packet_lock_node", name="packet_lock_node", output="screen", parameters=[params_file]),
        Node(
            package="beacon_tracker", executable="doa_angle_node", name="doa_angle_node", output="screen",
            parameters=[params_file, {"require_enable": True}],
        ),
        Node(
            package="beacon_tracker", executable="map_evidence_search_node", name="map_evidence_search_node", output="screen",
            parameters=[params_file, {"search_enabled": search_enabled}],
        ),
        Node(
            package="beacon_tracker",
            executable="evidence_diagnostics_logger_node",
            name="evidence_diagnostics_logger_node",
            output="screen",
            parameters=[params_file],
            condition=IfCondition(launch_diagnostics),
        ),
        Node(
            package="beacon_tracker", executable="camera_jpeg_relay_node", name="camera_jpeg_relay_node", output="screen",
            parameters=[params_file], condition=IfCondition(launch_yolo),
        ),
        Node(
            package="beacon_tracker", executable="person_localizer_node", name="person_localizer_node", output="screen",
            parameters=[params_file], condition=IfCondition(launch_yolo),
        ),
    ])
