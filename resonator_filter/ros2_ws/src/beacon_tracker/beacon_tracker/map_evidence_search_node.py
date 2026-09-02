#!/usr/bin/env python3
"""Autonomous map-coverage beacon search using UWB and validated XYXY evidence.

The node does not use a hard-coded room list, map image, or acoustic bearing.
It creates a safe coverage route from ``/map`` and only enables DOA after a
map-derived room/entry decision has reached an open interior pose.
"""

from __future__ import annotations

import csv
import fcntl
import json
import math
import os
import sys
import time
import zlib
from collections import deque
from datetime import datetime
from pathlib import Path

import rclpy
import numpy as np
from ament_index_python.packages import get_package_share_directory
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav2_msgs.msg import Costmap
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from beacon_tracker.map_evidence import (
    DoaObservation,
    DoaSpaceAssessment,
    EvidenceConfig,
    EvidenceWaypoint,
    MapPoint,
    RoomDecision,
    assess_doa_open_space,
    build_topology,
    closed_space_first_coverage_waypoints,
    coverage_waypoints,
    decide_room_entry,
    doa_bearing_is_open,
    estimate_zone_probabilities,
    packet_audio_level_dbfs,
    zone_local_waypoints,
)
from beacon_tracker.recovery_policy import (
    assess_continuous_fresh_recovery,
    assess_stationary_uwb_recovery,
)
from beacon_tracker.survey_policy import assess_survey_departure, assess_survey_gate
from beacon_tracker.uwb_trilateration import aggregate_ranges


TRANSIENT_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _wrap_angle_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class MapEvidenceSearchNode(Node):
    """Run generic coverage, evidence collection, entry, then DOA gating."""

    def __init__(self) -> None:
        super().__init__("map_evidence_search_node")
        self._declare_parameters()
        self._map_frame = str(self.get_parameter("map_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._config = EvidenceConfig(
            robot_clearance_m=float(self.get_parameter("robot_clearance_m").value),
            coverage_spacing_m=float(self.get_parameter("coverage_spacing_m").value),
            coverage_max_waypoints=int(self.get_parameter("coverage_max_waypoints").value),
            zone_clearance_m=float(self.get_parameter("zone_clearance_m").value),
            portal_min_clearance_m=float(self.get_parameter("portal_min_clearance_m").value),
            portal_max_clearance_m=float(self.get_parameter("portal_max_clearance_m").value),
            minimum_zone_area_m2=float(self.get_parameter("minimum_zone_area_m2").value),
            candidate_spacing_m=float(self.get_parameter("candidate_spacing_m").value),
            max_range_sigma_m=float(self.get_parameter("max_range_sigma_m").value),
            min_evidence_waypoints=int(self.get_parameter("min_evidence_waypoints").value),
            min_packet_valid_packets=int(self.get_parameter("min_packet_valid_packets").value),
            portal_evidence_radius_m=float(self.get_parameter("portal_evidence_radius_m").value),
            coverage_unknown_clearance_m=float(self.get_parameter("coverage_unknown_clearance_m").value),
            portal_unknown_clearance_m=float(self.get_parameter("portal_unknown_clearance_m").value),
            exclude_border_connected_free_space=bool(
                self.get_parameter("exclude_border_connected_free_space").value
            ),
            doa_weight=float(self.get_parameter("doa_fusion_weight").value),
            doa_sigma_rad=float(self.get_parameter("doa_fusion_sigma_rad").value),
            doa_min_observations=int(self.get_parameter("doa_probe_required_observations").value),
            doa_require_line_of_sight=bool(self.get_parameter("doa_fusion_require_line_of_sight").value),
            min_decision_margin=float(self.get_parameter("min_decision_margin").value),
            min_audio_margin_db=float(self.get_parameter("min_audio_margin_db").value),
            entry_staging_distance_m=float(self.get_parameter("entry_staging_distance_m").value),
            entry_interior_distance_m=float(self.get_parameter("entry_interior_distance_m").value),
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self._path_client = ActionClient(self, ComputePathToPose, "compute_path_to_pose")
        self._map: OccupancyGrid | None = None
        # The static /map is used to create generic coverage.  The Nav2 global
        # costmap additionally contains the live, inflated obstacle space and
        # is the final authority before a NavigateToPose request is sent.
        self._nav_costmap: Costmap | None = None
        self._nav_costmap_wait_logged = False
        self._map_signature: tuple | None = None
        self._topology = None
        self._coverage: list[MapPoint] = []
        self._coverage_index = 0
        self._coverage_cycle = 0
        self._evidence: list[EvidenceWaypoint] = []
        self._state = "WAIT_MAP"
        self._state_since = time.monotonic()
        self._nav_goal_handle = None
        self._nav_goal_pending = False
        # Invalidates callbacks from cancelled goals so a delayed Nav2 cancel
        # result cannot fail the next safe goal after UWB recontact.
        self._nav_generation = 0
        self._path_preflight_pending = False
        self._path_preflight_signature: tuple | None = None
        self._path_preflight_passed_signature: tuple | None = None
        self._nav_result: str | None = None
        self._goal_kind = ""
        self._goal_signature: tuple | None = None
        self._current_waypoint: MapPoint | None = None
        self._current_waypoint_id = 0
        self._collect_started = 0.0
        self._range_values: list[float] = []
        self._packet_metrics: list[dict[str, object]] = []
        # Initial search is a map survey, not UWB gradient following.  UWB
        # and full-XYXY packet levels are sampled only at reached Nav2
        # waypoints, with the real TF pose and timestamps retained in CSV.
        self._survey_first_fresh_at: float | None = None
        self._survey_first_fresh_pose: MapPoint | None = None
        self._survey_first_fresh_range_m: float | None = None
        # The survey boundary is based on stable sliding-window medians.  A
        # single NLOS value must never end collection.
        self._survey_departure_samples: deque[tuple[float, float]] = deque(
            maxlen=500
        )
        self._survey_best_window_range_m: float | None = None
        self._survey_departure_assessment: dict[str, object] | None = None
        self._survey_range_limit_seen = False
        self._survey_range_limit_at: float | None = None
        self._survey_selection_ready = False
        self._survey_selection_reason: str | None = None
        self._survey_room_approach_active = False
        self._survey_waypoint_records = 0
        # During map coverage raw observations are always timestamped.  When
        # repeated full packets and synchronized FRESH samples occur in one
        # spatial bin, that moving bin also becomes conservative joint room
        # evidence instead of being discarded merely because Nav2 is moving.
        self._survey_motion_bins: dict[tuple[int, int], dict[str, object]] = {}
        self._survey_motion_recorded_bins: set[tuple[int, int]] = set()
        self._survey_motion_evidence_id = 0
        # Fully validated XYXY packet samples received while Nav2 is moving.
        # These always use the resonator direction metric (900/1050 Hz), not
        # the four-tone identity weak-link level.
        self._xyxy_navigation_samples: deque[tuple[float, float]] = deque()
        self._latest_xyxy_direction_level: float | None = None
        self._latest_xyxy_direction_quality: float | None = None
        self._latest_xyxy_direction_at = 0.0
        self._latest_packet_metric: dict[str, object] | None = None
        self._last_stationary_xyxy_level: float | None = None
        self._xyxy_goal_reference_level: float | None = None
        self._xyxy_goal_start_pose: MapPoint | None = None
        self._xyxy_goal_started_at = 0.0
        self._xyxy_degradation_started_at: float | None = None
        self._xyxy_degradation_resume_state: str | None = None
        # History is made only from complete XYXY packets. It remains useful
        # for diagnostics and the route-worsening guard, but it must never be
        # used as a one-shot acoustic bearing.
        self._xyxy_observations: deque[tuple[float, float, MapPoint]] = deque()
        # UWB FRESH observations are spatial anchors, not a live range
        # solution.  When the link becomes HOLD/LOST, recovery begins from
        # these *measured map positions* and then uses only map-safe coverage
        # waypoints while collecting full XYXY packet amplitudes.
        self._fresh_uwb_observations: deque[tuple[float, float, MapPoint]] = deque()
        self._xyxy_map_search_queue: list[tuple[MapPoint, str, float | None]] = []
        self._xyxy_map_search_index = 0
        self._xyxy_map_search_cycle = 0
        # A degraded-search Nav2 result is valid only for the exact queue
        # index/point that created it.  This prevents a delayed result from
        # starting collection for the next waypoint.
        self._xyxy_active_goal_index: int | None = None
        self._xyxy_active_goal_point: MapPoint | None = None
        self._xyxy_map_search_samples: deque[dict[str, object]] = deque(maxlen=200)
        self._xyxy_priority_reference_level: float | None = None
        # UWB recovery is decided only after a complete stationary XYXY
        # collection window. FRESH received while Nav2 is moving is retained
        # as evidence but cannot cancel the current map-safe goal.
        self._uwb_recovery_samples: list[tuple[float, str, float | None]] = []
        self._uwb_recovery_start_pose: MapPoint | None = None
        self._last_range_received_at = 0.0
        # Prevent immediate A<->B oscillation after a cancelled/visited search
        # point. Only the two most recent points are retained for a bounded TTL.
        self._recent_goal_tabu: deque[tuple[float, MapPoint, str]] = deque()
        # Point-only cooldown still allowed the planner to oscillate between
        # nearby points in the same room/section.  Zone cooldown is derived
        # from the current map topology and therefore stays map-generic.
        self._zone_cooldowns: dict[int, tuple[float, str, MapPoint]] = {}
        self._last_zone_cooldown_wait_log = 0.0
        self._last_range_m: float | None = None
        self._uwb_status = "INVALID"
        self._last_uwb_usable_at = 0.0
        self._uwb_hold_started_at: float | None = None
        self._fresh_uwb_started_at: float | None = None
        self._fresh_uwb_values: deque[tuple[float, float]] = deque()
        self._uwb_degraded_started_at: float | None = None
        # Coverage starts without a beacon by design. Once one validated XYXY
        # packet and a stationary UWB observation have been collected, loss
        # of UWB is evidence that the robot is moving into a bad/NLOS route.
        self._beacon_evidence_seen = False
        self._uwb_loss_resume_state: str | None = None
        self._packet_locked = False
        self._last_decision: RoomDecision | None = None
        self._close_anchor: EvidenceWaypoint | None = None
        self._close_anchor_zone_id: int | None = None
        self._close_anchor_local_samples = 0
        # YOLO publishes a depth-confirmed person position in map coordinates.
        # The search node owns the resulting Nav2 action so it never races a
        # second vision node or a teleop publisher for goal ownership.
        self._latest_person_pose: MapPoint | None = None
        self._latest_person_received_at = 0.0
        self._active_person_approach: MapPoint | None = None
        self._victim_resume_state = "DOA_ACTIVE"
        self._yolo_verify_started_at = 0.0
        self._yolo_verify_authorized_until = 0.0
        self._yolo_verify_rearm_at = 0.0
        self._mission_complete = False
        self._doa_enabled = False
        # ReSpeaker output is deliberately retained as a weak, map-frame
        # section likelihood.  It never becomes a direct coordinate or a
        # bare /cmd_vel heading command.
        self._latest_doa_angle_rad: float | None = None
        self._latest_doa_angle_at = 0.0
        self._latest_doa_metric: dict[str, object] | None = None
        self._latest_doa_metric_at = 0.0
        self._doa_stable = False
        self._doa_observations: deque[tuple[float, DoaObservation]] = deque()
        self._doa_probe_target: MapPoint | None = None
        self._doa_probe_targets: list[MapPoint] = []
        self._doa_probe_angles: list[float] = []
        self._doa_probe_metrics: list[dict[str, object]] = []
        self._doa_probe_space: DoaSpaceAssessment | None = None
        self._doa_stationary_reference: MapPoint | None = None
        self._doa_stationary_since: float | None = None
        self._doa_active_space: DoaSpaceAssessment | None = None
        self._doa_active_space_at = 0.0
        self._doa_probe_attempts = 0
        self._doa_probe_last_evidence_count = 0
        self._yolo_ready = False

        self._state_pub = self.create_publisher(String, "/beacon_search/state", TRANSIENT_QOS)
        self._doa_enable_pub = self.create_publisher(Bool, "/beacon/doa_enabled", TRANSIENT_QOS)
        self._selected_room_pub = self.create_publisher(PoseStamped, "/beacon_search/selected_room", TRANSIENT_QOS)
        self._entry_pub = self.create_publisher(PoseStamped, "/beacon_search/entry_pose", TRANSIENT_QOS)
        # Observation-only tests publish the goal the algorithm would have
        # requested here.  This is deliberately not a Nav2 action/topic and
        # therefore cannot move the robot.
        self._proposed_goal_pub = self.create_publisher(
            PoseStamped,
            "/beacon_search/proposed_goal",
            TRANSIENT_QOS,
        )
        self._yolo_ready_pub = self.create_publisher(Bool, "/beacon_search/yolo_ready", TRANSIENT_QOS)
        self._mission_complete_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("mission_complete_topic").value),
            TRANSIENT_QOS,
        )
        self._room_probability_pub = self.create_publisher(
            String,
            str(self.get_parameter("room_probability_topic").value),
            TRANSIENT_QOS,
        )
        self._room_probability_marker_pub = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("room_probability_marker_topic").value),
            TRANSIENT_QOS,
        )
        self._diagnostic_pub = self.create_publisher(
            String,
            str(self.get_parameter("diagnostic_topic").value),
            50,
        )
        self._diagnostic_config_pub = self.create_publisher(
            String,
            str(self.get_parameter("diagnostic_config_topic").value),
            TRANSIENT_QOS,
        )

        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, str(self.get_parameter("map_topic").value), self._map_callback, map_qos)
        self.create_subscription(
            Costmap,
            str(self.get_parameter("nav_costmap_topic").value),
            self._nav_costmap_callback,
            5,
        )
        self.create_subscription(Float32, "/uwb/range_m", self._range_callback, 10)
        self.create_subscription(String, "/uwb/status", self._status_callback, 10)
        self.create_subscription(Bool, "/beacon/packet_locked", self._packet_callback, TRANSIENT_QOS)
        self.create_subscription(String, "/beacon/packet_metric", self._packet_metric_callback, 20)
        self.create_subscription(
            Float32,
            str(self.get_parameter("doa_angle_topic").value),
            self._doa_angle_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("doa_metric_topic").value),
            self._doa_metric_callback,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("doa_stable_topic").value),
            self._doa_stable_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("person_pose_topic").value),
            self._person_pose_callback,
            10,
        )

        self._raw_writer, self._summary_writer, self._streams = self._open_logs()
        self._publish_doa_enabled(False)
        self._publish_yolo_ready(False)
        self._publish_mission_complete(False)
        self._timer = self.create_timer(0.2, self._tick)
        self._diagnostic_timer = self.create_timer(
            float(self.get_parameter("diagnostic_publish_sec").value),
            self._publish_diagnostic_snapshot,
        )
        self._probability_timer = self.create_timer(
            float(self.get_parameter("room_probability_publish_sec").value),
            self._publish_room_probabilities,
        )
        self.get_logger().info(
            "map evidence search ready: disabled until search_enabled:=true; "
            "coverage + UWB/XYXY records + live section probabilities + interior DOA"
        )
        self._publish_diagnostic_config()
        self._publish_diagnostic_event("SESSION_START")

    def _declare_parameters(self) -> None:
        self.declare_parameter("search_enabled", False)
        # Production defaults to motion enabled.  The dedicated no-motion
        # launch overrides this to false and blocks every Nav2 action request.
        self.declare_parameter("navigation_commands_enabled", True)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("nav_costmap_topic", "/global_costmap/costmap_raw")
        self.declare_parameter("require_nav_costmap", True)
        # Nav2 costs 253--255 mean inscribed/lethal/unknown and are rejected.
        self.declare_parameter("nav_costmap_max_cost", 252)
        # A clear costmap cell alone does not prove it is connected to the
        # robot. Confirm planner reachability before NavigateToPose can launch
        # recovery spins/backups for a map-generated point.
        self.declare_parameter("require_nav_path_preflight", True)
        self.declare_parameter(
            "map_search_behavior_tree",
            str(
                Path(get_package_share_directory("beacon_tracker"))
                / "config"
                / "navigate_to_pose_no_recovery.xml"
            ),
        )
        self.declare_parameter("map_occupancy_threshold", 50)
        self.declare_parameter("map_unknown_is_blocked", True)
        self.declare_parameter("robot_clearance_m", 0.30)
        self.declare_parameter("coverage_spacing_m", 1.8)
        self.declare_parameter("coverage_max_waypoints", 60)
        self.declare_parameter("coverage_closed_space_first", True)
        self.declare_parameter("zone_clearance_m", 0.60)
        self.declare_parameter("portal_min_clearance_m", 0.25)
        self.declare_parameter("portal_max_clearance_m", 0.65)
        self.declare_parameter("minimum_zone_area_m2", 1.2)
        self.declare_parameter("candidate_spacing_m", 0.50)
        self.declare_parameter("max_range_sigma_m", 0.45)
        self.declare_parameter("min_evidence_waypoints", 4)
        self.declare_parameter("min_packet_valid_packets", 2)
        self.declare_parameter("portal_evidence_radius_m", 1.5)
        self.declare_parameter("coverage_unknown_clearance_m", 0.40)
        self.declare_parameter("portal_unknown_clearance_m", 0.35)
        self.declare_parameter("exclude_border_connected_free_space", True)
        # Map-first survey.  Before selecting a room, UWB is evidence only:
        # it never cancels or redirects a coverage goal.  8 m is the current
        # BU03 field-use boundary and remains a YAML parameter because the
        # final threshold still needs to be measured in the deployment site.
        self.declare_parameter("initial_map_survey_enabled", True)
        self.declare_parameter("survey_uwb_max_valid_range_m", 8.0)
        self.declare_parameter("survey_collection_stop_range_m", 8.0)
        self.declare_parameter("survey_range_increase_m", 1.0)
        self.declare_parameter("survey_departure_window_sec", 3.0)
        self.declare_parameter("survey_departure_min_span_sec", 2.0)
        self.declare_parameter("survey_departure_min_samples", 10)
        self.declare_parameter("survey_departure_max_sigma_m", 0.45)
        self.declare_parameter("survey_min_fresh_samples_per_waypoint", 3)
        self.declare_parameter("survey_min_joint_waypoints", 4)
        self.declare_parameter("survey_min_observed_zones", 2)
        self.declare_parameter("survey_select_on_coverage_complete", True)
        self.declare_parameter("survey_motion_logging_enabled", True)
        self.declare_parameter("survey_motion_evidence_enabled", True)
        self.declare_parameter("survey_motion_bin_m", 0.60)
        self.declare_parameter("survey_motion_sync_window_sec", 3.0)
        self.declare_parameter("survey_motion_bin_history_sec", 30.0)
        self.declare_parameter("survey_motion_min_fresh_samples", 3)
        self.declare_parameter("survey_motion_min_packets", 2)
        self.declare_parameter("survey_extrema_max_distance_m", 3.0)
        # ReSpeaker DOA is only a low-weight map-section likelihood. It is
        # sampled at two separated, Nav2-safe probe poses after UWB + XYXY
        # have already produced a plausible section ranking.
        self.declare_parameter("doa_fusion_enabled", True)
        self.declare_parameter("doa_angle_topic", "/beacon/doa_angle_rad")
        self.declare_parameter("doa_stable_topic", "/beacon/doa_stable")
        self.declare_parameter("doa_metric_topic", "/beacon/doa_metric")
        self.declare_parameter("doa_fusion_weight", 0.20)
        self.declare_parameter("doa_fusion_sigma_rad", 0.55)
        self.declare_parameter("doa_fusion_require_line_of_sight", True)
        self.declare_parameter("doa_probe_max_range_m", 6.0)
        self.declare_parameter("doa_probe_min_base_probability", 0.55)
        self.declare_parameter("doa_probe_min_base_margin", 0.08)
        self.declare_parameter("doa_probe_timeout_sec", 8.0)
        self.declare_parameter("doa_probe_min_stable_samples", 2)
        self.declare_parameter("doa_probe_max_spread_rad", math.radians(20.0))
        self.declare_parameter("doa_probe_required_observations", 2)
        self.declare_parameter("doa_probe_min_separation_m", 0.60)
        self.declare_parameter("doa_probe_min_clearance_m", 0.60)
        self.declare_parameter("doa_probe_max_attempts", 2)
        self.declare_parameter("doa_observation_history_sec", 180.0)
        # Static-map acoustic-space gate. A general open area carries
        # 27.5% DOA weight; a one-sided 120-degree fan carries only 12.5%.
        self.declare_parameter("doa_space_ray_step_deg", 10.0)
        self.declare_parameter("doa_space_ray_max_range_m", 3.0)
        self.declare_parameter("doa_space_min_wall_clearance_m", 0.60)
        self.declare_parameter("doa_space_general_wall_clearance_m", 1.20)
        self.declare_parameter("doa_space_open_depth_m", 1.50)
        self.declare_parameter("doa_space_general_min_open_sectors", 4)
        self.declare_parameter("doa_space_fan_width_deg", 120.0)
        self.declare_parameter("doa_space_fan_center_depth_m", 2.0)
        self.declare_parameter("doa_space_fan_min_open_ratio", 0.70)
        self.declare_parameter("doa_space_general_confidence_weight", 1.375)
        self.declare_parameter("doa_space_fan_confidence_weight", 0.625)
        self.declare_parameter("doa_stationary_sec", 1.50)
        self.declare_parameter("doa_stationary_position_tolerance_m", 0.03)
        self.declare_parameter("doa_stationary_yaw_tolerance_rad", math.radians(3.0))
        self.declare_parameter("doa_metric_max_age_sec", 0.75)
        self.declare_parameter("doa_min_active_tone_snr_db", 6.0)
        self.declare_parameter("doa_min_music_peak_margin_db", 3.0)
        self.declare_parameter("doa_max_srp_candidate_diff_deg", 20.0)
        # ReSpeaker location/orientation relative to base_link.  It is
        # intentionally independent from the SIPEED +0.25m packet sensor.
        self.declare_parameter("doa_sensor_x_m", 0.0)
        self.declare_parameter("doa_sensor_y_m", 0.0)
        self.declare_parameter("doa_sensor_yaw_rad", 0.0)
        self.declare_parameter("min_decision_margin", 0.12)
        self.declare_parameter("min_audio_margin_db", 2.0)
        self.declare_parameter("entry_staging_distance_m", 0.45)
        self.declare_parameter("entry_interior_distance_m", 0.90)
        self.declare_parameter("close_anchor_range_m", 0.90)
        self.declare_parameter("close_anchor_max_sigma_m", 0.12)
        self.declare_parameter("close_anchor_min_packets", 2)
        self.declare_parameter("close_anchor_radius_m", 2.0)
        self.declare_parameter("close_anchor_max_local_waypoints", 4)
        self.declare_parameter("uwb_navigation_loss_timeout_sec", 3.0)
        # HOLD is useful while stationary at a waypoint, but it can also be
        # a BU03 stale response after the radio path is lost.  Do not let it
        # keep a moving Nav2 goal alive unless explicitly opted in.
        self.declare_parameter("uwb_navigation_allow_hold", False)
        self.declare_parameter("uwb_navigation_hold_timeout_sec", 5.0)
        self.declare_parameter("uwb_guard_min_valid_packets", 1)
        # Moving-route guard based on fully validated XYXY packets. The
        # decoder's direction metric is the weaker of 900 Hz and 1050 Hz.
        # It is negative evidence only: a sustained drop stops/rechecks a
        # route, but a single loud packet never creates a direction goal.
        self.declare_parameter("xyxy_navigation_guard_enabled", True)
        self.declare_parameter("xyxy_navigation_drop_db", 6.0)
        self.declare_parameter("xyxy_navigation_window_sec", 8.0)
        self.declare_parameter("xyxy_navigation_min_samples", 2)
        self.declare_parameter("xyxy_navigation_hold_sec", 3.0)
        self.declare_parameter("xyxy_navigation_min_travel_m", 0.40)
        self.declare_parameter("xyxy_navigation_reference_max_age_sec", 15.0)
        self.declare_parameter("xyxy_navigation_recheck_delay_sec", 3.0)
        # UWB-degraded map search is stricter than a selected-room route.  A
        # sustained 2 dB loss rejects the current candidate so that the next
        # map-safe point is chosen by full-XYXY amplitude priority.
        self.declare_parameter("xyxy_priority_route_drop_db", 2.0)
        self.declare_parameter("xyxy_priority_route_hold_sec", 2.0)
        self.declare_parameter("xyxy_priority_route_min_travel_m", 0.30)
        # A close, stable UWB observation plus a recent full XYXY packet can
        # pause map travel for a short camera verification before room entry.
        self.declare_parameter("yolo_xyxy_gate_enabled", True)
        self.declare_parameter("yolo_xyxy_metric_max_age_sec", 10.0)
        self.declare_parameter("yolo_uwb_fresh_sec", 2.0)
        self.declare_parameter("yolo_uwb_min_fresh_samples", 3)
        self.declare_parameter("yolo_activation_uwb_range_m", 3.0)
        self.declare_parameter("yolo_verify_timeout_sec", 10.0)
        self.declare_parameter("yolo_verify_rearm_sec", 15.0)
        # A HOLD/LOST link must not leave the robot stopped indefinitely.
        # Build a map-safe search queue from positions where UWB was actually
        # FRESH, then compare full XYXY 900/1050-Hz amplitudes at those
        # anchors and nearby generic coverage points.  The queue is bounded so
        # one dropout cannot become a long second coverage mission.
        self.declare_parameter("uwb_fresh_map_search_enabled", True)
        self.declare_parameter("uwb_fresh_history_sec", 600.0)
        self.declare_parameter("uwb_fresh_history_min_spacing_m", 0.35)
        self.declare_parameter("uwb_fresh_map_search_max_waypoints", 12)
        self.declare_parameter("uwb_fresh_map_search_min_goal_distance_m", 0.25)
        # Never turn a UWB dropout into an unrestricted second coverage pass.
        # Generic comparison points must remain near a physically visited
        # FRESH anchor; otherwise one cooled-down room can send the robot to a
        # distant, acoustically unrelated map section.
        self.declare_parameter("uwb_fresh_map_search_coverage_radius_m", 2.0)
        # Reorder only the remaining map-coverage candidates after two
        # stationary full-XYXY samples disagree by this much.  Historical
        # FRESH anchors are still visited first; this is not bearing steering.
        self.declare_parameter("uwb_fresh_map_search_reprioritize_min_packets", 2)
        self.declare_parameter("uwb_fresh_map_search_reprioritize_margin_db", 3.0)
        self.declare_parameter("uwb_fresh_map_search_reprioritize_min_repeats", 2)
        self.declare_parameter("uwb_fresh_map_search_reprioritize_repeat_radius_m", 0.75)
        self.declare_parameter("uwb_fresh_map_search_reprioritize_max_spread_db", 4.0)
        # A FRESH label alone only means the BU03 value changed by epsilon.
        # Accept link recovery after the complete stationary evidence window
        # passes ratio, variance, and odom/range-consistency checks.
        self.declare_parameter("uwb_recovery_min_fresh_samples", 12)
        self.declare_parameter("uwb_recovery_min_fresh_ratio", 0.60)
        self.declare_parameter("uwb_recovery_max_sigma_m", 0.35)
        self.declare_parameter("uwb_recovery_max_stationary_motion_m", 0.15)
        self.declare_parameter("uwb_recovery_range_motion_margin_m", 0.35)
        self.declare_parameter("uwb_recovery_range_status_sync_sec", 0.50)
        self.declare_parameter("uwb_recovery_fresh_confirm_sec", 3.0)
        self.declare_parameter("uwb_recovery_fresh_confirm_min_samples", 10)
        self.declare_parameter("uwb_recovery_fresh_confirm_max_sigma_m", 0.35)
        self.declare_parameter("xyxy_priority_history_bin_m", 0.50)
        self.declare_parameter("xyxy_priority_min_gain_db", 1.0)
        self.declare_parameter("recent_goal_tabu_count", 2)
        self.declare_parameter("recent_goal_tabu_sec", 45.0)
        self.declare_parameter("recent_goal_tabu_radius_m", 0.45)
        self.declare_parameter("zone_cooldown_sec", 60.0)
        self.declare_parameter("sipeed_sensor_x_m", 0.25)
        self.declare_parameter("sipeed_sensor_y_m", 0.0)
        self.declare_parameter("room_probability_topic", "/beacon_search/room_probabilities")
        self.declare_parameter("room_probability_marker_topic", "/beacon_search/room_probability_markers")
        self.declare_parameter("room_probability_publish_sec", 1.0)
        self.declare_parameter("room_probability_temperature", 0.18)
        # YOLO target handoff. Camera inference is admitted at 3 m by the
        # independent yolo_activation_uwb_range_m gate above. Robot motion
        # remains stricter: either UWB is within the close-approach range or a
        # fresh aligned-depth/TF person pose is inside the bounded depth-safe
        # approach range.
        self.declare_parameter("person_pose_topic", "/beacon_search/person_pose")
        self.declare_parameter("mission_complete_topic", "/beacon_search/mission_complete")
        self.declare_parameter("person_pose_timeout_sec", 2.0)
        self.declare_parameter("person_uwb_enable_range_m", 1.0)
        self.declare_parameter("person_depth_approach_enabled", True)
        self.declare_parameter("person_depth_approach_min_m", 0.35)
        self.declare_parameter("person_depth_approach_max_m", 3.0)
        self.declare_parameter("person_approach_standoff_m", 0.80)
        self.declare_parameter("person_arrival_tolerance_m", 0.30)
        self.declare_parameter("waypoint_settle_sec", 1.0)
        self.declare_parameter("evidence_window_sec", 8.0)
        self.declare_parameter("evaluation_interval_waypoints", 6)
        self.declare_parameter("max_coverage_cycles", 0)
        self.declare_parameter("output_dir", "~/resonator_filter/outputs")
        # Structured state-machine telemetry consumed by
        # evidence_diagnostics_logger_node.  It records the exact goal,
        # guard, threshold, and state variables that led to each decision.
        self.declare_parameter("diagnostic_topic", "/beacon_search/diagnostics")
        self.declare_parameter(
            "diagnostic_config_topic", "/beacon_search/diagnostics_config"
        )
        self.declare_parameter("diagnostic_publish_sec", 0.2)

    @staticmethod
    def _diagnostic_point(point: MapPoint | None) -> dict[str, float] | None:
        if point is None:
            return None
        return {
            "x_m": round(float(point.x_m), 4),
            "y_m": round(float(point.y_m), 4),
            "yaw_rad": round(float(point.yaw_rad), 4),
        }

    @staticmethod
    def _diagnostic_age(now: float, timestamp: float | None) -> float | None:
        if timestamp is None or timestamp <= 0.0:
            return None
        return round(max(0.0, now - timestamp), 4)

    @staticmethod
    def _diagnostic_parameter_value(value):
        """Keep a parameter snapshot JSON-safe without losing arrays."""

        if isinstance(value, (str, bool, int, float)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [MapEvidenceSearchNode._diagnostic_parameter_value(item) for item in value]
        return str(value)

    def _algorithm_parameter_snapshot(self) -> dict[str, object]:
        """Record every effective map-evidence parameter once per session."""

        # ``Node.list_parameters`` exists in Jazzy but is absent from the
        # Humble rclpy used by the NUC.  The node-local parameter dictionary
        # is populated by ``declare_parameter`` on both distributions.
        names = sorted(self._parameters)
        return {
            name: self._diagnostic_parameter_value(self.get_parameter(name).value)
            for name in names
        }

    def _diagnostic_snapshot(self) -> dict[str, object]:
        """Return all inputs and internal controls relevant to one decision."""

        now = time.monotonic()
        robot = self._robot_pose()
        topology = self._topology
        decision = self._last_decision
        return {
            "state": self._state,
            "state_age_sec": round(now - self._state_since, 4),
            "robot_map": self._diagnostic_point(robot),
            "current_waypoint": self._diagnostic_point(self._current_waypoint),
            "nav": {
                "commands_enabled": bool(
                    self.get_parameter("navigation_commands_enabled").value
                ),
                "goal_kind": self._goal_kind,
                "goal_signature": list(self._goal_signature) if self._goal_signature else None,
                "goal_pending": self._nav_goal_pending,
                "goal_active": self._nav_goal_handle is not None,
                "result": self._nav_result,
                "generation": self._nav_generation,
                "path_preflight_pending": self._path_preflight_pending,
                "path_preflight_signature": (
                    list(self._path_preflight_signature)
                    if self._path_preflight_signature else None
                ),
                "xyxy_active_goal_index": self._xyxy_active_goal_index,
                "xyxy_active_goal_point": self._diagnostic_point(
                    self._xyxy_active_goal_point
                ),
            },
            "coverage": {
                "cycle": self._coverage_cycle,
                "index": self._coverage_index,
                "total_waypoints": len(self._coverage),
                "current_waypoint_id": self._current_waypoint_id,
                "topology_zones": 0 if topology is None else len(topology.zones),
                "topology_portals": 0 if topology is None else len(topology.portals),
            },
            "survey": {
                "enabled": bool(
                    self.get_parameter("initial_map_survey_enabled").value
                ),
                "first_fresh_age_sec": self._diagnostic_age(
                    now, self._survey_first_fresh_at
                ),
                "first_fresh_pose": self._diagnostic_point(
                    self._survey_first_fresh_pose
                ),
                "first_fresh_range_m": _round_or_none(
                    self._survey_first_fresh_range_m
                ),
                "best_window_range_m": _round_or_none(
                    self._survey_best_window_range_m
                ),
                "departure_assessment": self._survey_departure_assessment,
                "range_limit_seen": self._survey_range_limit_seen,
                "range_limit_age_sec": self._diagnostic_age(
                    now, self._survey_range_limit_at
                ),
                "selection_ready": self._survey_selection_ready,
                "selection_reason": self._survey_selection_reason,
                "room_approach_active": self._survey_room_approach_active,
                "logged_waypoints": self._survey_waypoint_records,
                "joint_waypoints": self._survey_joint_evidence_count(),
                "observed_zones": len(self._survey_observed_zone_ids()),
                "motion_bins": len(self._survey_motion_bins),
                "motion_evidence_bins": len(self._survey_motion_recorded_bins),
            },
            "uwb": {
                "status": self._uwb_status,
                "range_m": _round_or_none(self._last_range_m),
                "link_usable": self._uwb_link_usable(),
                "last_usable_age_sec": self._diagnostic_age(now, self._last_uwb_usable_at),
                "hold_age_sec": self._diagnostic_age(now, self._uwb_hold_started_at),
                "fresh_age_sec": self._diagnostic_age(now, self._fresh_uwb_started_at),
                "fresh_samples": len(self._fresh_uwb_values),
                "fresh_median_m": _round_or_none(self._fresh_uwb_median_m()),
                "degraded_age_sec": self._diagnostic_age(now, self._uwb_degraded_started_at),
                "collect_samples": len(self._range_values),
                "fresh_history_count": len(self._fresh_uwb_observations),
                "recovery_samples": len(self._uwb_recovery_samples),
                "recovery_start_pose": self._diagnostic_point(
                    self._uwb_recovery_start_pose
                ),
            },
            "xyxy": {
                "locked": self._packet_locked,
                "latest_metric": self._latest_packet_metric,
                "direction_level_dbfs": _round_or_none(self._latest_xyxy_direction_level),
                "direction_quality_db": _round_or_none(self._latest_xyxy_direction_quality),
                "direction_metric_age_sec": self._diagnostic_age(now, self._latest_xyxy_direction_at),
                "recent_metric": self._recent_xyxy_available(float(
                    self.get_parameter("yolo_xyxy_metric_max_age_sec").value
                )),
                "history_count": len(self._xyxy_observations),
                "last_stationary_level_dbfs": _round_or_none(self._last_stationary_xyxy_level),
                "goal_reference_level_dbfs": _round_or_none(self._xyxy_goal_reference_level),
                "goal_start_pose": self._diagnostic_point(self._xyxy_goal_start_pose),
                "goal_samples": len(self._xyxy_navigation_samples),
                "degradation_age_sec": self._diagnostic_age(now, self._xyxy_degradation_started_at),
                "map_search_cycle": self._xyxy_map_search_cycle,
                "map_search_index": self._xyxy_map_search_index,
                "map_search_queue_size": len(self._xyxy_map_search_queue),
                "map_search_samples": len(self._xyxy_map_search_samples),
                "priority_reference_level_dbfs": _round_or_none(
                    self._xyxy_priority_reference_level
                ),
                "map_search_target": self._diagnostic_point(self._current_waypoint)
                if self._state in {"NAVIGATE_XYXY_MAP_SEARCH", "COLLECT_XYXY_MAP_SEARCH"}
                else None,
                "recent_goal_tabu": [
                    {
                        "age_sec": round(max(0.0, now - stamp), 4),
                        "point": self._diagnostic_point(point),
                        "reason": reason,
                    }
                    for stamp, point, reason in self._recent_goal_tabu
                ],
                "zone_cooldowns": [
                    {
                        "zone_id": zone_id,
                        "age_sec": round(max(0.0, now - stamp), 4),
                        "point": self._diagnostic_point(point),
                        "reason": reason,
                    }
                    for zone_id, (stamp, reason, point)
                    in sorted(self._zone_cooldowns.items())
                ],
            },
            "evidence": {
                "records": len(self._evidence),
                "packet_metrics_in_window": len(self._packet_metrics),
                "beacon_evidence_seen": self._beacon_evidence_seen,
                "doa": {
                    "fusion_enabled": bool(
                        self.get_parameter("doa_fusion_enabled").value
                    ),
                    "stable": self._doa_stable,
                    "latest_angle_rad": _round_or_none(
                        self._latest_doa_angle_rad
                    ),
                    "latest_angle_age_sec": self._diagnostic_age(
                        now, self._latest_doa_angle_at
                    ),
                    "latest_metric_age_sec": self._diagnostic_age(
                        now, self._latest_doa_metric_at
                    ),
                    "latest_metric": self._latest_doa_metric,
                    "observations": len(self._doa_observations),
                    "probe_attempts": self._doa_probe_attempts,
                    "probe_target": self._diagnostic_point(self._doa_probe_target),
                    "probe_raw_samples": len(self._doa_probe_angles),
                    "probe_space": self._doa_space_payload(
                        self._doa_probe_space
                    ),
                    "active_space": self._doa_space_payload(
                        self._doa_active_space
                    ),
                    "stationary_age_sec": self._diagnostic_age(
                        now, self._doa_stationary_since
                    ),
                },
                "close_anchor": (
                    None if self._close_anchor is None else {
                        "range_m": _round_or_none(self._close_anchor.range_m),
                        "range_sigma_m": _round_or_none(self._close_anchor.range_sigma_m),
                        "zone_id": self._close_anchor_zone_id,
                        "local_samples": self._close_anchor_local_samples,
                    }
                ),
                "selected_zone_id": (
                    None if decision is None else decision.zone.zone_id
                ),
                "decision_reason": None if decision is None else decision.reason,
            },
            "handoff": {
                "doa_enabled": self._doa_enabled,
                "yolo_ready": self._yolo_ready,
                "yolo_verify_age_sec": self._diagnostic_age(now, self._yolo_verify_started_at),
                "yolo_verify_remaining_sec": round(
                    max(0.0, self._yolo_verify_authorized_until - now), 4
                ),
                "person_pose": self._diagnostic_point(self._latest_person_pose),
                "person_pose_age_sec": self._diagnostic_age(now, self._latest_person_received_at),
                "person_approach": self._diagnostic_point(self._active_person_approach),
                "mission_complete": self._mission_complete,
            },
        }

    def _publish_diagnostic_event(
        self,
        event: str,
        details: dict[str, object] | None = None,
    ) -> None:
        if not rclpy.ok():
            return
        payload = {
            "event": event,
            "wall_time_sec": round(self.get_clock().now().nanoseconds / 1e9, 6),
            "monotonic_sec": round(time.monotonic(), 6),
            "snapshot": self._diagnostic_snapshot(),
        }
        if details:
            payload["details"] = details
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        self._diagnostic_pub.publish(message)

    def _publish_diagnostic_config(self) -> None:
        """Retain effective algorithm parameters for a later-starting logger."""

        if not rclpy.ok():
            return
        message = String()
        message.data = json.dumps({
            "event": "ALGORITHM_CONFIG",
            "wall_time_sec": round(self.get_clock().now().nanoseconds / 1e9, 6),
            "parameters": self._algorithm_parameter_snapshot(),
        }, separators=(",", ":"), sort_keys=True)
        self._diagnostic_config_pub.publish(message)

    def _publish_diagnostic_snapshot(self) -> None:
        self._publish_diagnostic_event("SNAPSHOT")

    def _open_logs(self):
        output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_stream = (output_dir / f"map_evidence_{stamp}_events.csv").open("w", newline="", encoding="utf-8")
        summary_stream = (output_dir / f"map_evidence_{stamp}_waypoints.csv").open("w", newline="", encoding="utf-8")
        raw_fields = ["wall_time_sec", "monotonic_sec", "event", "state", "waypoint_id", "map_x_m", "map_y_m", "map_yaw_rad", "sipeed_x_m", "sipeed_y_m", "uwb_status", "uwb_range_m", "packet_level_dbfs", "packet_quality_db", "packet_direction_level_dbfs", "packet_direction_quality_db", "packet_locked", "packet_metric_json"]
        summary_fields = ["wall_time_sec", "monotonic_sec", "record_kind", "waypoint_id", "cycle", "map_x_m", "map_y_m", "map_yaw_rad", "sipeed_x_m", "sipeed_y_m", "zone_id", "uwb_status", "uwb_median_m", "uwb_sigma_m", "uwb_samples", "packet_valid_packets", "packet_locked", "packet_level_dbfs", "packet_quality_db", "packet_direction_level_dbfs", "packet_direction_quality_db"]
        raw_writer = csv.DictWriter(raw_stream, fieldnames=raw_fields)
        summary_writer = csv.DictWriter(summary_stream, fieldnames=summary_fields)
        raw_writer.writeheader()
        summary_writer.writeheader()
        raw_stream.flush()
        summary_stream.flush()
        self.get_logger().info(f"map-evidence logs: {raw_stream.name}, {summary_stream.name}")
        return raw_writer, summary_writer, (raw_stream, summary_stream)

    def _map_callback(self, message: OccupancyGrid) -> None:
        signature = (
            int(message.info.width), int(message.info.height), float(message.info.resolution),
            round(message.info.origin.position.x, 4), round(message.info.origin.position.y, 4),
            round(message.info.origin.orientation.z, 5), round(message.info.origin.orientation.w, 5),
            zlib.crc32(bytes(np.asarray(message.data, dtype=np.int8))),
        )
        if signature != self._map_signature:
            previous_signature = self._map_signature
            self._map_signature = signature
            self._topology = None
            self._coverage = []
            self._coverage_index = 0
            self._close_anchor = None
            self._close_anchor_zone_id = None
            self._close_anchor_local_samples = 0
            if previous_signature is not None:
                # Evidence coordinates belong to one immutable map geometry.
                # A replacement map starts a fresh, map-generic survey.
                self._evidence = []
                self._survey_first_fresh_at = None
                self._survey_first_fresh_pose = None
                self._survey_first_fresh_range_m = None
                self._survey_departure_samples.clear()
                self._survey_best_window_range_m = None
                self._survey_departure_assessment = None
                self._survey_range_limit_seen = False
                self._survey_range_limit_at = None
                self._survey_selection_ready = False
                self._survey_selection_reason = None
                self._survey_room_approach_active = False
                self._survey_waypoint_records = 0
                self._survey_motion_bins.clear()
                self._survey_motion_recorded_bins.clear()
                self._survey_motion_evidence_id = 0
                self._beacon_evidence_seen = False
                self._fresh_uwb_observations.clear()
                self._xyxy_observations.clear()
                self._doa_observations.clear()
                self._last_decision = None
                self._publish_diagnostic_event("SURVEY_RESET_FOR_NEW_MAP")
            if self._state not in {
                "WAIT_MAP", "PLAN_COVERAGE", "PLAN_UWB_FRESH_MAP_SEARCH", "DISABLED"
            }:
                self.get_logger().warning("map changed: cancel current goal and rebuild generic coverage")
                self._cancel_navigation()
                self._transition(
                    "PLAN_UWB_FRESH_MAP_SEARCH"
                    if self._state in {"NAVIGATE_XYXY_MAP_SEARCH", "COLLECT_XYXY_MAP_SEARCH"}
                    else "PLAN_COVERAGE"
                )
        self._map = message

    def _nav_costmap_callback(self, message: Costmap) -> None:
        """Keep the latest live Nav2 global costmap for goal preflight."""

        self._nav_costmap = message
        self._nav_costmap_wait_logged = False

    def _range_callback(self, message: Float32) -> None:
        self._last_range_m = float(message.data)
        self._last_range_received_at = time.monotonic()

    def _survey_uwb_range_valid(self, value: float | None) -> bool:
        if value is None or not math.isfinite(value) or value < 0.0:
            return False
        if not bool(self.get_parameter("initial_map_survey_enabled").value):
            return True
        return value <= float(
            self.get_parameter("survey_uwb_max_valid_range_m").value
        )

    def _collection_measurement_window_open(
        self,
        now: float | None = None,
    ) -> bool:
        """Collect only after Nav2 arrival and the configured settle delay."""

        now = time.monotonic() if now is None else now
        if self._state != "COLLECT" or self._collect_started <= 0.0:
            return False
        elapsed = now - self._collect_started
        settle_sec = float(self.get_parameter("waypoint_settle_sec").value)
        window_sec = float(self.get_parameter("evidence_window_sec").value)
        return settle_sec <= elapsed <= settle_sec + window_sec

    def _survey_navigation_is_moving(self) -> bool:
        return bool(
            self.get_parameter("initial_map_survey_enabled").value
        ) and bool(
            self.get_parameter("navigation_commands_enabled").value
        ) and self._state in {"NAVIGATE_COVERAGE", "ANCHOR_FOCUS"}

    def _survey_motion_bin_key(self, pose: MapPoint) -> tuple[int, int]:
        spacing = max(
            0.10, float(self.get_parameter("survey_motion_bin_m").value)
        )
        return (
            math.floor(pose.x_m / spacing),
            math.floor(pose.y_m / spacing),
        )

    def _survey_motion_bin(self, pose: MapPoint) -> dict[str, object]:
        key = self._survey_motion_bin_key(pose)
        return self._survey_motion_bins.setdefault(
            key,
            {
                "ranges": deque(maxlen=120),
                "paired": deque(maxlen=20),
                "pose": pose,
            },
        )

    def _record_survey_motion_range(
        self,
        now: float,
        range_m: float,
        pose: MapPoint | None,
    ) -> None:
        if not self._survey_navigation_is_moving() or pose is None:
            return
        if bool(self.get_parameter("survey_motion_logging_enabled").value):
            self._write_event("SURVEY_MOVING_UWB", uwb_range_m=range_m)
        if not bool(self.get_parameter("survey_motion_evidence_enabled").value):
            return
        entry = self._survey_motion_bin(pose)
        ranges = entry["ranges"]
        assert isinstance(ranges, deque)
        ranges.append((now, range_m, pose))
        entry["pose"] = pose
        history_sec = float(
            self.get_parameter("survey_motion_bin_history_sec").value
        )
        while ranges and now - float(ranges[0][0]) > history_sec:
            ranges.popleft()

    def _record_survey_motion_packet(
        self,
        now: float,
        metric: dict[str, object],
        pose: MapPoint | None,
    ) -> None:
        if not self._survey_navigation_is_moving() or pose is None:
            return
        if bool(self.get_parameter("survey_motion_logging_enabled").value):
            self._write_event("SURVEY_MOVING_XYXY", packet_metric=metric)
        if not bool(self.get_parameter("survey_motion_evidence_enabled").value):
            return
        direction_level = metric.get("direction_level_dbfs")
        if not isinstance(direction_level, float):
            return

        key = self._survey_motion_bin_key(pose)
        if key in self._survey_motion_recorded_bins:
            return
        entry = self._survey_motion_bin(pose)
        ranges = entry["ranges"]
        paired = entry["paired"]
        assert isinstance(ranges, deque) and isinstance(paired, deque)
        sync_sec = float(
            self.get_parameter("survey_motion_sync_window_sec").value
        )
        fresh = [
            (float(stamp), float(value), range_pose)
            for stamp, value, range_pose in ranges
            if now - float(stamp) <= sync_sec
        ]
        minimum_fresh = int(
            self.get_parameter("survey_motion_min_fresh_samples").value
        )
        if len(fresh) < minimum_fresh:
            return
        aggregate = aggregate_ranges([value for _stamp, value, _pose in fresh])
        if aggregate is None or aggregate.sigma_m > float(
            self.get_parameter("max_range_sigma_m").value
        ):
            return
        paired.append(
            {
                "stamp": now,
                "pose": pose,
                "ranges": tuple((stamp, value) for stamp, value, _pose in fresh),
                "metric": dict(metric),
            }
        )
        history_sec = float(
            self.get_parameter("survey_motion_bin_history_sec").value
        )
        while paired and now - float(paired[0]["stamp"]) > history_sec:
            paired.popleft()
        if len(paired) < int(
            self.get_parameter("survey_motion_min_packets").value
        ):
            return
        self._commit_survey_motion_bin(key, paired)

    def _commit_survey_motion_bin(
        self,
        key: tuple[int, int],
        paired: deque,
    ) -> None:
        """Convert one repeated moving spatial bin into joint evidence."""

        if key in self._survey_motion_recorded_bins or not paired:
            return
        range_by_stamp: dict[float, float] = {}
        metrics: list[dict[str, object]] = []
        poses: list[MapPoint] = []
        for item in paired:
            metrics.append(dict(item["metric"]))
            poses.append(item["pose"])
            for stamp, value in item["ranges"]:
                range_by_stamp[float(stamp)] = float(value)
        aggregate = aggregate_ranges(list(range_by_stamp.values()))
        if aggregate is None:
            return
        direction_levels = [
            float(metric["direction_level_dbfs"])
            for metric in metrics
            if isinstance(metric.get("direction_level_dbfs"), float)
        ]
        if not direction_levels:
            return
        levels = [float(metric["level_dbfs"]) for metric in metrics]
        qualities = [float(metric["quality_db"]) for metric in metrics]
        direction_qualities = [
            float(metric["direction_quality_db"])
            for metric in metrics
            if isinstance(metric.get("direction_quality_db"), float)
        ]
        base_pose = poses[-1]
        sipeed_pose = self._sipeed_pose(base_pose)
        wall_time = self.get_clock().now().nanoseconds / 1e9
        monotonic = time.monotonic()
        self._survey_motion_evidence_id += 1
        record = EvidenceWaypoint(
            waypoint_id=-self._survey_motion_evidence_id,
            pose=base_pose,
            range_m=aggregate.range_m,
            range_sigma_m=aggregate.sigma_m,
            fresh_samples=aggregate.used_count,
            packet_valid_packets=len(metrics),
            packet_locked=True,
            packet_level_dbfs=float(np.median(levels)),
            packet_quality_db=float(np.median(qualities)),
            packet_direction_level_dbfs=float(np.median(direction_levels)),
            packet_direction_quality_db=(
                float(np.median(direction_qualities))
                if direction_qualities else None
            ),
            packet_pose=sipeed_pose,
            observed_wall_time_sec=wall_time,
            observed_monotonic_sec=monotonic,
        )
        self._evidence.append(record)
        self._survey_motion_recorded_bins.add(key)
        self._survey_waypoint_records += 1
        zone_id = self._topology.zone_for_point(base_pose) if self._topology else None
        self._summary_writer.writerow({
            "wall_time_sec": f"{wall_time:.6f}",
            "monotonic_sec": f"{monotonic:.6f}",
            "record_kind": "MOVING_JOINT",
            "waypoint_id": record.waypoint_id,
            "cycle": self._coverage_cycle,
            "map_x_m": f"{base_pose.x_m:.3f}",
            "map_y_m": f"{base_pose.y_m:.3f}",
            "map_yaw_rad": f"{base_pose.yaw_rad:.3f}",
            "sipeed_x_m": f"{sipeed_pose.x_m:.3f}",
            "sipeed_y_m": f"{sipeed_pose.y_m:.3f}",
            "zone_id": zone_id or "",
            "uwb_status": self._uwb_status,
            "uwb_median_m": f"{record.range_m:.3f}",
            "uwb_sigma_m": f"{record.range_sigma_m:.3f}",
            "uwb_samples": record.fresh_samples,
            "packet_valid_packets": record.packet_valid_packets,
            "packet_locked": 1,
            "packet_level_dbfs": f"{record.packet_level_dbfs:.3f}",
            "packet_quality_db": f"{record.packet_quality_db:.3f}",
            "packet_direction_level_dbfs": (
                f"{record.packet_direction_level_dbfs:.3f}"
            ),
            "packet_direction_quality_db": (
                "" if record.packet_direction_quality_db is None
                else f"{record.packet_direction_quality_db:.3f}"
            ),
        })
        self._streams[1].flush()
        self._beacon_evidence_seen = True
        self._publish_diagnostic_event(
            "SURVEY_MOVING_JOINT_RECORDED",
            {
                "spatial_bin": list(key),
                "zone_id": zone_id,
                "base_pose": self._diagnostic_point(base_pose),
                "sipeed_pose": self._diagnostic_point(sipeed_pose),
                "uwb_range_m": round(record.range_m, 4),
                "uwb_sigma_m": round(record.range_sigma_m, 4),
                "fresh_samples": record.fresh_samples,
                "xyxy_packets": record.packet_valid_packets,
                "xyxy_direction_level_dbfs": round(
                    float(record.packet_direction_level_dbfs), 4
                ),
            },
        )
        self.get_logger().info(
            "moving map evidence committed: "
            f"zone={zone_id} UWB={record.range_m:.2f}m "
            f"XYXY={record.packet_direction_level_dbfs:.1f}dBFS"
        )

    def _record_synchronized_fresh_range(
        self,
        now: float,
        range_m: float,
    ) -> None:
        """Record one range only after its matching FRESH status arrives."""

        if self._survey_first_fresh_at is None and self._survey_uwb_range_valid(
            range_m
        ):
            self._survey_first_fresh_at = now
            self._survey_first_fresh_pose = self._robot_pose()
            self._survey_first_fresh_range_m = range_m
            self._publish_diagnostic_event(
                "SURVEY_UWB_FIRST_FRESH",
                {
                    "range_m": round(range_m, 4),
                    "pose": self._diagnostic_point(self._survey_first_fresh_pose),
                },
            )
            self.get_logger().info(
                "map survey acquired first usable UWB FRESH: "
                f"{range_m:.2f} m; continue collecting map/XYXY evidence"
            )

        self._survey_departure_samples.append((now, range_m))
        window_sec = float(
            self.get_parameter("survey_departure_window_sec").value
        )
        while (
            self._survey_departure_samples
            and now - self._survey_departure_samples[0][0] > 2.0 * window_sec
        ):
            self._survey_departure_samples.popleft()
        departure = assess_survey_departure(
            tuple(self._survey_departure_samples),
            previous_best_median_m=self._survey_best_window_range_m,
            window_sec=window_sec,
            minimum_span_sec=float(
                self.get_parameter("survey_departure_min_span_sec").value
            ),
            minimum_samples=int(
                self.get_parameter("survey_departure_min_samples").value
            ),
            required_increase_m=float(
                self.get_parameter("survey_range_increase_m").value
            ),
            max_sigma_m=float(
                self.get_parameter("survey_departure_max_sigma_m").value
            ),
        )
        self._survey_best_window_range_m = departure.best_median_m
        self._survey_departure_assessment = {
            "ready": departure.ready,
            "reason": departure.reason,
            "best_median_m": _round_or_none(departure.best_median_m),
            "window_median_m": _round_or_none(departure.window_median_m),
            "window_sigma_m": _round_or_none(departure.window_sigma_m),
            "increase_m": _round_or_none(departure.increase_m),
            "sample_count": departure.sample_count,
            "span_sec": round(departure.span_sec, 4),
        }
        if (
            self._survey_first_fresh_at is not None
            and not self._survey_range_limit_seen
            and departure.ready
        ):
            self._survey_range_limit_seen = True
            self._survey_range_limit_at = now
            self._publish_diagnostic_event(
                "SURVEY_UWB_RANGE_LIMIT_REACHED",
                {
                    "range_m": _round_or_none(departure.window_median_m),
                    "best_median_m": _round_or_none(departure.best_median_m),
                    "increase_m": _round_or_none(departure.increase_m),
                    "required_increase_m": float(
                        self.get_parameter("survey_range_increase_m").value
                    ),
                    "window_sec": window_sec,
                    "sample_count": departure.sample_count,
                    "sigma_m": _round_or_none(departure.window_sigma_m),
                },
            )
            self.get_logger().info(
                "map survey sustained UWB departure confirmed: "
                f"median={float(departure.window_median_m):.2f}m, "
                f"best={float(departure.best_median_m):.2f}m, "
                f"increase={float(departure.increase_m):.2f}m; select only "
                "after minimum map/XYXY evidence gates pass"
            )

        # A FRESH status above the reliable range can end the survey but must
        # not become a distance anchor or room-fit sample.
        if not self._survey_uwb_range_valid(range_m):
            return
        self._fresh_uwb_values.append((now, range_m))
        self._trim_fresh_uwb_values(now)
        self._record_fresh_uwb_observation(now, range_m)
        self._record_survey_motion_range(now, range_m, self._robot_pose())
        if self._collection_measurement_window_open(now):
            self._record_range(range_m)

    def _status_callback(self, message: String) -> None:
        previous_status = self._uwb_status
        self._uwb_status = message.data.upper()
        now = time.monotonic()
        sync_sec = float(
            self.get_parameter("uwb_recovery_range_status_sync_sec").value
        )
        paired_range = (
            self._last_range_m
            if self._last_range_received_at > 0.0
            and now - self._last_range_received_at <= sync_sec
            and self._last_range_m is not None
            and math.isfinite(self._last_range_m)
            else None
        )
        allow_hold = bool(
            self.get_parameter("uwb_navigation_allow_hold").value
        )
        if self._uwb_status == "FRESH":
            self._last_uwb_usable_at = now
            self._uwb_hold_started_at = None
            if previous_status != "FRESH":
                self._fresh_uwb_started_at = now
                self._fresh_uwb_values.clear()
            if paired_range is not None:
                self._record_synchronized_fresh_range(now, paired_range)
        elif self._uwb_status == "HOLD":
            self._fresh_uwb_started_at = None
            self._fresh_uwb_values.clear()
            if self._uwb_hold_started_at is None:
                self._uwb_hold_started_at = now
            if allow_hold:
                self._last_uwb_usable_at = now
        else:
            # LOST/INVALID must stop navigation immediately; a later HOLD
            # is then timed from its first received status message.
            self._uwb_hold_started_at = None
            self._fresh_uwb_started_at = None
            self._fresh_uwb_values.clear()

        if self._xyxy_map_search_measurement_window_open(now):
            if self._uwb_recovery_start_pose is None:
                # Measure stationarity from the end of the settle delay, not
                # from the instant Nav2 reports success while the base may
                # still be mechanically settling.
                self._uwb_recovery_start_pose = self._robot_pose()
            self._uwb_recovery_samples.append(
                (now, self._uwb_status, paired_range)
            )

    def _xyxy_map_search_measurement_window_open(
        self,
        now: float | None = None,
    ) -> bool:
        """Return true only after Nav2 arrival and the settle delay."""

        now = time.monotonic() if now is None else now
        return (
            self._state == "COLLECT_XYXY_MAP_SEARCH"
            and self._collect_started > 0.0
            and now - self._collect_started
            >= float(self.get_parameter("waypoint_settle_sec").value)
        )

    def _packet_callback(self, message: Bool) -> None:
        self._packet_locked = bool(message.data)

    def _doa_angle_callback(self, message: Float32) -> None:
        angle = float(message.data)
        if not math.isfinite(angle):
            return
        self._latest_doa_angle_rad = _wrap_angle_rad(angle)
        self._latest_doa_angle_at = time.monotonic()

    def _doa_metric_callback(self, message: String) -> None:
        """Retain only finite MUSIC/SRP/TDOA diagnostics for the next angle."""

        try:
            metric = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            self.get_logger().warning("ignored malformed /beacon/doa_metric")
            return
        if not isinstance(metric, dict):
            return
        numeric_keys = (
            "active_tone_snr_db",
            "music_top_peak_margin_db",
            "srp_candidate_diff_deg",
            "tdoa_inlier_pairs",
            "tdoa_total_pairs",
            "cluster_count",
            "cluster_max_pair_diff_deg",
        )
        for key in numeric_keys:
            try:
                value = float(metric[key])
            except (KeyError, TypeError, ValueError):
                return
            if not math.isfinite(value):
                return
        self._latest_doa_metric = metric
        self._latest_doa_metric_at = time.monotonic()

    def _doa_metric_passes(self, metric: dict[str, object] | None) -> bool:
        if metric is None:
            return False
        try:
            return (
                float(metric["active_tone_snr_db"])
                >= float(self.get_parameter("doa_min_active_tone_snr_db").value)
                and float(metric["music_top_peak_margin_db"])
                >= float(self.get_parameter("doa_min_music_peak_margin_db").value)
                and float(metric["srp_candidate_diff_deg"])
                <= float(self.get_parameter("doa_max_srp_candidate_diff_deg").value)
                and int(float(metric["tdoa_inlier_pairs"])) >= 3
                and int(float(metric["cluster_count"])) >= 2
                and float(metric["cluster_max_pair_diff_deg"])
                <= math.degrees(float(
                    self.get_parameter("doa_probe_max_spread_rad").value
                ))
            )
        except (KeyError, TypeError, ValueError):
            return False

    def _doa_stable_callback(self, message: Bool) -> None:
        """Accept new ReSpeaker estimates only during a deliberate probe.

        ``doa_angle_node`` publishes angle first and then ``stable=true`` for
        every finished estimator cycle.  Using that edge prevents the 5 Hz
        map state timer from duplicating the same angle as several samples.
        """

        self._doa_stable = bool(message.data)
        if not self._doa_stable or self._state != "COLLECT_DOA_PROBE":
            return
        if self._latest_doa_angle_rad is None:
            return
        now = time.monotonic()
        if now - self._latest_doa_angle_at > 0.5:
            return
        if (
            now - self._latest_doa_metric_at
            > float(self.get_parameter("doa_metric_max_age_sec").value)
            or not self._doa_metric_passes(self._latest_doa_metric)
        ):
            self._publish_diagnostic_event(
                "DOA_PROBE_METRIC_REJECTED",
                {"metric": self._latest_doa_metric},
            )
            return
        self._doa_probe_angles.append(self._latest_doa_angle_rad)
        self._doa_probe_metrics.append(dict(self._latest_doa_metric or {}))
        self._publish_diagnostic_event(
            "DOA_PROBE_STABLE_SAMPLE",
            {
                "sample_count": len(self._doa_probe_angles),
                "angle_rad": _round_or_none(self._latest_doa_angle_rad),
                "metric": self._latest_doa_metric,
                "target": self._diagnostic_point(self._doa_probe_target),
            },
        )

    def _packet_metric_callback(self, message: String) -> None:
        try:
            metric = json.loads(message.data)
            level = float(metric["level_dbfs"])
            quality = float(metric["quality_db"])
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            self.get_logger().warning("ignored malformed /beacon/packet_metric")
            return
        if not math.isfinite(level) or not math.isfinite(quality):
            return
        # The decoder emits this only after full XYXY validation. Do not
        # reject an older decoder during rolling deployment; the model falls
        # back to the conservative four-tone level for that packet.
        for key in ("direction_level_dbfs", "direction_quality_db"):
            if key not in metric:
                continue
            try:
                value = float(metric[key])
            except (TypeError, ValueError):
                metric.pop(key, None)
                continue
            if math.isfinite(value):
                metric[key] = value
            else:
                metric.pop(key, None)
        direction_level = metric.get("direction_level_dbfs")
        if isinstance(direction_level, float):
            now = time.monotonic()
            self._latest_packet_metric = dict(metric)
            self._latest_xyxy_direction_level = direction_level
            direction_quality = metric.get("direction_quality_db")
            self._latest_xyxy_direction_quality = (
                float(direction_quality)
                if isinstance(direction_quality, float)
                else None
            )
            self._latest_xyxy_direction_at = now
            observation_pose = self._robot_pose()
            if observation_pose is not None:
                self._xyxy_observations.append(
                    (now, direction_level, observation_pose)
                )
                self._trim_xyxy_observations(now)
            # Only samples after the current Nav2 goal began can prove that
            # moving along this particular route is making the signal weaker.
            if now >= self._xyxy_goal_started_at:
                self._xyxy_navigation_samples.append((now, direction_level))
                self._trim_xyxy_navigation_samples(now)
            self._record_survey_motion_packet(now, metric, observation_pose)
        collect_metric = self._collection_measurement_window_open() or (
            self._state == "COLLECT_XYXY_MAP_SEARCH"
            and self._xyxy_map_search_measurement_window_open()
        )
        if collect_metric:
            self._packet_metrics.append(metric)
            self._write_event(
                "PACKET_XYXY_MAP_SEARCH"
                if self._state == "COLLECT_XYXY_MAP_SEARCH"
                else "PACKET",
                packet_metric=metric,
            )

    def _person_pose_callback(self, message: PoseStamped) -> None:
        """Accept only finite, map-frame people from the depth-aware YOLO node."""

        frame_id = message.header.frame_id.lstrip("/")
        if frame_id and frame_id != self._map_frame.lstrip("/"):
            self.get_logger().warning(
                f"ignored person pose in unexpected frame {message.header.frame_id!r}"
            )
            return
        point = message.pose.position
        if not (math.isfinite(point.x) and math.isfinite(point.y)):
            return
        self._latest_person_pose = MapPoint(float(point.x), float(point.y), 0.0)
        self._latest_person_received_at = time.monotonic()

    def _trim_fresh_uwb_values(self, now: float) -> None:
        window_sec = float(self.get_parameter("yolo_uwb_fresh_sec").value)
        while (
            self._fresh_uwb_values
            and now - self._fresh_uwb_values[0][0] > window_sec
        ):
            self._fresh_uwb_values.popleft()

    def _trim_fresh_uwb_observations(self, now: float) -> None:
        """Retain only map positions observed while the UWB link was FRESH."""

        history_sec = float(self.get_parameter("uwb_fresh_history_sec").value)
        while (
            self._fresh_uwb_observations
            and now - self._fresh_uwb_observations[0][0] > history_sec
        ):
            self._fresh_uwb_observations.popleft()

    def _record_fresh_uwb_observation(self, now: float, range_m: float) -> None:
        """Store spatially distinct FRESH UWB anchors for degraded-link search.

        Keeping every 5 Hz range reply would only duplicate a stationary
        point.  A lower fresh distance at the same physical point replaces
        the previous record because it is the more useful map-search anchor.
        """

        pose = self._robot_pose()
        if pose is None:
            return
        self._trim_fresh_uwb_observations(now)
        spacing_m = float(
            self.get_parameter("uwb_fresh_history_min_spacing_m").value
        )
        if self._fresh_uwb_observations:
            previous_time, previous_range, previous_pose = self._fresh_uwb_observations[-1]
            if self._distance(pose, previous_pose) < spacing_m:
                if range_m < previous_range:
                    self._fresh_uwb_observations[-1] = (now, range_m, pose)
                return
        self._fresh_uwb_observations.append((now, range_m, pose))

    def _fresh_uwb_is_stable(self) -> bool:
        """Require a continuous FRESH run before camera or map re-entry."""

        started_at = self._fresh_uwb_started_at
        if self._uwb_status != "FRESH" or started_at is None:
            return False
        if time.monotonic() - started_at < float(
            self.get_parameter("yolo_uwb_fresh_sec").value
        ):
            return False
        return len(self._fresh_uwb_values) >= int(
            self.get_parameter("yolo_uwb_min_fresh_samples").value
        )

    def _fresh_uwb_median_m(self) -> float | None:
        if not self._fresh_uwb_values:
            return None
        values = [value for _stamp, value in self._fresh_uwb_values]
        return float(np.median(values))

    def _survey_joint_evidence(self) -> list[EvidenceWaypoint]:
        """Return waypoints containing both FRESH UWB and validated XYXY."""

        minimum_packets = int(
            self.get_parameter("min_packet_valid_packets").value
        )
        return [
            item
            for item in self._evidence
            if (
                self._survey_uwb_range_valid(item.range_m)
                and item.fresh_samples
                >= int(
                    self.get_parameter(
                        "survey_min_fresh_samples_per_waypoint"
                    ).value
                )
                and item.packet_valid_packets >= minimum_packets
                and packet_audio_level_dbfs(item) is not None
            )
        ]

    def _survey_joint_evidence_count(self) -> int:
        return len(self._survey_joint_evidence())

    def _survey_observed_zone_ids(self) -> set[int]:
        if self._topology is None:
            return set()
        return {
            zone_id
            for item in self._survey_joint_evidence()
            for zone_id in (self._topology.zone_for_point(item.pose),)
            if zone_id is not None
        }

    def _survey_decision_gate_reason(
        self,
        *,
        coverage_complete: bool,
    ) -> str | None:
        """Return why the map survey may choose a room, or ``None``."""

        if not bool(self.get_parameter("initial_map_survey_enabled").value):
            return "LEGACY_EVALUATION"
        if self._topology is None:
            return None
        assessment = assess_survey_gate(
            first_fresh_seen=self._survey_first_fresh_at is not None,
            range_limit_seen=self._survey_range_limit_seen,
            coverage_complete=coverage_complete,
            select_on_coverage_complete=bool(
                self.get_parameter("survey_select_on_coverage_complete").value
            ),
            joint_waypoints=self._survey_joint_evidence_count(),
            configured_min_joint_waypoints=int(
                self.get_parameter("survey_min_joint_waypoints").value
            ),
            evidence_min_waypoints=int(
                self.get_parameter("min_evidence_waypoints").value
            ),
            observed_zones=len(self._survey_observed_zone_ids()),
            configured_min_observed_zones=int(
                self.get_parameter("survey_min_observed_zones").value
            ),
            total_zones=len(self._topology.zones),
        )
        return assessment.reason if assessment.ready else None

    def _survey_extrema_supports_decision(
        self,
        decision: RoomDecision,
    ) -> tuple[bool, dict[str, object]]:
        """Require the shortest UWB and loudest XYXY areas to support a room.

        The robust all-record score still selects the room.  This additional
        gate implements the requested physical interpretation: its zone must
        be the zone of one extrema, while two different extrema zones must be
        connected by a map-derived portal rather than unrelated reflections.
        """

        joint = self._survey_joint_evidence()
        if self._topology is None or not joint:
            return False, {"reason": "no_joint_evidence"}
        shortest = min(joint, key=lambda item: item.range_m)
        loudest = max(
            joint,
            key=lambda item: float(
                packet_audio_level_dbfs(item)
                if packet_audio_level_dbfs(item) is not None
                else -math.inf
            ),
        )
        range_zone = self._topology.zone_for_point(shortest.pose)
        audio_pose = loudest.packet_pose or loudest.pose
        audio_zone = self._topology.zone_for_point(audio_pose)
        selected_zone = decision.zone.zone_id
        candidate_zones = {
            zone_id for zone_id in (range_zone, audio_zone) if zone_id is not None
        }
        adjacent = (
            range_zone == audio_zone
            or any(
                {portal.zone_a, portal.zone_b} == {range_zone, audio_zone}
                for portal in self._topology.portals
            )
        )
        extrema_distance_m = self._distance(shortest.pose, audio_pose)
        extrema_close = extrema_distance_m <= float(
            self.get_parameter("survey_extrema_max_distance_m").value
        )
        supported = (
            selected_zone in candidate_zones
            and bool(adjacent)
            and extrema_close
        )
        payload = {
            "supported": supported,
            "selected_zone_id": selected_zone,
            "shortest_uwb": {
                "range_m": round(shortest.range_m, 4),
                "zone_id": range_zone,
                "pose": self._diagnostic_point(shortest.pose),
                "wall_time_sec": _round_or_none(shortest.observed_wall_time_sec, 6),
            },
            "loudest_xyxy": {
                "level_dbfs": _round_or_none(packet_audio_level_dbfs(loudest)),
                "zone_id": audio_zone,
                "pose": self._diagnostic_point(audio_pose),
                "wall_time_sec": _round_or_none(loudest.observed_wall_time_sec, 6),
            },
            "extrema_zones_adjacent": bool(adjacent),
            "extrema_distance_m": round(extrema_distance_m, 4),
            "extrema_max_distance_m": float(
                self.get_parameter("survey_extrema_max_distance_m").value
            ),
            "extrema_close": extrema_close,
        }
        return supported, payload

    def _continuous_uwb_recovery_assessment(self) -> dict[str, object]:
        """Require three continuous seconds of stable FRESH before exit."""

        return assess_continuous_fresh_recovery(
            tuple(self._fresh_uwb_values),
            fresh_started_at=self._fresh_uwb_started_at,
            now=time.monotonic(),
            required_duration_sec=float(
                self.get_parameter("uwb_recovery_fresh_confirm_sec").value
            ),
            min_samples=int(
                self.get_parameter(
                    "uwb_recovery_fresh_confirm_min_samples"
                ).value
            ),
            max_sigma_m=float(
                self.get_parameter(
                    "uwb_recovery_fresh_confirm_max_sigma_m"
                ).value
            ),
        )

    def _stationary_uwb_recovery_assessment(self) -> dict[str, object]:
        """Evaluate recovery only after one complete stopped measurement."""

        start_pose = self._uwb_recovery_start_pose
        end_pose = self._robot_pose()
        displacement_m = (
            self._distance(start_pose, end_pose)
            if start_pose is not None and end_pose is not None
            else None
        )
        return assess_stationary_uwb_recovery(
            self._uwb_recovery_samples,
            robot_displacement_m=displacement_m,
            min_fresh_samples=int(
                self.get_parameter("uwb_recovery_min_fresh_samples").value
            ),
            min_fresh_ratio=float(
                self.get_parameter("uwb_recovery_min_fresh_ratio").value
            ),
            max_sigma_m=float(
                self.get_parameter("uwb_recovery_max_sigma_m").value
            ),
            max_stationary_motion_m=float(
                self.get_parameter("uwb_recovery_max_stationary_motion_m").value
            ),
            range_motion_margin_m=float(
                self.get_parameter("uwb_recovery_range_motion_margin_m").value
            ),
        )

    def _trim_recent_goal_tabu(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        ttl_sec = float(self.get_parameter("recent_goal_tabu_sec").value)
        while self._recent_goal_tabu and now - self._recent_goal_tabu[0][0] > ttl_sec:
            self._recent_goal_tabu.popleft()
        limit = max(0, int(self.get_parameter("recent_goal_tabu_count").value))
        while len(self._recent_goal_tabu) > limit:
            self._recent_goal_tabu.popleft()

    def _trim_zone_cooldowns(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        ttl_sec = max(0.0, float(self.get_parameter("zone_cooldown_sec").value))
        if ttl_sec <= 0.0:
            self._zone_cooldowns.clear()
            return
        expired = [
            zone_id
            for zone_id, (stamp, _reason, _point) in self._zone_cooldowns.items()
            if now - stamp > ttl_sec
        ]
        for zone_id in expired:
            self._zone_cooldowns.pop(zone_id, None)

    def _mark_zone_cooldown(self, point: MapPoint, reason: str) -> None:
        if self._topology is None:
            return
        self._trim_zone_cooldowns()
        zone_id = self._topology.zone_for_point(point)
        if zone_id is None:
            return
        self._zone_cooldowns[zone_id] = (time.monotonic(), reason, point)
        self._publish_diagnostic_event(
            "ZONE_COOLDOWN_ADDED",
            {
                "zone_id": zone_id,
                "goal": self._diagnostic_point(point),
                "reason": reason,
                "ttl_sec": float(self.get_parameter("zone_cooldown_sec").value),
            },
        )

    def _goal_zone_on_cooldown(self, point: MapPoint) -> bool:
        if self._topology is None:
            return False
        self._trim_zone_cooldowns()
        zone_id = self._topology.zone_for_point(point)
        return zone_id is not None and zone_id in self._zone_cooldowns

    def _remember_recent_goal(
        self,
        point: MapPoint | None,
        reason: str,
        *,
        mark_zone: bool = True,
    ) -> None:
        if point is None:
            return
        if mark_zone:
            self._mark_zone_cooldown(point, reason)
        if int(self.get_parameter("recent_goal_tabu_count").value) <= 0:
            return
        now = time.monotonic()
        self._trim_recent_goal_tabu(now)
        radius_m = float(self.get_parameter("recent_goal_tabu_radius_m").value)
        retained = deque(
            item for item in self._recent_goal_tabu
            if self._distance(item[1], point) >= radius_m
        )
        self._recent_goal_tabu = retained
        self._recent_goal_tabu.append((now, point, reason))
        self._trim_recent_goal_tabu(now)
        self._publish_diagnostic_event(
            "RECENT_GOAL_TABU_ADDED",
            {"goal": self._diagnostic_point(point), "reason": reason},
        )

    def _goal_is_recent_tabu(self, point: MapPoint) -> bool:
        self._trim_recent_goal_tabu()
        radius_m = float(self.get_parameter("recent_goal_tabu_radius_m").value)
        return any(
            self._distance(point, recent_point) < radius_m
            for _stamp, recent_point, _reason in self._recent_goal_tabu
        )

    def _trim_xyxy_observations(self, now: float) -> None:
        # Keep diagnostic XYXY pose history for the same bounded interval as
        # the UWB FRESH anchor history. It no longer drives a one-shot return
        # goal after UWB loss.
        history_sec = float(self.get_parameter("uwb_fresh_history_sec").value)
        while (
            self._xyxy_observations
            and now - self._xyxy_observations[0][0] > history_sec
        ):
            self._xyxy_observations.popleft()

    def _recent_xyxy_available(self, max_age_sec: float) -> bool:
        return (
            self._latest_xyxy_direction_level is not None
            and time.monotonic() - self._latest_xyxy_direction_at <= max_age_sec
        )

    def _yolo_verify_gate_ready(self) -> bool:
        """Gate camera verification with recent XYXY and stable near UWB."""

        if not bool(self.get_parameter("yolo_xyxy_gate_enabled").value):
            return False
        if self._mission_complete or time.monotonic() < self._yolo_verify_rearm_at:
            return False
        if self._state not in {
            "PLAN_COVERAGE",
            "NAVIGATE_COVERAGE",
            "ANCHOR_FOCUS",
            "COLLECT",
            "EVALUATE",
            "NAVIGATE_DOA_PROBE",
            "COLLECT_DOA_PROBE",
            "NAVIGATE_ENTRY",
            "NAVIGATE_INTERIOR",
            "DOA_ACTIVE",
        }:
            return False
        if not self._recent_xyxy_available(float(
            self.get_parameter("yolo_xyxy_metric_max_age_sec").value
        )):
            return False
        if not self._packet_locked:
            return False
        if not self._fresh_uwb_is_stable():
            return False
        range_m = self._fresh_uwb_median_m()
        return (
            range_m is not None
            and 0.0 <= range_m
            <= float(self.get_parameter("yolo_activation_uwb_range_m").value)
        )

    def _uwb_link_usable(self) -> bool:
        """Return whether UWB is presently communicating, not merely cached."""

        allow_hold = bool(
            self.get_parameter("uwb_navigation_allow_hold").value
        )
        if self._uwb_status == "HOLD" and not allow_hold:
            hold_started_at = self._uwb_hold_started_at
            hold_timeout = float(
                self.get_parameter("uwb_navigation_hold_timeout_sec").value
            )
            return (
                hold_started_at is not None
                and time.monotonic() - hold_started_at < hold_timeout
            )
        allowed_statuses = {"FRESH", "HOLD"} if allow_hold else {"FRESH"}
        if self._uwb_status not in allowed_statuses:
            return False
        timeout = float(
            self.get_parameter("uwb_navigation_loss_timeout_sec").value
        )
        return (
            self._last_uwb_usable_at > 0.0
            and time.monotonic() - self._last_uwb_usable_at <= timeout
        )

    def _uwb_navigation_guard_expired(self) -> bool:
        """Enter XYXY-priority map search after UWB is no longer live.

        HOLD remains tolerated for ``uwb_navigation_hold_timeout_sec`` (five
        seconds in the robot YAML).  Afterwards every map-search decision is
        ranked by validated XYXY amplitude until continuous FRESH recovery is
        confirmed.  The gate still requires previously validated joint
        evidence, so startup silence cannot launch an acoustic search.
        """

        return (
            self._beacon_evidence_seen
            and self._state in {
                "NAVIGATE_COVERAGE",
                "ANCHOR_FOCUS",
                "COLLECT",
                "NAVIGATE_DOA_PROBE",
                "NAVIGATE_ENTRY",
                "NAVIGATE_INTERIOR",
            }
            and not self._uwb_link_usable()
        )

    def _trim_xyxy_navigation_samples(self, now: float) -> None:
        """Retain only the recent moving-route XYXY strength window."""

        window_sec = float(self.get_parameter("xyxy_navigation_window_sec").value)
        while (
            self._xyxy_navigation_samples
            and now - self._xyxy_navigation_samples[0][0] > window_sec
        ):
            self._xyxy_navigation_samples.popleft()

    def _xyxy_navigation_guard_expired(self) -> bool:
        """Return true for a sustained, route-worsening XYXY observation.

        A single packet can be changed by body rotation or reflection. The
        guard therefore needs at least two *fully validated* XYXY packets,
        evaluates their 900/1050 Hz median, requires robot travel from the
        goal start, and keeps the degradation condition for a further hold.
        """

        if not bool(self.get_parameter("xyxy_navigation_guard_enabled").value):
            return False
        if not self._beacon_evidence_seen:
            return False
        if bool(self.get_parameter("initial_map_survey_enabled").value):
            if self._state in {"NAVIGATE_COVERAGE", "ANCHOR_FOCUS"}:
                return False
            if (
                self._state in {"NAVIGATE_ENTRY", "NAVIGATE_INTERIOR"}
                and not self._survey_room_approach_active
            ):
                return False
        if self._state not in {
            "NAVIGATE_COVERAGE",
            "ANCHOR_FOCUS",
            "NAVIGATE_XYXY_MAP_SEARCH",
            "NAVIGATE_DOA_PROBE",
            "NAVIGATE_ENTRY",
            "NAVIGATE_INTERIOR",
        }:
            return False
        if self._xyxy_goal_reference_level is None or self._xyxy_goal_start_pose is None:
            return False

        now = time.monotonic()
        self._trim_xyxy_navigation_samples(now)
        required_samples = int(
            self.get_parameter("xyxy_navigation_min_samples").value
        )
        if len(self._xyxy_navigation_samples) < required_samples:
            self._xyxy_degradation_started_at = None
            return False

        robot = self._robot_pose()
        if robot is None:
            return False
        travelled_m = self._distance(robot, self._xyxy_goal_start_pose)
        recovery_route = self._state == "NAVIGATE_XYXY_MAP_SEARCH"
        required_travel_m = float(self.get_parameter(
            "xyxy_priority_route_min_travel_m"
            if recovery_route else "xyxy_navigation_min_travel_m"
        ).value)
        if travelled_m < required_travel_m:
            self._xyxy_degradation_started_at = None
            return False

        recent_level = float(np.median([
            level for _stamp, level in self._xyxy_navigation_samples
        ]))
        drop_db = self._xyxy_goal_reference_level - recent_level
        required_drop_db = float(self.get_parameter(
            "xyxy_priority_route_drop_db"
            if recovery_route else "xyxy_navigation_drop_db"
        ).value)
        if drop_db < required_drop_db:
            self._xyxy_degradation_started_at = None
            return False

        if self._xyxy_degradation_started_at is None:
            self._xyxy_degradation_started_at = now
            self.get_logger().warning(
                "XYXY route signal degrading: "
                f"reference={self._xyxy_goal_reference_level:.1f} dBFS, "
                f"recent={recent_level:.1f} dBFS, drop={drop_db:.1f} dB, "
                f"travel={travelled_m:.2f} m; holding for confirmation"
            )
            return False

        required_hold_sec = float(self.get_parameter(
            "xyxy_priority_route_hold_sec"
            if recovery_route else "xyxy_navigation_hold_sec"
        ).value)
        return now - self._xyxy_degradation_started_at >= required_hold_sec

    def _reject_decreasing_xyxy_map_search_route(self) -> None:
        """Cancel a degraded-link route that is measurably getting quieter."""

        target = self._current_waypoint
        source = None
        if self._xyxy_map_search_index < len(self._xyxy_map_search_queue):
            queued_target, source, _range_m = self._xyxy_map_search_queue[
                self._xyxy_map_search_index
            ]
            target = target or queued_target

        recent_level = (
            float(np.median([
                level for _stamp, level in self._xyxy_navigation_samples
            ]))
            if self._xyxy_navigation_samples else None
        )
        stop_pose = self._robot_pose()
        payload = {
            "index": self._xyxy_map_search_index,
            "source": source,
            "rejected_goal": self._diagnostic_point(target),
            "stop_pose": self._diagnostic_point(stop_pose),
            "reference_level_dbfs": _round_or_none(
                self._xyxy_goal_reference_level
            ),
            "recent_level_dbfs": _round_or_none(recent_level),
            "drop_db": (
                None
                if self._xyxy_goal_reference_level is None
                or recent_level is None
                else round(self._xyxy_goal_reference_level - recent_level, 4)
            ),
        }
        self._cancel_navigation()
        self._nav_result = None
        self._remember_recent_goal(
            target,
            "XYXY_PRIORITY_ROUTE_DECREASED",
            mark_zone=False,
        )
        self._current_waypoint = None
        self._xyxy_map_search_index += 1
        self._xyxy_degradation_started_at = None
        self._xyxy_goal_reference_level = None
        self._xyxy_goal_start_pose = None
        self._xyxy_navigation_samples.clear()
        self._publish_diagnostic_event(
            "XYXY_PRIORITY_ROUTE_REJECTED_DECREASING", payload
        )
        self.get_logger().warning(
            "UWB degraded and full-XYXY level decreased during travel: "
            "cancel current map goal and choose the next louder map-safe "
            "candidate"
        )

    def _hold_for_xyxy_degradation(self) -> None:
        """Stop a route that has persistently weakened validated XYXY audio."""

        previous_state = self._state
        stop_pose = self._robot_pose()
        self._cancel_navigation()
        self._nav_result = None
        self._xyxy_degradation_started_at = None
        self._xyxy_goal_reference_level = None
        self._xyxy_goal_start_pose = None
        self._xyxy_navigation_samples.clear()

        # The forward candidate has been disproved. Do not immediately retry
        # it; instead, gather a fresh stationary UWB/XYXY record at the safe
        # stopping point and let normal map evidence re-rank the sections.
        if previous_state in {"NAVIGATE_COVERAGE", "ANCHOR_FOCUS"}:
            if self._coverage_index < len(self._coverage):
                self._coverage_index += 1
        else:
            rejected_goal = None
            if self._last_decision is not None:
                rejected_goal = (
                    self._last_decision.staging_pose
                    if previous_state == "NAVIGATE_ENTRY"
                    else self._last_decision.interior_pose
                )
            self._remember_recent_goal(
                rejected_goal,
                "SELECTED_ROUTE_XYXY_DROP",
            )
            self._last_decision = None
            self._survey_room_approach_active = False
            self._survey_selection_ready = False
            self._survey_selection_reason = None
            self._publish_doa_enabled(False)
            self._publish_yolo_ready(False)

        if stop_pose is None:
            self._current_waypoint = None
            self._xyxy_degradation_resume_state = "PLAN_COVERAGE"
        else:
            self._current_waypoint = stop_pose
            self._current_waypoint_id += 1
            self._xyxy_degradation_resume_state = "COLLECT"

        self._write_event("XYXY_DEGRADING_NAVIGATION")
        self._transition("XYXY_DEGRADING_HOLD")
        self.get_logger().warning(
            "validated XYXY 900/1050 signal stayed weaker while moving: "
            "Nav2 goal cancelled; stopping to re-measure evidence"
        )

    def _advance_xyxy_degrading_hold(self) -> None:
        """Collect new stationary evidence before choosing another route."""

        required_delay = float(
            self.get_parameter("xyxy_navigation_recheck_delay_sec").value
        )
        if time.monotonic() - self._state_since < required_delay:
            return
        resume_state = self._xyxy_degradation_resume_state or "PLAN_COVERAGE"
        self._xyxy_degradation_resume_state = None
        self.get_logger().info(
            f"XYXY route stop settled; resume map evidence state {resume_state}"
        )
        if resume_state == "COLLECT":
            self._begin_collection()
        else:
            self._transition(resume_state)

    def _hold_for_uwb_loss(self) -> None:
        """Switch to map-safe XYXY comparison when UWB stops being live.

        HOLD/LOST is not allowed to keep a stale UWB value driving Nav2, but
        it also must not strand the robot.  The recovery planner uses only
        map-safe targets: first prior FRESH UWB positions, then generic map
        coverage targets around those positions.  XYXY is measured at each
        stop and is never converted into an unmeasured acoustic bearing.
        """

        previous_state = self._state
        interrupted_waypoint = self._current_waypoint
        self._cancel_navigation()
        self._nav_result = None
        self._remember_recent_goal(
            interrupted_waypoint,
            "UWB_LOSS_CANCELLED",
            mark_zone=False,
        )
        if previous_state in {"NAVIGATE_COVERAGE", "ANCHOR_FOCUS"}:
            if self._coverage_index < len(self._coverage):
                self._coverage_index += 1
            self._uwb_loss_resume_state = (
                previous_state
                if self._coverage_index < len(self._coverage)
                else "EVALUATE"
            )
        else:
            # Entry/DOA/interior decisions may have become stale while the
            # radio path was lost. Re-evaluate accumulated map evidence after
            # recovery instead of recreating coverage from index zero.
            self._uwb_loss_resume_state = "EVALUATE"
        self._current_waypoint = None
        self._last_decision = None
        self._publish_doa_enabled(False)
        self._publish_yolo_ready(False)
        self._doa_probe_target = None
        self._doa_probe_angles = []
        self._uwb_degraded_started_at = time.monotonic()
        self._xyxy_map_search_queue = []
        self._xyxy_map_search_index = 0
        self._xyxy_active_goal_index = None
        self._xyxy_active_goal_point = None
        self._write_event("UWB_LOST_NAVIGATION")
        self._transition("PLAN_UWB_FRESH_MAP_SEARCH")
        self.get_logger().warning(
            "UWB HOLD/LOST persisted after validated XYXY: Nav2 goal cancelled; "
            "start map-safe XYXY-amplitude-priority search and keep it active "
            "until UWB FRESH remains stable for three continuous seconds"
        )

    def _resume_after_fresh_uwb(self, assessment: dict[str, object]) -> None:
        """Resume prior map progress after stationary UWB quality checks."""

        self._cancel_navigation()
        self._nav_result = None
        self._current_waypoint = None
        resume_state = self._uwb_loss_resume_state or "EVALUATE"
        self._uwb_loss_resume_state = None
        self._uwb_degraded_started_at = None
        self._xyxy_map_search_queue = []
        self._xyxy_map_search_index = 0
        self._xyxy_active_goal_index = None
        self._xyxy_active_goal_point = None
        self._uwb_recovery_samples = []
        self._uwb_recovery_start_pose = None
        if resume_state in {"NAVIGATE_COVERAGE", "ANCHOR_FOCUS"} and (
            not self._coverage or self._coverage_index >= len(self._coverage)
        ):
            resume_state = "EVALUATE"
        self.get_logger().info(
            "stationary UWB recovery accepted; leave XYXY map comparison and "
            f"resume {resume_state} without resetting coverage progress"
        )
        self._publish_diagnostic_event("UWB_FRESH_RECOVERED", assessment)
        self._transition(resume_state)

    def _fresh_uwb_anchor_candidates(
        self,
        robot: MapPoint,
    ) -> list[tuple[float, float, MapPoint]]:
        """Return spatially distinct historic FRESH anchors, lowest range first."""

        now = time.monotonic()
        self._trim_fresh_uwb_observations(now)
        spacing_m = float(
            self.get_parameter("uwb_fresh_history_min_spacing_m").value
        )
        ranked = sorted(
            self._fresh_uwb_observations,
            key=lambda item: (item[1], self._distance(robot, item[2]), -item[0]),
        )
        selected: list[tuple[float, float, MapPoint]] = []
        for item in ranked:
            if all(self._distance(item[2], existing[2]) >= spacing_m for existing in selected):
                selected.append(item)
        return selected

    def _xyxy_priority_observations(
        self,
    ) -> list[tuple[float, float, MapPoint]]:
        """Return spatially consolidated full-XYXY amplitude observations."""

        now = time.monotonic()
        self._trim_xyxy_observations(now)
        spacing = max(
            0.10,
            float(self.get_parameter("xyxy_priority_history_bin_m").value),
        )
        grouped: dict[tuple[int, int], list[tuple[float, float, MapPoint]]] = {}
        for stamp, level, point in self._xyxy_observations:
            if self._topology is not None and not self._topology.safe(point):
                continue
            key = (
                math.floor(point.x_m / spacing),
                math.floor(point.y_m / spacing),
            )
            grouped.setdefault(key, []).append((stamp, level, point))

        result: list[tuple[float, float, MapPoint]] = []
        for values in grouped.values():
            levels = [item[1] for item in values]
            representative = max(values, key=lambda item: item[0])
            result.append(
                (
                    representative[0],
                    float(np.median(levels)),
                    representative[2],
                )
            )
        result.sort(key=lambda item: (item[1], item[0]), reverse=True)
        return result

    @staticmethod
    def _expected_xyxy_level(
        point: MapPoint,
        observations: list[tuple[float, float, MapPoint]],
    ) -> float | None:
        """Inverse-distance estimate used only to rank map-safe candidates."""

        if not observations:
            return None
        nearest = sorted(
            observations,
            key=lambda item: math.hypot(
                point.x_m - item[2].x_m,
                point.y_m - item[2].y_m,
            ),
        )[:6]
        weighted = 0.0
        weight_sum = 0.0
        for _stamp, level, observed in nearest:
            distance_m = math.hypot(
                point.x_m - observed.x_m,
                point.y_m - observed.y_m,
            )
            weight = 1.0 / max(0.20, distance_m) ** 2
            weighted += weight * level
            weight_sum += weight
        return weighted / weight_sum if weight_sum > 0.0 else None

    def _plan_uwb_fresh_map_search(self) -> None:
        """Build one safe XYXY comparison route around historic FRESH points."""

        if not bool(self.get_parameter("uwb_fresh_map_search_enabled").value):
            self.get_logger().warning(
                "UWB map search is disabled; wait for UWB FRESH before normal coverage"
            )
            self._transition("UWB_LOST_HOLD")
            return
        if self._topology is None or not self._coverage:
            # The next normal planning tick rebuilds topology from the map. It
            # does not send a goal before this degraded-search planner runs.
            self._transition("PLAN_COVERAGE")
            return
        robot = self._robot_pose()
        if robot is None:
            return

        anchors = [
            item for item in self._fresh_uwb_anchor_candidates(robot)
            if self._topology.safe(item[2])
        ]
        queue: list[tuple[MapPoint, str, float | None]] = []
        xyxy_priority = self._xyxy_priority_observations()
        priority_reference = self._latest_xyxy_direction_level
        self._xyxy_priority_reference_level = priority_reference
        priority_gain = float(
            self.get_parameter("xyxy_priority_min_gain_db").value
        )
        priority_bin = max(
            0.10,
            float(self.get_parameter("xyxy_priority_history_bin_m").value),
        )
        tabu_skipped = 0
        zone_cooldown_skipped = 0

        def append_candidate(
            point: MapPoint,
            source: str,
            anchor_range_m: float | None,
            *,
            ignore_zone_cooldown: bool = False,
            ignore_xyxy_decrease: bool = False,
        ) -> None:
            nonlocal tabu_skipped, zone_cooldown_skipped
            if not self._topology.safe(point):
                return
            if self._distance(point, robot) < float(
                self.get_parameter(
                    "uwb_fresh_map_search_min_goal_distance_m"
                ).value
            ):
                return
            nearby_levels = [
                level
                for _stamp, level, observed in xyxy_priority
                if self._distance(point, observed) <= priority_bin
            ]
            if (
                not ignore_xyxy_decrease
                and
                priority_reference is not None
                and nearby_levels
                and float(np.median(nearby_levels))
                < priority_reference + priority_gain
            ):
                # Never deliberately choose a branch already measured weaker
                # than the packet level at which HOLD recovery started.
                return
            if self._goal_is_recent_tabu(point):
                tabu_skipped += 1
                return
            if not ignore_zone_cooldown and self._goal_zone_on_cooldown(point):
                zone_cooldown_skipped += 1
                return
            minimum_spacing = float(
                self.get_parameter("uwb_fresh_map_search_min_goal_distance_m").value
            )
            if any(self._distance(point, existing[0]) < minimum_spacing for existing in queue):
                return
            queue.append((point, source, anchor_range_m))

        # Full-XYXY amplitude is the first-priority detection list while UWB
        # is degraded.  These are physically visited poses and are sorted from
        # loudest to weakest before any range-only anchor or unmeasured map
        # candidate is appended.
        for _stamp, level, point in xyxy_priority:
            if (
                priority_reference is not None
                and level < priority_reference + priority_gain
            ):
                continue
            append_candidate(
                point,
                "XYXY_HISTORY_STRONG",
                None,
                ignore_zone_cooldown=True,
            )

        # A historical FRESH pose is a physically visited point and is more
        # authoritative than a zone cooldown created by the later UWB loss.
        # The point-level tabu still blocks the exact interrupted goal, while
        # older FRESH poses in the same zone remain eligible for recovery.
        for _stamp, range_m, point in anchors:
            append_candidate(
                point,
                "UWB_FRESH_ANCHOR",
                range_m,
                ignore_zone_cooldown=True,
            )

        # Continue only with map points close to a measured FRESH anchor.  The
        # old implementation appended the entire coverage lattice; when a
        # zone cooldown removed all anchors it selected a distant room and
        # labelled it MAP_COVERAGE_NEAR_FRESH even though its range was None.
        # Every fallback below is now tied to its nearest measured anchor.
        if anchors:
            coverage_radius_m = float(
                self.get_parameter(
                    "uwb_fresh_map_search_coverage_radius_m"
                ).value
            )
            ranked_coverage: list[tuple[float, float, float, MapPoint]] = []
            for point in self._coverage:
                nearest = min(
                    anchors,
                    key=lambda item: self._distance(point, item[2]),
                )
                anchor_distance_m = self._distance(point, nearest[2])
                if anchor_distance_m > coverage_radius_m:
                    continue
                expected_level = self._expected_xyxy_level(
                    point, xyxy_priority
                )
                ranked_coverage.append((
                    -math.inf if expected_level is None else expected_level,
                    nearest[1],
                    anchor_distance_m,
                    point,
                ))
            ranked_coverage.sort(
                key=lambda item: (
                    -item[0],
                    item[1] + 0.50 * item[2],
                )
            )
            for _expected, nearest_range_m, _distance_m, point in ranked_coverage:
                append_candidate(
                    point,
                    "MAP_COVERAGE_NEAR_FRESH",
                    nearest_range_m,
                )
            if not queue and ranked_coverage:
                _expected, nearest_range_m, _distance_m, point = ranked_coverage[0]
                append_candidate(
                    point,
                    "XYXY_MAX_EXPECTED_FALLBACK",
                    nearest_range_m,
                    ignore_zone_cooldown=True,
                    ignore_xyxy_decrease=True,
                )
        else:
            self.get_logger().warning(
                "UWB degraded before any usable FRESH map anchor was saved; "
                "do not start an unrestricted XYXY recovery route"
            )
            self._publish_diagnostic_event(
                "XYXY_MAP_SEARCH_WAITING_FOR_FRESH_ANCHOR",
                {"fresh_anchor_count": 0},
            )
            self._transition("UWB_LOST_HOLD")
            return

        max_waypoints = int(
            self.get_parameter("uwb_fresh_map_search_max_waypoints").value
        )
        if max_waypoints > 0:
            queue = queue[:max_waypoints]
        if not queue:
            if zone_cooldown_skipped > 0:
                self.get_logger().warning(
                    "all currently safe UWB/XYXY map-search zones are cooling down; "
                    "wait before rebuilding the route"
                )
                self._publish_diagnostic_event(
                    "XYXY_MAP_SEARCH_ZONE_COOLDOWN_WAIT",
                    {"zone_cooldown_skipped": zone_cooldown_skipped},
                )
                self._transition("UWB_ZONE_COOLDOWN_WAIT")
            else:
                self.get_logger().error(
                    "UWB degraded: map contains no safe XYXY comparison waypoint"
                )
                self._transition("WAIT_MAP")
            return

        self._xyxy_map_search_queue = queue
        self._xyxy_map_search_index = 0
        self._xyxy_active_goal_index = None
        self._xyxy_active_goal_point = None
        self._current_waypoint = None
        self._xyxy_map_search_cycle += 1
        self._publish_diagnostic_event(
            "XYXY_MAP_SEARCH_PLANNED",
            {
                "cycle": self._xyxy_map_search_cycle,
                "fresh_anchor_count": len(anchors),
                "xyxy_priority_observations": len(xyxy_priority),
                "xyxy_priority_reference_level_dbfs": _round_or_none(
                    priority_reference
                ),
                "queue_size": len(queue),
                "xyxy_priority_candidates": sum(
                    1 for _point, source, _range in queue
                    if source == "XYXY_HISTORY_STRONG"
                ),
                "anchor_candidates": sum(
                    1 for _point, source, _range in queue
                    if source == "UWB_FRESH_ANCHOR"
                ),
                "near_anchor_coverage_candidates": sum(
                    1 for _point, source, _range in queue
                    if source == "MAP_COVERAGE_NEAR_FRESH"
                ),
                "recent_tabu_skipped": tabu_skipped,
                "zone_cooldown_skipped": zone_cooldown_skipped,
            },
        )
        self.get_logger().warning(
            "UWB degraded map search "
            f"cycle {self._xyxy_map_search_cycle}: "
            f"{len(xyxy_priority)} XYXY-priority detections, "
            f"{len(anchors)} FRESH anchors, {len(queue)} map-safe comparison points"
        )
        self._transition("NAVIGATE_XYXY_MAP_SEARCH")

    def _advance_uwb_zone_cooldown_wait(self) -> None:
        """Pause safely until at least one recently visited zone is eligible."""

        self._trim_zone_cooldowns()
        if not self._zone_cooldowns:
            self._transition("PLAN_UWB_FRESH_MAP_SEARCH")
            return
        now = time.monotonic()
        if now - self._last_zone_cooldown_wait_log >= 5.0:
            ttl_sec = float(self.get_parameter("zone_cooldown_sec").value)
            remaining = min(
                max(0.0, ttl_sec - (now - stamp))
                for stamp, _reason, _point in self._zone_cooldowns.values()
            )
            self._last_zone_cooldown_wait_log = now
            self.get_logger().info(
                f"map zone cooldown active; retry degraded search in about {remaining:.1f}s"
            )

    def _begin_xyxy_map_search_collection(self) -> None:
        """Collect only complete XYXY packet amplitudes at one safe stop."""

        if (
            self._nav_goal_handle is not None
            or self._nav_goal_pending
            or self._path_preflight_pending
        ):
            self._publish_diagnostic_event(
                "XYXY_MAP_SEARCH_COLLECTION_REFUSED_WHILE_MOVING",
                {"index": self._xyxy_map_search_index},
            )
            return
        self._range_values = []
        self._packet_metrics = []
        self._uwb_recovery_samples = []
        # This is populated only after waypoint_settle_sec by the first UWB
        # status callback in the real measurement window.
        self._uwb_recovery_start_pose = None
        self._collect_started = time.monotonic()
        self._transition("COLLECT_XYXY_MAP_SEARCH")
        self.get_logger().info(
            "UWB degraded: collect full XYXY 900/1050-Hz amplitude at "
            f"map-search point {self._xyxy_map_search_index + 1}/"
            f"{len(self._xyxy_map_search_queue)}"
        )

    def _advance_xyxy_map_search_navigation(self) -> None:
        """Visit every costmap/path-approved fallback point; never freeze."""

        if self._nav_result is not None:
            result = self._nav_result
            result_index = self._xyxy_active_goal_index
            result_point = self._xyxy_active_goal_point
            self._nav_result = None
            self._xyxy_active_goal_index = None
            self._xyxy_active_goal_point = None
            self._goal_signature = None
            self._path_preflight_passed_signature = None
            if (
                result_index is None
                or result_point is None
                or result_index != self._xyxy_map_search_index
                or result_index >= len(self._xyxy_map_search_queue)
            ):
                self._publish_diagnostic_event(
                    "XYXY_MAP_SEARCH_STALE_NAV_RESULT",
                    {
                        "result": result,
                        "result_index": result_index,
                        "current_index": self._xyxy_map_search_index,
                        "result_point": self._diagnostic_point(result_point),
                    },
                )
                return
            planned_point, source, _range_m = self._xyxy_map_search_queue[
                result_index
            ]
            if self._distance(planned_point, result_point) > 0.05:
                self._publish_diagnostic_event(
                    "XYXY_MAP_SEARCH_STALE_NAV_RESULT",
                    {
                        "result": result,
                        "reason": "bound_point_changed",
                        "result_index": result_index,
                        "planned_point": self._diagnostic_point(planned_point),
                        "result_point": self._diagnostic_point(result_point),
                    },
                )
                return
            if result == "FAILED":
                self.get_logger().warning(
                    f"XYXY map-search {source} goal failed; skip "
                    f"({planned_point.x_m:.2f}, {planned_point.y_m:.2f})"
                )
                self._publish_diagnostic_event(
                    "XYXY_MAP_SEARCH_GOAL_SKIPPED",
                    {
                        "index": result_index,
                        "source": source,
                        "goal": self._diagnostic_point(planned_point),
                    },
                )
                self._current_waypoint = None
                self._xyxy_map_search_index += 1
                return
            if result == "SUCCEEDED":
                self._current_waypoint = planned_point
                self._publish_diagnostic_event(
                    "XYXY_MAP_SEARCH_GOAL_CONFIRMED",
                    {
                        "index": result_index,
                        "source": source,
                        "goal": self._diagnostic_point(planned_point),
                        "actual_pose": self._diagnostic_point(self._robot_pose()),
                    },
                )
                self._begin_xyxy_map_search_collection()
                return

        # A waypoint cannot be changed or considered reached while its path
        # preflight or NavigateToPose action is still outstanding.
        if (
            self._nav_goal_handle is not None
            or self._nav_goal_pending
            or self._path_preflight_pending
        ):
            return

        while (
            self._xyxy_map_search_index < len(self._xyxy_map_search_queue)
            and self._goal_is_recent_tabu(
                self._xyxy_map_search_queue[self._xyxy_map_search_index][0]
            )
        ):
            target, source, _range_m = self._xyxy_map_search_queue[
                self._xyxy_map_search_index
            ]
            self._publish_diagnostic_event(
                "XYXY_MAP_SEARCH_RECENT_GOAL_SKIPPED",
                {
                    "index": self._xyxy_map_search_index,
                    "source": source,
                    "goal": self._diagnostic_point(target),
                },
            )
            self._xyxy_map_search_index += 1
        if self._xyxy_map_search_index >= len(self._xyxy_map_search_queue):
            self.get_logger().warning(
                "UWB still degraded after one complete XYXY map-search pass; "
                "rebuild the map-safe queue"
            )
            self._publish_diagnostic_event("XYXY_MAP_SEARCH_PASS_COMPLETE")
            self._transition("PLAN_UWB_FRESH_MAP_SEARCH")
            return
        target, source, anchor_range_m = self._xyxy_map_search_queue[
            self._xyxy_map_search_index
        ]
        self._current_waypoint = target
        if (
            self._xyxy_active_goal_index != self._xyxy_map_search_index
            or self._xyxy_active_goal_point is None
            or self._distance(self._xyxy_active_goal_point, target) > 0.05
        ):
            self._current_waypoint_id += 1
            self.get_logger().info(
                f"XYXY map-search {source}: ({target.x_m:.2f}, {target.y_m:.2f}), "
                f"prior FRESH UWB={anchor_range_m}"
            )
        self._send_goal(target, "XYXY_MAP_SEARCH")
        # Bind even while the asynchronous path preflight is pending.  Any
        # FAILED/SUCCEEDED result is accepted only for this exact queue item.
        self._xyxy_active_goal_index = self._xyxy_map_search_index
        self._xyxy_active_goal_point = target

    def _finish_xyxy_map_search_collection(self) -> None:
        """Record an amplitude comparison, then move to the next map point."""

        required = float(self.get_parameter("waypoint_settle_sec").value) + float(
            self.get_parameter("evidence_window_sec").value
        )
        if time.monotonic() - self._collect_started < required:
            return
        planned_waypoint = self._current_waypoint
        if planned_waypoint is None or self._xyxy_map_search_index >= len(self._xyxy_map_search_queue):
            self._transition("NAVIGATE_XYXY_MAP_SEARCH")
            return
        actual_waypoint = self._robot_pose()
        if actual_waypoint is None:
            self._publish_diagnostic_event(
                "XYXY_MAP_SEARCH_SAMPLE_REJECTED",
                {
                    "reason": "actual_robot_pose_unavailable",
                    "planned_point": self._diagnostic_point(planned_waypoint),
                },
            )
            self._current_waypoint = None
            self._xyxy_map_search_index += 1
            self._transition("NAVIGATE_XYXY_MAP_SEARCH")
            return
        _target, source, anchor_range_m = self._xyxy_map_search_queue[
            self._xyxy_map_search_index
        ]
        levels = [float(metric["level_dbfs"]) for metric in self._packet_metrics]
        qualities = [float(metric["quality_db"]) for metric in self._packet_metrics]
        direction_levels = [
            float(metric["direction_level_dbfs"])
            for metric in self._packet_metrics
            if "direction_level_dbfs" in metric
        ]
        direction_qualities = [
            float(metric["direction_quality_db"])
            for metric in self._packet_metrics
            if "direction_quality_db" in metric
        ]
        direction_level = (
            float(np.median(direction_levels)) if direction_levels else None
        )
        if direction_level is not None:
            self._last_stationary_xyxy_level = direction_level
        sample = {
            "cycle": self._xyxy_map_search_cycle,
            "index": self._xyxy_map_search_index,
            "source": source,
            "map_point": self._diagnostic_point(actual_waypoint),
            "planned_map_point": self._diagnostic_point(planned_waypoint),
            "prior_fresh_range_m": _round_or_none(anchor_range_m),
            "packet_valid_packets": len(self._packet_metrics),
            "packet_locked": self._packet_locked,
            "packet_level_dbfs": _round_or_none(
                float(np.median(levels)) if levels else None
            ),
            "packet_quality_db": _round_or_none(
                float(np.median(qualities)) if qualities else None
            ),
            "direction_level_dbfs": _round_or_none(direction_level),
            "direction_quality_db": _round_or_none(
                float(np.median(direction_qualities)) if direction_qualities else None
            ),
        }
        self._xyxy_map_search_samples.append(sample)
        self._write_event(
            "XYXY_MAP_SEARCH_SAMPLE",
            packet_metric={
                "level_dbfs": sample["packet_level_dbfs"],
                "quality_db": sample["packet_quality_db"],
                "direction_level_dbfs": sample["direction_level_dbfs"],
                "direction_quality_db": sample["direction_quality_db"],
            },
        )
        self._publish_diagnostic_event("XYXY_MAP_SEARCH_SAMPLE", sample)
        self.get_logger().info(
            "XYXY map-search sample: "
            f"source={source} packets={len(self._packet_metrics)} "
            f"direction(900/1050)={direction_level} dBFS"
        )
        self._remember_recent_goal(
            actual_waypoint,
            "XYXY_MAP_SEARCH_MEASURED",
            mark_zone=False,
        )
        recovery = self._continuous_uwb_recovery_assessment()
        if bool(recovery["accepted"]):
            self._publish_diagnostic_event("UWB_RECOVERY_ACCEPTED", recovery)
            self._resume_after_fresh_uwb(recovery)
            return
        self._publish_diagnostic_event("UWB_RECOVERY_REJECTED", recovery)
        self.get_logger().info(
            "continuous UWB recovery not yet confirmed; continue XYXY-priority "
            f"map search: reasons={recovery['reasons']} "
            f"fresh={recovery['fresh_samples']} duration={recovery['duration_sec']}s "
            f"sigma={recovery['sigma_m']}m"
        )
        self._current_waypoint = None
        self._xyxy_map_search_index += 1
        self._reprioritize_xyxy_map_search_queue()
        self._transition("NAVIGATE_XYXY_MAP_SEARCH")

    def _reprioritize_xyxy_map_search_queue(self) -> None:
        """Always choose the map-safe direction with the strongest XYXY estimate.

        During UWB HOLD/LOST this is the primary selector.  Candidates already
        measured weaker than the current stop are removed; unmeasured points
        are ordered by inverse-distance interpolation of every full-XYXY
        observation.  Thus no lower measured branch is deliberately chosen,
        while every actual goal remains bounded by the static map, live Nav2
        costmap and planner preflight.
        """

        if not self._xyxy_map_search_samples:
            return
        latest = self._xyxy_map_search_samples[-1]
        current_level_value = latest.get("direction_level_dbfs")
        if current_level_value is None:
            self._publish_diagnostic_event(
                "XYXY_PRIORITY_REQUIRES_VALID_PACKET", latest
            )
            return
        current_level = float(current_level_value)
        self._xyxy_priority_reference_level = current_level
        minimum_gain = float(
            self.get_parameter("xyxy_priority_min_gain_db").value
        )
        observations = self._xyxy_priority_observations()
        completed = self._xyxy_map_search_queue[:self._xyxy_map_search_index]
        remaining = self._xyxy_map_search_queue[self._xyxy_map_search_index:]
        bin_radius = max(
            0.10,
            float(self.get_parameter("xyxy_priority_history_bin_m").value),
        )

        ranked: list[tuple[float, tuple[MapPoint, str, float | None], float | None]] = []
        rejected_measured = 0
        for candidate in remaining:
            point = candidate[0]
            nearby_levels = [
                level
                for _stamp, level, observed in observations
                if self._distance(point, observed) <= bin_radius
            ]
            measured_level = (
                float(np.median(nearby_levels)) if nearby_levels else None
            )
            if (
                measured_level is not None
                and measured_level < current_level + minimum_gain
            ):
                rejected_measured += 1
                continue
            expected = self._expected_xyxy_level(point, observations)
            if expected is None:
                expected = -math.inf
            ranked.append((expected, candidate, measured_level))

        proven_increasing = [
            item for item in ranked
            if item[2] is not None
            and float(item[2]) >= current_level + minimum_gain
        ]
        unmeasured = [item for item in ranked if item[2] is None]
        proven_increasing.sort(key=lambda item: item[0], reverse=True)
        unmeasured.sort(key=lambda item: item[0], reverse=True)
        ordered = proven_increasing + unmeasured

        # If every remaining point was measured weaker, rebuild around the
        # globally strongest historical full-XYXY pose. This keeps the robot
        # from intentionally taking a known decreasing branch while recovery
        # remains mandatory until three seconds of FRESH.
        if not ordered:
            strongest = next(
                (
                    item for item in observations
                    if item[1] >= current_level + minimum_gain
                    and self._topology is not None
                    and self._topology.safe(item[2])
                    and not self._goal_is_recent_tabu(item[2])
                ),
                None,
            )
            if strongest is not None:
                ordered = [
                    (
                        strongest[1],
                        (strongest[2], "XYXY_RETURN_TO_STRONGEST", None),
                        strongest[1],
                    )
                ]

        # With no measured proof, the least-bad safe action is still the
        # maximum expected level. The diagnostic explicitly records that this
        # is a prediction, not a guaranteed physical increase.
        if not ordered and remaining:
            fallback = [
                (
                    self._expected_xyxy_level(item[0], observations)
                    if self._expected_xyxy_level(item[0], observations)
                    is not None else -math.inf,
                    item,
                    None,
                )
                for item in remaining
            ]
            fallback.sort(key=lambda item: item[0], reverse=True)
            ordered = fallback

        self._xyxy_map_search_queue = completed + [item[1] for item in ordered]
        self._publish_diagnostic_event(
            "XYXY_PRIORITY_QUEUE_UPDATED",
            {
                "current_level_dbfs": round(current_level, 4),
                "required_gain_db": minimum_gain,
                "proven_increasing_candidates": len(proven_increasing),
                "unmeasured_ranked_candidates": len(unmeasured),
                "rejected_measured_weaker": rejected_measured,
                "remaining_candidates": len(ordered),
                "next_goal": (
                    self._diagnostic_point(ordered[0][1][0])
                    if ordered else None
                ),
                "next_expected_level_dbfs": (
                    _round_or_none(ordered[0][0]) if ordered else None
                ),
            },
        )
        self.get_logger().info(
            "XYXY-priority recovery queue updated: "
            f"current={current_level:.1f}dBFS, "
            f"proven_stronger={len(proven_increasing)}, "
            f"unmeasured_ranked={len(unmeasured)}, "
            f"known_weaker_skipped={rejected_measured}"
        )

    def _trim_doa_observations(self, now: float) -> None:
        history_sec = float(self.get_parameter("doa_observation_history_sec").value)
        while (
            self._doa_observations
            and now - self._doa_observations[0][0] > history_sec
        ):
            self._doa_observations.popleft()

    def _doa_sensor_pose(self, base_pose: MapPoint) -> MapPoint:
        """Transform the ReSpeaker origin into map coordinates."""

        x = float(self.get_parameter("doa_sensor_x_m").value)
        y = float(self.get_parameter("doa_sensor_y_m").value)
        cos_yaw, sin_yaw = math.cos(base_pose.yaw_rad), math.sin(base_pose.yaw_rad)
        return MapPoint(
            x_m=base_pose.x_m + cos_yaw * x - sin_yaw * y,
            y_m=base_pose.y_m + sin_yaw * x + cos_yaw * y,
            yaw_rad=base_pose.yaw_rad,
        )

    @staticmethod
    def _doa_space_payload(
        assessment: DoaSpaceAssessment | None,
    ) -> dict[str, object] | None:
        if assessment is None:
            return None
        return {
            "allowed": assessment.allowed,
            "mode": assessment.mode,
            "reason": assessment.reason,
            "nearest_wall_m": _round_or_none(assessment.nearest_wall_m),
            "open_sector_count": assessment.open_sector_count,
            "fan_center_rad": _round_or_none(assessment.fan_center_rad),
            "fan_width_deg": _round_or_none(
                math.degrees(assessment.fan_width_rad)
            ),
            "fan_center_depth_m": _round_or_none(
                assessment.fan_center_depth_m
            ),
            "fan_open_ratio": _round_or_none(assessment.fan_open_ratio),
            "confidence_weight": _round_or_none(
                assessment.confidence_weight
            ),
        }

    def _assess_doa_space(self, base_pose: MapPoint) -> DoaSpaceAssessment | None:
        if self._topology is None:
            return None
        return assess_doa_open_space(
            self._topology,
            self._doa_sensor_pose(base_pose),
            ray_step_deg=float(
                self.get_parameter("doa_space_ray_step_deg").value
            ),
            ray_max_range_m=float(
                self.get_parameter("doa_space_ray_max_range_m").value
            ),
            minimum_wall_clearance_m=float(
                self.get_parameter("doa_space_min_wall_clearance_m").value
            ),
            general_wall_clearance_m=float(
                self.get_parameter("doa_space_general_wall_clearance_m").value
            ),
            open_depth_m=float(
                self.get_parameter("doa_space_open_depth_m").value
            ),
            general_min_open_sectors=int(
                self.get_parameter("doa_space_general_min_open_sectors").value
            ),
            fan_width_deg=float(
                self.get_parameter("doa_space_fan_width_deg").value
            ),
            fan_center_depth_m=float(
                self.get_parameter("doa_space_fan_center_depth_m").value
            ),
            fan_min_open_ratio=float(
                self.get_parameter("doa_space_fan_min_open_ratio").value
            ),
            general_confidence_weight=float(
                self.get_parameter("doa_space_general_confidence_weight").value
            ),
            fan_confidence_weight=float(
                self.get_parameter("doa_space_fan_confidence_weight").value
            ),
        )

    def _doa_inputs_ready(self) -> bool:
        if not self._fresh_uwb_is_stable():
            return False
        range_m = self._fresh_uwb_median_m()
        return bool(
            range_m is not None
            and range_m <= float(self.get_parameter("doa_probe_max_range_m").value)
            and self._packet_locked
            and self._recent_xyxy_available(
                float(self.get_parameter("yolo_xyxy_metric_max_age_sec").value)
            )
        )

    def _doa_stationary_duration(self) -> float:
        """Track actual TF motion instead of assuming Nav2 success means stopped."""

        now = time.monotonic()
        pose = self._robot_pose()
        if pose is None:
            self._doa_stationary_reference = None
            self._doa_stationary_since = None
            return 0.0
        reference = self._doa_stationary_reference
        if reference is None or self._doa_stationary_since is None:
            self._doa_stationary_reference = pose
            self._doa_stationary_since = now
            return 0.0
        moved = self._distance(pose, reference) > float(
            self.get_parameter("doa_stationary_position_tolerance_m").value
        )
        rotated = abs(_wrap_angle_rad(pose.yaw_rad - reference.yaw_rad)) > float(
            self.get_parameter("doa_stationary_yaw_tolerance_rad").value
        )
        if moved or rotated:
            self._doa_stationary_reference = pose
            self._doa_stationary_since = now
            if self._doa_enabled:
                self._publish_doa_enabled(False)
            return 0.0
        return now - self._doa_stationary_since

    def _stable_doa_probe_cluster(self) -> tuple[float, list[float]] | None:
        """Require at least two agreeing bearings among the latest three."""

        required = int(self.get_parameter("doa_probe_min_stable_samples").value)
        maximum_difference = float(
            self.get_parameter("doa_probe_max_spread_rad").value
        )
        recent = self._doa_probe_angles[-3:]
        best: list[float] = []
        for seed in recent:
            cluster = [
                angle
                for angle in recent
                if abs(_wrap_angle_rad(angle - seed)) <= maximum_difference
            ]
            if len(cluster) > len(best):
                best = cluster
        if len(best) < required:
            return None
        center = math.atan2(
            float(np.mean(np.sin(best))),
            float(np.mean(np.cos(best))),
        )
        if any(
            abs(_wrap_angle_rad(angle - center)) > maximum_difference
            for angle in best
        ):
            return None
        return center, best

    def _doa_probe_gate_ready(self, snapshot) -> bool:
        """Check the strict preconditions before enabling ReSpeaker DOA."""

        if not bool(self.get_parameter("doa_fusion_enabled").value):
            return False
        if self._topology is None or not snapshot.sections:
            return False
        now = time.monotonic()
        self._trim_doa_observations(now)
        required_observations = int(
            self.get_parameter("doa_probe_required_observations").value
        )
        if len(self._doa_observations) >= required_observations:
            return False
        if not self._doa_inputs_ready():
            return False

        max_attempts = int(self.get_parameter("doa_probe_max_attempts").value)
        if self._doa_probe_attempts >= max_attempts:
            # Do not loop at the same weak evidence set.  After another full
            # evidence interval the map probability may genuinely have
            # changed, so allow a new two-pose DOA attempt.
            interval = int(self.get_parameter("evaluation_interval_waypoints").value)
            if len(self._evidence) < self._doa_probe_last_evidence_count + interval:
                return False
            self._doa_probe_attempts = 0
            self._doa_probe_targets = []
            self._doa_probe_last_evidence_count = len(self._evidence)
        return True

    def _next_doa_probe_target(self, snapshot) -> MapPoint | None:
        """Choose a map-safe, spatially independent probe in top sections."""

        if self._topology is None:
            return None
        robot = self._robot_pose()
        if robot is None:
            return None
        top_sections = list(snapshot.sections[:2])
        candidates: list[MapPoint] = []
        # Use the current safe pose as the first probe when possible. It
        # avoids a needless Nav2 motion before the first measurement; the
        # second target is still forced to be spatially independent.
        if not self._doa_probe_targets and self._topology.safe(robot):
            candidates.append(robot)
        for item in top_sections:
            candidates.append(item.zone.interior)
            candidates.extend(
                zone_local_waypoints(
                    self._topology,
                    item.zone.zone_id,
                    item.zone.interior,
                    radius_m=1.8,
                    max_waypoints=8,
                )
            )

        separation_m = float(
            self.get_parameter("doa_probe_min_separation_m").value
        )
        prior_sensor_poses = [
            observation.microphone_pose
            for _stamp, observation in self._doa_observations
        ]
        prior_sensor_poses.extend(
            self._doa_sensor_pose(target) for target in self._doa_probe_targets
        )
        unique: list[MapPoint] = []
        for point in candidates:
            if not self._topology.safe(point):
                continue
            space = self._assess_doa_space(point)
            if space is None or not space.allowed:
                # A collision-safe corridor is not necessarily acoustically
                # open. Long two-sided corridors fail the six-sector/fan
                # geometry even when Nav2 can traverse them.
                continue
            expected_sensor = self._doa_sensor_pose(point)
            if any(
                self._distance(expected_sensor, prior) < separation_m
                for prior in prior_sensor_poses
            ):
                continue
            if any(self._distance(point, selected) < 0.25 for selected in unique):
                continue
            unique.append(point)
        if not unique:
            return None
        return min(unique, key=lambda point: self._distance(point, robot))

    def _start_doa_probe_if_ready(self, snapshot) -> bool:
        """Enter the two-pose DOA branch before a final room-entry goal."""

        if not self._doa_probe_gate_ready(snapshot):
            return False
        target = self._next_doa_probe_target(snapshot)
        if target is None:
            self.get_logger().warning(
                "DOA fusion eligible but no second map-safe, separated probe pose exists"
            )
            self._doa_probe_attempts += 1
            self._doa_probe_last_evidence_count = len(self._evidence)
            return False
        self._doa_probe_target = target
        self._current_waypoint = target
        self._publish_diagnostic_event(
            "DOA_PROBE_PLANNED",
            {
                "target": self._diagnostic_point(target),
                "existing_observations": len(self._doa_observations),
                "top_section": snapshot.sections[0].zone.zone_id,
                "top_probability": _round_or_none(snapshot.sections[0].probability),
            },
        )
        self._transition("NAVIGATE_DOA_PROBE")
        return True

    def _begin_doa_probe_collection(self) -> None:
        self._doa_probe_angles = []
        self._doa_probe_metrics = []
        self._doa_probe_space = None
        self._doa_stationary_reference = self._robot_pose()
        self._doa_stationary_since = time.monotonic()
        self._collect_started = time.monotonic()
        # DOA stays disabled through the mechanical settle interval. It is
        # enabled only after TF stationarity and the live map-space gate pass.
        self._publish_doa_enabled(False)
        self._transition("COLLECT_DOA_PROBE")
        self.get_logger().info(
            "DOA probe: verify 1.5 s stationarity and map openness at "
            f"({self._doa_probe_target.x_m:.2f}, {self._doa_probe_target.y_m:.2f})"
            if self._doa_probe_target is not None
            else "DOA probe: collect packet-locked stable ReSpeaker bearings"
        )

    def _advance_doa_probe_navigation(self) -> None:
        target = self._doa_probe_target
        if target is None:
            self._transition("EVALUATE")
            return
        if self._nav_result == "FAILED":
            self.get_logger().warning("DOA probe goal failed; retain base UWB/XYXY ranking")
            self._nav_result = None
            self._doa_probe_attempts += 1
            self._doa_probe_last_evidence_count = len(self._evidence)
            self._doa_probe_targets.append(target)
            self._doa_probe_target = None
            self._transition("EVALUATE")
            return
        if self._nav_result == "SUCCEEDED":
            self._nav_result = None
            self._begin_doa_probe_collection()
            return
        robot = self._robot_pose()
        if (
            robot is not None
            and self._distance(robot, target)
            <= float(self.get_parameter("uwb_fresh_map_search_min_goal_distance_m").value)
        ):
            self._begin_doa_probe_collection()
            return
        if self._nav_goal_handle is None and not self._nav_goal_pending:
            self._send_goal(target, "DOA_PROBE")

    def _finish_doa_probe_collection(self) -> None:
        """Turn repeated stable samples at one pose into one bearing record."""

        elapsed = time.monotonic() - self._collect_started
        settle = float(self.get_parameter("waypoint_settle_sec").value)
        timeout = float(self.get_parameter("doa_probe_timeout_sec").value)
        required_samples = int(
            self.get_parameter("doa_probe_min_stable_samples").value
        )
        if elapsed < settle:
            return

        stationary_sec = self._doa_stationary_duration()
        if stationary_sec < float(self.get_parameter("doa_stationary_sec").value):
            self._publish_doa_enabled(False)
            if elapsed < timeout:
                return

        live_inputs_ready = self._doa_inputs_ready()
        if not live_inputs_ready:
            self._publish_doa_enabled(False)
            if elapsed < timeout:
                return

        base = self._robot_pose() or self._doa_probe_target
        if (
            self._doa_probe_space is None
            and base is not None
            and stationary_sec >= float(self.get_parameter("doa_stationary_sec").value)
        ):
            self._doa_probe_space = self._assess_doa_space(base)
            self._publish_diagnostic_event(
                "DOA_SPACE_ASSESSED",
                {
                    "base_pose": self._diagnostic_point(base),
                    "assessment": self._doa_space_payload(self._doa_probe_space),
                },
            )

        space_allowed = bool(
            self._doa_probe_space is not None
            and self._doa_probe_space.allowed
        )
        if live_inputs_ready and space_allowed and stationary_sec >= float(
            self.get_parameter("doa_stationary_sec").value
        ):
            self._publish_doa_enabled(True)
        else:
            self._publish_doa_enabled(False)

        cluster = self._stable_doa_probe_cluster()
        terminal_failure = bool(
            elapsed >= timeout
            or (
                self._doa_probe_space is not None
                and not self._doa_probe_space.allowed
            )
        )
        if cluster is None and not terminal_failure:
            return

        target = self._doa_probe_target
        self._publish_doa_enabled(False)
        self._doa_probe_attempts += 1
        self._doa_probe_last_evidence_count = len(self._evidence)
        if target is not None:
            self._doa_probe_targets.append(target)

        accepted = False
        if cluster is not None and base is not None and self._doa_probe_space is not None:
            relative, clustered_angles = cluster
            residuals = np.abs(np.asarray([
                _wrap_angle_rad(float(angle) - relative)
                for angle in clustered_angles
            ]))
            microphone = self._doa_sensor_pose(base)
            bearing = _wrap_angle_rad(
                base.yaw_rad
                + float(self.get_parameter("doa_sensor_yaw_rad").value)
                + relative
            )
            bearing_open = doa_bearing_is_open(
                self._doa_probe_space,
                bearing,
                minimum_depth_m=float(
                    self.get_parameter("doa_space_open_depth_m").value
                ),
            )
            if live_inputs_ready and self._doa_probe_space.allowed and bearing_open:
                observation = DoaObservation(
                    microphone,
                    bearing,
                    self._doa_probe_space.confidence_weight,
                )
                self._doa_observations.append((time.monotonic(), observation))
                self._trim_doa_observations(time.monotonic())
                accepted = True
                self._publish_diagnostic_event(
                    "DOA_PROBE_ACCEPTED",
                    {
                        "raw_samples": len(self._doa_probe_angles),
                        "clustered_samples": len(clustered_angles),
                        "relative_angle_rad": _round_or_none(relative),
                        "world_bearing_rad": _round_or_none(bearing),
                        "max_cluster_spread_rad": _round_or_none(
                            float(np.max(residuals)) if len(residuals) else 0.0
                        ),
                        "microphone_pose": self._diagnostic_point(microphone),
                        "space": self._doa_space_payload(self._doa_probe_space),
                        "bearing_open": bearing_open,
                        "metrics": self._doa_probe_metrics[-3:],
                        "observation_count": len(self._doa_observations),
                    },
                )
                self.get_logger().info(
                    "DOA probe accepted as map likelihood: "
                    f"mode={self._doa_probe_space.mode}, "
                    f"effective weight="
                    f"{self._config.doa_weight * self._doa_probe_space.confidence_weight:.3f}, "
                    f"{len(self._doa_observations)}/"
                    f"{int(self.get_parameter('doa_probe_required_observations').value)} "
                    "independent observations"
                )
            elif not bearing_open:
                self.get_logger().warning(
                    "DOA probe rejected: final bearing points outside the mapped open fan"
                )
        if not accepted:
            self._publish_diagnostic_event(
                "DOA_PROBE_REJECTED",
                {
                    "raw_samples": len(self._doa_probe_angles),
                    "required_samples": required_samples,
                    "elapsed_sec": _round_or_none(elapsed),
                    "stationary_sec": _round_or_none(stationary_sec),
                    "live_inputs_ready": live_inputs_ready,
                    "space": self._doa_space_payload(self._doa_probe_space),
                    "target": self._diagnostic_point(target),
                },
            )
            self.get_logger().warning(
                "DOA probe yielded no reliable bearing; keep UWB/XYXY-only ranking"
            )
        self._doa_probe_target = None
        self._doa_probe_angles = []
        self._doa_probe_metrics = []
        self._doa_probe_space = None
        self._doa_stationary_reference = None
        self._doa_stationary_since = None
        self._transition("EVALUATE")

    def _maintain_doa_active_gate(self) -> None:
        """Keep post-entry DOA off unless the same live/map gates still pass."""

        now = time.monotonic()
        stationary_sec = self._doa_stationary_duration()
        base = self._robot_pose()
        if base is not None and (
            self._doa_active_space is None
            or now - self._doa_active_space_at >= 1.0
        ):
            self._doa_active_space = self._assess_doa_space(base)
            self._doa_active_space_at = now
        enabled = bool(
            self._doa_inputs_ready()
            and stationary_sec
            >= float(self.get_parameter("doa_stationary_sec").value)
            and self._doa_active_space is not None
            and self._doa_active_space.allowed
        )
        if enabled != self._doa_enabled:
            self._publish_doa_enabled(enabled)
            self._publish_diagnostic_event(
                "DOA_ACTIVE_GATE_CHANGED",
                {
                    "enabled": enabled,
                    "stationary_sec": _round_or_none(stationary_sec),
                    "inputs_ready": self._doa_inputs_ready(),
                    "space": self._doa_space_payload(self._doa_active_space),
                },
            )

    def _tick(self) -> None:
        if not bool(self.get_parameter("search_enabled").value):
            if self._state != "DISABLED":
                self._cancel_navigation()
                self._publish_doa_enabled(False)
                self._publish_yolo_ready(False)
                self._transition("DISABLED")
            return
        if self._state in {
            "PLAN_UWB_FRESH_MAP_SEARCH",
            "UWB_ZONE_COOLDOWN_WAIT",
            "NAVIGATE_XYXY_MAP_SEARCH",
            "COLLECT_XYXY_MAP_SEARCH",
            "UWB_LOST_HOLD",
        }:
            recovery = self._continuous_uwb_recovery_assessment()
            if bool(recovery["accepted"]):
                self._publish_diagnostic_event(
                    "UWB_CONTINUOUS_FRESH_RECOVERY_ACCEPTED", recovery
                )
                self._resume_after_fresh_uwb(recovery)
                return
        if self._state == "UWB_LOST_HOLD":
            # Compatibility for a state published by an older running node.
            # New UWB loss handling never waits here.
            self._transition("PLAN_UWB_FRESH_MAP_SEARCH")
            return
        if self._state == "PLAN_UWB_FRESH_MAP_SEARCH":
            self._plan_uwb_fresh_map_search()
            return
        if self._state == "UWB_ZONE_COOLDOWN_WAIT":
            self._advance_uwb_zone_cooldown_wait()
            return
        if self._state == "NAVIGATE_XYXY_MAP_SEARCH":
            if self._xyxy_navigation_guard_expired():
                self._reject_decreasing_xyxy_map_search_route()
                return
            self._advance_xyxy_map_search_navigation()
            return
        if self._state == "COLLECT_XYXY_MAP_SEARCH":
            self._finish_xyxy_map_search_collection()
            return
        if self._state == "XYXY_DEGRADING_HOLD":
            self._advance_xyxy_degrading_hold()
            return
        if self._uwb_navigation_guard_expired():
            self._hold_for_uwb_loss()
            return
        if self._xyxy_navigation_guard_expired():
            self._hold_for_xyxy_degradation()
            return
        if self._yolo_verify_gate_ready():
            self._enter_yolo_verify()
            return
        if (
            bool(self.get_parameter("initial_map_survey_enabled").value)
            and self._state in {"NAVIGATE_COVERAGE", "ANCHOR_FOCUS"}
        ):
            gate_reason = self._survey_decision_gate_reason(
                coverage_complete=False
            )
            if gate_reason is not None:
                self._cancel_navigation()
                self._survey_selection_ready = True
                self._survey_selection_reason = gate_reason
                self._publish_diagnostic_event(
                    "SURVEY_SELECTION_GATE_OPEN_WHILE_MOVING",
                    {
                        "reason": gate_reason,
                        "joint_waypoints": self._survey_joint_evidence_count(),
                        "observed_zones": sorted(
                            self._survey_observed_zone_ids()
                        ),
                    },
                )
                self._transition("EVALUATE")
                return
        if self._state in {"DISABLED", "WAIT_MAP"}:
            if self._map is None:
                return
            self._transition("PLAN_COVERAGE")
        if self._state == "PLAN_COVERAGE":
            self._plan_coverage()
        elif self._state == "NAVIGATE_COVERAGE":
            self._advance_coverage_navigation()
        elif self._state == "ANCHOR_FOCUS":
            self._advance_coverage_navigation()
        elif self._state == "COLLECT":
            self._finish_collection_when_ready()
        elif self._state == "EVALUATE":
            self._evaluate_evidence()
        elif self._state == "NAVIGATE_DOA_PROBE":
            self._advance_doa_probe_navigation()
        elif self._state == "COLLECT_DOA_PROBE":
            self._finish_doa_probe_collection()
        elif self._state == "NAVIGATE_ENTRY":
            self._advance_entry_navigation()
        elif self._state == "NAVIGATE_INTERIOR":
            self._advance_interior_navigation()
        elif self._state == "YOLO_VERIFY":
            self._advance_yolo_verify()
        elif self._state == "DOA_ACTIVE":
            self._maintain_doa_active_gate()
            self._start_person_approach_if_ready()
        elif self._state == "NAVIGATE_VICTIM":
            self._advance_person_navigation()

    def _plan_coverage(self) -> None:
        if self._map is None:
            self._transition("WAIT_MAP")
            return
        pose = self._robot_pose()
        if pose is None:
            return
        try:
            grid = self._map
            origin = grid.info.origin
            yaw = math.atan2(2.0 * (origin.orientation.w * origin.orientation.z + origin.orientation.x * origin.orientation.y), 1.0 - 2.0 * (origin.orientation.y**2 + origin.orientation.z**2))
            self._topology = build_topology(
                grid.data,
                width=int(grid.info.width), height=int(grid.info.height), resolution_m=float(grid.info.resolution),
                origin_x_m=float(origin.position.x), origin_y_m=float(origin.position.y), origin_yaw_rad=yaw,
                occupancy_threshold=int(self.get_parameter("map_occupancy_threshold").value),
                unknown_is_blocked=bool(self.get_parameter("map_unknown_is_blocked").value), config=self._config,
            )
        except ValueError as exc:
            self.get_logger().error(f"map topology build failed: {exc}")
            self._transition("WAIT_MAP")
            return
        self._coverage = (
            closed_space_first_coverage_waypoints(self._topology, pose)
            if bool(
                self.get_parameter("coverage_closed_space_first").value
            )
            else coverage_waypoints(self._topology, pose)
        )
        self._coverage_index = 0
        self._coverage_cycle += 1
        if not self._coverage:
            self.get_logger().error("map has no safe coverage waypoint")
            self._transition("WAIT_MAP")
            return
        self.get_logger().info(
            f"coverage cycle {self._coverage_cycle}: {len(self._coverage)} safe points, "
            f"zones={len(self._topology.zones)} portals={len(self._topology.portals)} "
            f"closed_space_first={bool(self.get_parameter('coverage_closed_space_first').value)}"
        )
        self._transition("NAVIGATE_COVERAGE")

    def _advance_coverage_navigation(self) -> None:
        while self._coverage_index < len(self._coverage):
            skipped = self._coverage[self._coverage_index]
            point_tabu = self._goal_is_recent_tabu(skipped)
            zone_tabu = self._goal_zone_on_cooldown(skipped)
            if not point_tabu and not zone_tabu:
                break
            if zone_tabu and not point_tabu:
                # Prefer another map section. If no alternative remains,
                # pause at this index so it becomes eligible after the TTL
                # instead of rebuilding coverage in a tight loop.
                alternative_exists = any(
                    not self._goal_is_recent_tabu(candidate)
                    and not self._goal_zone_on_cooldown(candidate)
                    for candidate in self._coverage[self._coverage_index + 1:]
                )
                if not alternative_exists:
                    now = time.monotonic()
                    if now - self._last_zone_cooldown_wait_log >= 5.0:
                        self._last_zone_cooldown_wait_log = now
                        self.get_logger().info(
                            "remaining coverage is in a recently visited map zone; "
                            "wait for the 60-second cooldown"
                        )
                    return
            self._publish_diagnostic_event(
                "COVERAGE_RECENT_ZONE_SKIPPED"
                if zone_tabu else "COVERAGE_RECENT_GOAL_SKIPPED",
                {
                    "coverage_index": self._coverage_index,
                    "goal": self._diagnostic_point(skipped),
                    "point_tabu": point_tabu,
                    "zone_cooldown": zone_tabu,
                },
            )
            self.get_logger().info(
                "skip recently cancelled/visited coverage point or zone "
                f"({skipped.x_m:.2f}, {skipped.y_m:.2f})"
            )
            self._coverage_index += 1
        if self._coverage_index >= len(self._coverage):
            self._transition("EVALUATE")
            return
        if self._nav_result == "FAILED":
            self.get_logger().warning("coverage goal failed; skip waypoint")
            self._nav_result = None
            self._coverage_index += 1
            return
        if self._nav_result == "SUCCEEDED":
            self._nav_result = None
            self._begin_collection()
            return
        if (
            self._nav_goal_handle is None
            and not self._nav_goal_pending
            and not self._path_preflight_pending
        ):
            self._current_waypoint = self._coverage[self._coverage_index]
            self._current_waypoint_id += 1
            self._send_goal(self._current_waypoint, "COVERAGE")

    def _begin_collection(self) -> None:
        self._range_values = []
        self._packet_metrics = []
        self._collect_started = time.monotonic()
        self._transition("COLLECT")
        self.get_logger().info(
            f"waypoint {self._current_waypoint_id}: settle then collect UWB/XYXY for "
            f"{float(self.get_parameter('evidence_window_sec').value):.1f}s"
        )

    def _finish_collection_when_ready(self) -> None:
        elapsed = time.monotonic() - self._collect_started
        required = float(self.get_parameter("waypoint_settle_sec").value) + float(self.get_parameter("evidence_window_sec").value)
        if elapsed < required:
            return
        waypoint = self._current_waypoint
        if waypoint is None:
            self.get_logger().warning("waypoint pose missing; continuing coverage")
            self._coverage_index += 1
            self._transition(
                "ANCHOR_FOCUS" if self._close_anchor is not None else "NAVIGATE_COVERAGE"
            )
            return

        observed_wall_time_sec = self.get_clock().now().nanoseconds / 1e9
        observed_monotonic_sec = time.monotonic()
        levels = [float(metric["level_dbfs"]) for metric in self._packet_metrics]
        qualities = [float(metric["quality_db"]) for metric in self._packet_metrics]
        direction_levels = [
            float(metric["direction_level_dbfs"])
            for metric in self._packet_metrics
            if "direction_level_dbfs" in metric
        ]
        direction_qualities = [
            float(metric["direction_quality_db"])
            for metric in self._packet_metrics
            if "direction_quality_db" in metric
        ]
        base_pose = self._robot_pose() or waypoint
        sipeed_pose = self._sipeed_pose(base_pose)
        minimum_fresh_samples = (
            int(
                self.get_parameter(
                    "survey_min_fresh_samples_per_waypoint"
                ).value
            )
            if bool(self.get_parameter("initial_map_survey_enabled").value)
            else 1
        )
        aggregate = (
            aggregate_ranges(self._range_values)
            if len(self._range_values) >= minimum_fresh_samples
            else None
        )
        if self._range_values and aggregate is None:
            self.get_logger().warning(
                "discarding sparse UWB waypoint evidence: "
                f"FRESH samples={len(self._range_values)}/{minimum_fresh_samples}"
            )

        packet_level = float(np.median(levels)) if levels else None
        packet_quality = float(np.median(qualities)) if qualities else None
        direction_level = (
            float(np.median(direction_levels)) if direction_levels else None
        )
        direction_quality = (
            float(np.median(direction_qualities)) if direction_qualities else None
        )
        record: EvidenceWaypoint | None = None
        if aggregate is not None:
            record = EvidenceWaypoint(
                waypoint_id=self._current_waypoint_id,
                pose=base_pose,
                range_m=aggregate.range_m,
                range_sigma_m=aggregate.sigma_m,
                fresh_samples=aggregate.used_count,
                packet_valid_packets=len(self._packet_metrics),
                packet_locked=self._packet_locked,
                packet_level_dbfs=packet_level,
                packet_quality_db=packet_quality,
                packet_direction_level_dbfs=direction_level,
                packet_direction_quality_db=direction_quality,
                packet_pose=sipeed_pose,
                observed_wall_time_sec=observed_wall_time_sec,
                observed_monotonic_sec=observed_monotonic_sec,
            )
            self._evidence.append(record)

        if (
            len(self._packet_metrics)
            >= int(self.get_parameter("uwb_guard_min_valid_packets").value)
            and direction_level is not None
        ):
            self._last_stationary_xyxy_level = direction_level
        zone_id = self._topology.zone_for_point(base_pose) if self._topology else None
        record_kind = (
            "JOINT"
            if aggregate is not None and direction_level is not None
            else "UWB_ONLY"
            if aggregate is not None
            else "XYXY_ONLY"
            if direction_level is not None
            else "EMPTY"
        )
        self._summary_writer.writerow({
            "wall_time_sec": f"{observed_wall_time_sec:.6f}",
            "monotonic_sec": f"{observed_monotonic_sec:.6f}",
            "record_kind": record_kind,
            "waypoint_id": self._current_waypoint_id, "cycle": self._coverage_cycle,
            "map_x_m": f"{base_pose.x_m:.3f}", "map_y_m": f"{base_pose.y_m:.3f}", "map_yaw_rad": f"{base_pose.yaw_rad:.3f}",
            "sipeed_x_m": f"{sipeed_pose.x_m:.3f}", "sipeed_y_m": f"{sipeed_pose.y_m:.3f}",
            "zone_id": zone_id or "", "uwb_status": self._uwb_status,
            "uwb_median_m": "" if aggregate is None else f"{aggregate.range_m:.3f}",
            "uwb_sigma_m": "" if aggregate is None else f"{aggregate.sigma_m:.3f}",
            "uwb_samples": len(self._range_values),
            "packet_valid_packets": len(self._packet_metrics),
            "packet_locked": int(self._packet_locked),
            "packet_level_dbfs": "" if packet_level is None else f"{packet_level:.3f}",
            "packet_quality_db": "" if packet_quality is None else f"{packet_quality:.3f}",
            "packet_direction_level_dbfs": "" if direction_level is None else f"{direction_level:.3f}",
            "packet_direction_quality_db": "" if direction_quality is None else f"{direction_quality:.3f}",
        })
        self._streams[1].flush()
        self._survey_waypoint_records += 1
        self.get_logger().info(
            f"survey wp={self._current_waypoint_id} zone={zone_id} kind={record_kind} "
            f"UWB={'--' if aggregate is None else f'{aggregate.range_m:.2f}±{aggregate.sigma_m:.2f}m'} "
            f"FRESH={len(self._range_values)} XYXY packets={len(self._packet_metrics)} "
            f"full={packet_level} direction(900/1050)={direction_level} "
            f"SIPEED=({sipeed_pose.x_m:.2f},{sipeed_pose.y_m:.2f})"
        )
        self._publish_diagnostic_event(
            "SURVEY_WAYPOINT_RECORDED",
            {
                "record_kind": record_kind,
                "wall_time_sec": round(observed_wall_time_sec, 6),
                "monotonic_sec": round(observed_monotonic_sec, 6),
                "zone_id": zone_id,
                "base_pose": self._diagnostic_point(base_pose),
                "sipeed_pose": self._diagnostic_point(sipeed_pose),
                "uwb_status": self._uwb_status,
                "uwb_range_m": None if aggregate is None else round(aggregate.range_m, 4),
                "uwb_sigma_m": None if aggregate is None else round(aggregate.sigma_m, 4),
                "fresh_samples": len(self._range_values),
                "xyxy_packets": len(self._packet_metrics),
                "xyxy_direction_level_dbfs": _round_or_none(direction_level),
            },
        )
        if (
            not self._beacon_evidence_seen
            and record is not None
            and record.packet_valid_packets
            >= int(self.get_parameter("uwb_guard_min_valid_packets").value)
            and record.packet_direction_level_dbfs is not None
        ):
            # packet_metric is emitted only after a complete XYXY packet, and
            # range aggregation now contains synchronized FRESH samples only.
            self._beacon_evidence_seen = True
            self.get_logger().info(
                "validated XYXY + FRESH UWB evidence acquired; keep map survey active"
            )
        self._coverage_index += 1
        if bool(self.get_parameter("initial_map_survey_enabled").value):
            coverage_complete = self._coverage_index >= len(self._coverage)
            gate_reason = self._survey_decision_gate_reason(
                coverage_complete=coverage_complete
            )
            if gate_reason is not None:
                self._survey_selection_ready = True
                self._survey_selection_reason = gate_reason
                self._publish_diagnostic_event(
                    "SURVEY_SELECTION_GATE_OPEN",
                    {
                        "reason": gate_reason,
                        "joint_waypoints": self._survey_joint_evidence_count(),
                        "observed_zones": sorted(self._survey_observed_zone_ids()),
                    },
                )
                self._transition("EVALUATE")
            else:
                self._transition(
                    "EVALUATE" if coverage_complete else "NAVIGATE_COVERAGE"
                )
            return

        anchor_was_created = record is not None and self._accept_close_anchor(record)
        if (
            self._close_anchor is not None
            and not anchor_was_created
            and record is not None
            and self._topology is not None
            and self._topology.zone_for_point(record.pose) == self._close_anchor_zone_id
        ):
            self._close_anchor_local_samples += 1

        interval = int(self.get_parameter("evaluation_interval_waypoints").value)
        if (
            len(self._evidence) >= self._config.min_evidence_waypoints
            and (
                self._close_anchor is not None
                or len(self._evidence) % interval == 0
            )
        ):
            self._transition("EVALUATE")
        else:
            self._transition(
                "ANCHOR_FOCUS" if self._close_anchor is not None else "NAVIGATE_COVERAGE"
            )

    def _evaluate_evidence(self) -> None:
        if self._topology is None:
            self._transition("PLAN_COVERAGE")
            return
        survey_enabled = bool(
            self.get_parameter("initial_map_survey_enabled").value
        )
        coverage_complete = self._coverage_index >= len(self._coverage)
        if survey_enabled:
            gate_reason = self._survey_decision_gate_reason(
                coverage_complete=coverage_complete
            )
            if gate_reason is None:
                self.get_logger().info(
                    "map survey continues: collect timestamped FRESH UWB and "
                    "validated XYXY evidence before choosing a room"
                )
                self._transition(
                    "PLAN_COVERAGE" if coverage_complete else "NAVIGATE_COVERAGE"
                )
                return
            self._survey_selection_ready = True
            self._survey_selection_reason = gate_reason
        self._trim_doa_observations(time.monotonic())
        # First obtain a UWB+XYXY-only ranking. It is the admission gate for
        # the DOA probe; DOA is never allowed to manufacture a candidate room
        # when range and a validated packet do not already support one.
        baseline = estimate_zone_probabilities(
            self._topology,
            self._evidence,
            temperature=float(
                self.get_parameter("room_probability_temperature").value
            ),
        )
        if not survey_enabled and self._start_doa_probe_if_ready(baseline):
            return
        doa_observations = (
            ()
            if survey_enabled
            else tuple(
                observation for _stamp, observation in self._doa_observations
            )
        )
        decision = decide_room_entry(
            self._topology,
            self._evidence,
            doa_observations=doa_observations,
        )
        if (
            decision is not None
            and survey_enabled
            and self._goal_zone_on_cooldown(decision.interior_pose)
        ):
            self.get_logger().warning(
                "survey room rejected temporarily: its previous fixed route "
                "failed or lost validated XYXY strength"
            )
            self._publish_diagnostic_event(
                "SURVEY_ROOM_ZONE_COOLDOWN_REJECTED",
                {
                    "zone_id": decision.zone.zone_id,
                    "interior": self._diagnostic_point(decision.interior_pose),
                },
            )
            decision = None
        if decision is not None and survey_enabled:
            extrema_supported, extrema = self._survey_extrema_supports_decision(
                decision
            )
            self._publish_diagnostic_event(
                "SURVEY_ROOM_EXTREMA_CHECK",
                extrema,
            )
            if not extrema_supported:
                self.get_logger().warning(
                    "survey room rejected: shortest FRESH UWB and loudest "
                    "validated XYXY areas do not support one adjacent map room"
                )
                decision = None
        if decision is None:
            self.get_logger().info("evidence not yet decisive: continue generic map coverage")
            if survey_enabled:
                # Keep every timestamped record. A failed decision only means
                # more map waypoints are needed; it does not start the old
                # UWB-loss recovery route or discard the completed route.
                self._survey_selection_ready = False
                self._survey_selection_reason = None
                self._transition(
                    "PLAN_COVERAGE" if coverage_complete else "NAVIGATE_COVERAGE"
                )
                return
            if (
                self._close_anchor is not None
                and self._close_anchor_local_samples
                >= int(self.get_parameter("close_anchor_max_local_waypoints").value)
            ):
                self.get_logger().warning(
                    "close-anchor local evidence exhausted without a room decision; "
                    "resume the remaining map coverage"
                )
                self._close_anchor = None
                self._close_anchor_zone_id = None
                self._close_anchor_local_samples = 0
            if self._coverage_index < len(self._coverage):
                self._transition(
                    "ANCHOR_FOCUS" if self._close_anchor is not None else "NAVIGATE_COVERAGE"
                )
                return
            limit = int(self.get_parameter("max_coverage_cycles").value)
            if limit > 0 and self._coverage_cycle >= limit:
                self.get_logger().warning("coverage cycle limit reached without a room decision")
                self._transition("WAIT_MAP")
                return
            self._transition("PLAN_COVERAGE")
            return
        self._last_decision = decision
        self._survey_room_approach_active = survey_enabled
        self._selected_room_pub.publish(self._pose_stamped(decision.interior_pose))
        if decision.staging_pose is not None:
            self._entry_pub.publish(self._pose_stamped(decision.staging_pose))
        self.get_logger().info(
            f"selected map zone={decision.zone.zone_id} score={decision.total_score:.2f} "
            f"survey_reason={self._survey_selection_reason}: {decision.reason}"
        )
        self._transition("NAVIGATE_ENTRY" if decision.staging_pose is not None else "NAVIGATE_INTERIOR")

    def _accept_close_anchor(self, record: EvidenceWaypoint) -> bool:
        """Freeze a strong UWB+XYXY observation and inspect nearby map space."""

        qualifies = (
            record.range_m <= float(self.get_parameter("close_anchor_range_m").value)
            and record.range_sigma_m <= float(self.get_parameter("close_anchor_max_sigma_m").value)
            and record.packet_locked
            and record.packet_valid_packets
            >= int(self.get_parameter("close_anchor_min_packets").value)
        )
        if not qualifies:
            return False
        if self._close_anchor is not None and record.range_m >= self._close_anchor.range_m:
            return False

        self._close_anchor = record
        self._close_anchor_zone_id = (
            self._topology.zone_for_point(record.pose)
            if self._topology is not None
            else None
        )
        self._close_anchor_local_samples = 0
        self._prioritize_anchor_neighborhood(
            record.pose,
            self._close_anchor_zone_id,
        )
        self.get_logger().warning(
            "close UWB anchor accepted: "
            f"range={record.range_m:.2f}±{record.range_sigma_m:.2f}m "
            f"packets={record.packet_valid_packets} zone={self._close_anchor_zone_id}; "
            "prioritize same-section map evidence"
        )
        return True

    def _prioritize_anchor_neighborhood(
        self,
        anchor: MapPoint,
        anchor_zone_id: int | None,
    ) -> None:
        """Verify an anchor in its section before crossing into another one."""

        if self._topology is None or anchor_zone_id is None:
            return
        radius_m = float(self.get_parameter("close_anchor_radius_m").value)
        local_limit = int(self.get_parameter("close_anchor_max_local_waypoints").value)
        completed = self._coverage[:self._coverage_index]
        remaining = self._coverage[self._coverage_index:]
        same_zone = sorted(
            (
                point
                for point in remaining
                if self._topology.zone_for_point(point) == anchor_zone_id
            ),
            key=lambda point: self._distance(point, anchor),
        )
        same_zone_near = [
            point for point in same_zone if self._distance(point, anchor) <= radius_m
        ]
        same_zone_far = [
            point for point in same_zone if self._distance(point, anchor) > radius_m
        ]
        supplemental = zone_local_waypoints(
            self._topology,
            anchor_zone_id,
            anchor,
            radius_m=radius_m,
            max_waypoints=local_limit,
        )
        local: list[MapPoint] = []
        for point in same_zone_near + supplemental + same_zone_far:
            if (
                all(self._distance(point, existing) >= 0.35 for existing in local)
                and all(self._distance(point, item.pose) >= 0.35 for item in self._evidence)
            ):
                local.append(point)
            if len(local) >= local_limit:
                break
        selected = local[:local_limit]
        unchanged = [
            point
            for point in remaining
            if all(self._distance(point, selected_point) >= 0.10 for selected_point in selected)
        ]
        self._coverage = completed + selected + unchanged

    @staticmethod
    def _distance(first: MapPoint, second: MapPoint) -> float:
        return math.hypot(first.x_m - second.x_m, first.y_m - second.y_m)

    def _advance_entry_navigation(self) -> None:
        decision = self._last_decision
        if decision is None or decision.staging_pose is None:
            self._transition("NAVIGATE_INTERIOR")
            return
        if self._nav_result == "FAILED":
            self.get_logger().warning("entry staging goal failed; resume coverage")
            self._nav_result = None
            self._remember_recent_goal(
                decision.staging_pose,
                "ENTRY_NAV_FAILED",
            )
            self._survey_room_approach_active = False
            self._survey_selection_ready = False
            self._survey_selection_reason = None
            self._last_decision = None
            self._transition("PLAN_COVERAGE")
            return
        if self._nav_result == "SUCCEEDED":
            self._nav_result = None
            self._transition("NAVIGATE_INTERIOR")
            return
        if self._nav_goal_handle is None and not self._nav_goal_pending:
            self._send_goal(decision.staging_pose, "ENTRY")

    def _advance_interior_navigation(self) -> None:
        decision = self._last_decision
        if decision is None:
            self._transition("PLAN_COVERAGE")
            return
        if self._nav_result == "FAILED":
            self.get_logger().warning("interior goal failed; DOA remains disabled and coverage resumes")
            self._nav_result = None
            self._remember_recent_goal(
                decision.interior_pose,
                "INTERIOR_NAV_FAILED",
            )
            self._survey_room_approach_active = False
            self._survey_selection_ready = False
            self._survey_selection_reason = None
            self._last_decision = None
            self._transition("PLAN_COVERAGE")
            return
        if self._nav_result == "SUCCEEDED":
            self._nav_result = None
            self._survey_room_approach_active = False
            self._publish_doa_enabled(False)
            # Re-enter the common 3 m FRESH+XYXY gate on the next tick instead
            # of bypassing it merely because the room interior was reached.
            self._publish_yolo_ready(False)
            self._doa_active_space = None
            self._doa_active_space_at = 0.0
            self._doa_stationary_reference = self._robot_pose()
            self._doa_stationary_since = time.monotonic()
            self._transition("DOA_ACTIVE")
            self.get_logger().info(
                "interior reached: YOLO waits for FRESH<=3m and recent XYXY "
                "lock; DOA waits for FRESH<=6m, XYXY lock, 1.5s "
                "stationarity, and open-map geometry"
            )
            return
        if self._nav_goal_handle is None and not self._nav_goal_pending:
            self._send_goal(decision.interior_pose, "INTERIOR")

    def _enter_yolo_verify(self) -> None:
        """Pause map navigation and allow a close validated camera check."""

        now = time.monotonic()
        range_m = self._fresh_uwb_median_m()
        self._cancel_navigation()
        self._nav_result = None
        self._latest_person_pose = None
        self._latest_person_received_at = 0.0
        self._yolo_verify_started_at = now
        self._yolo_verify_authorized_until = now + float(
            self.get_parameter("yolo_verify_timeout_sec").value
        )
        self._publish_doa_enabled(False)
        self._publish_yolo_ready(True)
        self._write_event("YOLO_VERIFY_ENABLED")
        self._transition("YOLO_VERIFY")
        self.get_logger().info(
            "recent full XYXY lock + stable UWB FRESH within activation range: "
            f"YOLO verification enabled (median UWB={range_m:.2f} m)"
        )

    def _advance_yolo_verify(self) -> None:
        """Approach only a fresh depth-localized person, otherwise resume map search."""

        self._start_person_approach_if_ready()
        if self._state == "NAVIGATE_VICTIM":
            return
        if time.monotonic() < self._yolo_verify_authorized_until:
            return
        self._publish_yolo_ready(False)
        self._latest_person_pose = None
        self._latest_person_received_at = 0.0
        self._yolo_verify_rearm_at = time.monotonic() + float(
            self.get_parameter("yolo_verify_rearm_sec").value
        )
        self.get_logger().info(
            "YOLO verification timed out without a depth-confirmed person; "
            "resume map evidence after rearm delay"
        )
        self._transition("PLAN_COVERAGE")

    def _person_approach_is_ready(self) -> bool:
        """Require a fresh depth pose plus close UWB or safe depth distance.

        UWB is intentionally not used as a continuous controller after Nav2
        begins the approach. A wall, a person, or body rotation can make the
        UWB distance rise even while the visual map target remains valid.
        """

        if self._mission_complete or self._latest_person_pose is None:
            return False
        if time.monotonic() - self._latest_person_received_at > float(
            self.get_parameter("person_pose_timeout_sec").value
        ):
            return False
        robot = self._robot_pose()
        if robot is None:
            return False
        depth_distance_m = self._distance(robot, self._latest_person_pose)
        depth_safe = bool(
            bool(self.get_parameter("person_depth_approach_enabled").value)
            and math.isfinite(depth_distance_m)
            and float(self.get_parameter("person_depth_approach_min_m").value)
            <= depth_distance_m
            <= float(self.get_parameter("person_depth_approach_max_m").value)
        )
        uwb_close = bool(
            self._uwb_status in {"FRESH", "HOLD"}
            and self._last_range_m is not None
            and math.isfinite(self._last_range_m)
            and 0.0 <= self._last_range_m
            <= float(self.get_parameter("person_uwb_enable_range_m").value)
        )
        if self._state == "YOLO_VERIFY":
            # Only accept a pose produced during this short authorization
            # window. The motion handoff itself still needs close UWB or the
            # bounded, depth-confirmed map distance above.
            return (
                self._latest_person_received_at >= self._yolo_verify_started_at
                and time.monotonic() <= self._yolo_verify_authorized_until
                and (uwb_close or depth_safe)
            )
        return uwb_close or depth_safe

    def _start_person_approach_if_ready(self) -> None:
        if not self._person_approach_is_ready():
            return
        person = self._latest_person_pose
        robot = self._robot_pose()
        if person is None or robot is None:
            return
        dx, dy = person.x_m - robot.x_m, person.y_m - robot.y_m
        distance = math.hypot(dx, dy)
        standoff = float(self.get_parameter("person_approach_standoff_m").value)
        tolerance = float(self.get_parameter("person_arrival_tolerance_m").value)
        if distance <= standoff + tolerance:
            self._finish_rescue_arrival("already inside person standoff distance")
            return
        ux, uy = dx / distance, dy / distance
        self._active_person_approach = MapPoint(
            person.x_m - ux * standoff,
            person.y_m - uy * standoff,
            math.atan2(dy, dx),
        )
        self._victim_resume_state = self._state
        self.get_logger().info(
            "depth-confirmed person accepted for Nav2 approach: "
            f"person=({person.x_m:.2f},{person.y_m:.2f}) "
            f"approach=({self._active_person_approach.x_m:.2f},"
            f"{self._active_person_approach.y_m:.2f})"
        )
        self._publish_doa_enabled(False)
        self._transition("NAVIGATE_VICTIM")

    def _advance_person_navigation(self) -> None:
        approach = self._active_person_approach
        if approach is None:
            self._transition(self._victim_resume_state)
            return
        if self._nav_result == "FAILED":
            self.get_logger().warning(
                "person approach goal failed; keep DOA/YOLO active and wait "
                "for a fresh person position"
            )
            self._nav_result = None
            self._active_person_approach = None
            self._latest_person_pose = None
            self._transition(self._victim_resume_state)
            return
        if self._nav_result == "SUCCEEDED":
            self._nav_result = None
            self._finish_rescue_arrival("Nav2 reached person approach standoff")
            return
        if self._nav_goal_handle is None and not self._nav_goal_pending:
            self._send_goal(approach, "PERSON")

    def _finish_rescue_arrival(self, reason: str) -> None:
        self._mission_complete = True
        self._active_person_approach = None
        self._publish_doa_enabled(False)
        self._publish_yolo_ready(False)
        self._publish_mission_complete(True)
        self._transition("RESCUE_ARRIVED")
        self.get_logger().info(f"rescue mission complete: {reason}")

    def _record_range(self, value: float) -> None:
        if not math.isfinite(value) or value < 0.0:
            return
        self._range_values.append(value)
        self._write_event("UWB", uwb_range_m=value)

    def _publish_room_probabilities(self) -> None:
        """Publish all map-section rankings for terminal and RViz monitoring."""

        if self._topology is None:
            return
        self._trim_doa_observations(time.monotonic())
        try:
            snapshot = estimate_zone_probabilities(
                self._topology,
                self._evidence,
                doa_observations=tuple(
                    observation for _stamp, observation in self._doa_observations
                ),
                temperature=float(
                    self.get_parameter("room_probability_temperature").value
                ),
            )
        except ValueError as exc:
            self.get_logger().error(f"room probability update skipped: {exc}")
            return

        selected_zone_id = (
            self._last_decision.zone.zone_id
            if self._last_decision is not None
            else None
        )
        sections = []
        for item in snapshot.sections:
            sections.append({
                "section_id": item.zone.zone_id,
                "probability": round(item.probability, 6),
                "probability_percent": round(item.probability * 100.0, 1),
                "area_m2": round(item.zone.area_m2, 2),
                "max_clearance_m": round(item.zone.max_clearance_m, 2),
                "interior_x_m": round(item.zone.interior.x_m, 3),
                "interior_y_m": round(item.zone.interior.y_m, 3),
                "uwb_fit_cost": _round_or_none(item.fit_cost),
                "uwb_proximity_cost_m": _round_or_none(item.proximity_cost_m),
                # This is 900/1050 when the new metric is available; older
                # evidence uses the four-tone weak-link value as fallback.
                "xyxy_level_dbfs": _round_or_none(item.audio_level_dbfs),
                "xyxy_direction_level_dbfs": _round_or_none(item.audio_level_dbfs),
                "range_score": _round_or_none(item.range_score),
                "audio_score": _round_or_none(item.audio_score),
                "doa_fit_cost": _round_or_none(item.doa_fit_cost),
                "doa_score": _round_or_none(item.doa_score),
                "doa_observations": item.doa_observation_count,
                "combined_score": _round_or_none(item.total_score),
                "selected": item.zone.zone_id == selected_zone_id,
            })
        payload = {
            "frame_id": self._map_frame,
            "state": self._state,
            "probability_kind": "relative_evidence_ranking_not_calibrated",
            "confidence": round(snapshot.confidence, 3),
            "confidence_percent": round(snapshot.confidence * 100.0, 1),
            "valid_uwb_waypoints": snapshot.valid_evidence_count,
            "valid_xyxy_waypoints": snapshot.packet_evidence_count,
            "stable_doa_observations": snapshot.doa_observation_count,
            "observed_sections": snapshot.observed_zone_count,
            "total_sections": len(snapshot.sections),
            "sections": sections,
        }
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        self._room_probability_pub.publish(message)
        self._publish_probability_markers(snapshot, selected_zone_id)

    def _publish_probability_markers(self, snapshot, selected_zone_id: int | None) -> None:
        """Show each auto section's probability directly over the Nav2 map."""

        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        now = self.get_clock().now().to_msg()
        for item in snapshot.sections:
            background = Marker()
            background.header.frame_id = self._map_frame
            background.header.stamp = now
            background.ns = "beacon_room_probability_background"
            background.id = int(item.zone.zone_id)
            background.type = Marker.CUBE
            background.action = Marker.ADD
            background.pose.position.x = item.zone.interior.x_m
            background.pose.position.y = item.zone.interior.y_m
            background.pose.position.z = 0.04
            background.pose.orientation.w = 1.0
            background.scale.x = 0.80
            background.scale.y = 0.52
            background.scale.z = 0.06
            background.color.r = 0.10 + 0.45 * float(1.0 - item.probability)
            background.color.g = 0.10 + 0.45 * float(item.probability)
            background.color.b = 0.10
            background.color.a = 0.92
            markers.markers.append(background)

            marker = Marker()
            marker.header.frame_id = self._map_frame
            marker.header.stamp = now
            marker.ns = "beacon_room_probability"
            marker.id = int(item.zone.zone_id)
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.position.x = item.zone.interior.x_m
            marker.pose.position.y = item.zone.interior.y_m
            marker.pose.position.z = 0.16
            marker.pose.orientation.w = 1.0
            marker.scale.z = 0.32
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            selected = "*" if item.zone.zone_id == selected_zone_id else ""
            marker.text = (
                f"S{item.zone.zone_id}{selected}\n"
                f"{item.probability * 100.0:.0f}%"
            )
            markers.markers.append(marker)
        self._room_probability_marker_pub.publish(markers)

    def _write_event(self, event: str, *, uwb_range_m: float | None = None, packet_metric: dict[str, object] | None = None) -> None:
        pose = self._robot_pose()
        if pose is None:
            return
        sipeed_pose = self._sipeed_pose(pose)
        self._raw_writer.writerow({
            "wall_time_sec": f"{self.get_clock().now().nanoseconds / 1e9:.6f}",
            "monotonic_sec": f"{time.monotonic():.6f}",
            "event": event,
            "state": self._state,
            "waypoint_id": self._current_waypoint_id, "map_x_m": f"{pose.x_m:.3f}", "map_y_m": f"{pose.y_m:.3f}", "map_yaw_rad": f"{pose.yaw_rad:.3f}",
            "sipeed_x_m": f"{sipeed_pose.x_m:.3f}", "sipeed_y_m": f"{sipeed_pose.y_m:.3f}",
            "uwb_status": self._uwb_status,
            "uwb_range_m": "" if uwb_range_m is None else f"{uwb_range_m:.3f}",
            "packet_level_dbfs": "" if packet_metric is None else packet_metric.get("level_dbfs", ""),
            "packet_quality_db": "" if packet_metric is None else packet_metric.get("quality_db", ""),
            "packet_direction_level_dbfs": "" if packet_metric is None else packet_metric.get("direction_level_dbfs", ""),
            "packet_direction_quality_db": "" if packet_metric is None else packet_metric.get("direction_quality_db", ""),
            "packet_locked": int(self._packet_locked), "packet_metric_json": "" if packet_metric is None else json.dumps(packet_metric, sort_keys=True),
        })
        self._streams[0].flush()

    def _robot_pose(self) -> MapPoint | None:
        try:
            transform = self._tf_buffer.lookup_transform(self._map_frame, self._base_frame, rclpy.time.Time())
        except TransformException:
            return None
        t, q = transform.transform.translation, transform.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return MapPoint(float(t.x), float(t.y), yaw)

    def _sipeed_pose(self, base_pose: MapPoint) -> MapPoint:
        """Transform the SIPEED acoustic sensor offset into the map frame."""

        x = float(self.get_parameter("sipeed_sensor_x_m").value)
        y = float(self.get_parameter("sipeed_sensor_y_m").value)
        cos_yaw, sin_yaw = math.cos(base_pose.yaw_rad), math.sin(base_pose.yaw_rad)
        return MapPoint(
            x_m=base_pose.x_m + cos_yaw * x - sin_yaw * y,
            y_m=base_pose.y_m + sin_yaw * x + cos_yaw * y,
            yaw_rad=base_pose.yaw_rad,
        )

    def _send_goal(self, point: MapPoint, kind: str) -> None:
        signature = (
            kind,
            round(point.x_m, 3),
            round(point.y_m, 3),
            round(point.yaw_rad, 3),
        )
        commands_enabled = bool(
            self.get_parameter("navigation_commands_enabled").value
        )
        if not commands_enabled and self._goal_signature == signature:
            return
        if self._topology is None or not self._topology.safe(point):
            self.get_logger().warning(f"{kind} goal rejected by map safety check")
            self._nav_result = "FAILED"
            self._publish_diagnostic_event(
                "NAV_GOAL_REJECTED_MAP",
                {"kind": kind, "goal": self._diagnostic_point(point)},
            )
            return
        nav_costmap_safe = self._nav_costmap_goal_safe(point)
        if nav_costmap_safe is None:
            if not self._nav_costmap_wait_logged:
                self.get_logger().info(
                    "waiting for current Nav2 global costmap before sending "
                    "map-evidence goals"
                )
                self._nav_costmap_wait_logged = True
            return
        if not nav_costmap_safe:
            self.get_logger().warning(
                f"{kind} goal rejected by current Nav2 global costmap; skip waypoint"
            )
            self._nav_result = "FAILED"
            self._publish_diagnostic_event(
                "NAV_GOAL_REJECTED_COSTMAP",
                {"kind": kind, "goal": self._diagnostic_point(point)},
            )
            return
        if not commands_enabled:
            # Dry-run authority boundary: do not call ComputePathToPose,
            # NavigateToPose, cancel_goal_async, or publish /cmd_vel.  Keep
            # the proposed pose visible for RViz/diagnostics only.
            self._cancel_navigation()
            self._goal_kind = kind
            self._goal_signature = signature
            self._nav_result = "DRY_RUN"
            self._proposed_goal_pub.publish(self._pose_stamped(point))
            self.get_logger().warning(
                f"NO-MOTION TEST {kind}: proposed goal only "
                f"({point.x_m:.2f}, {point.y_m:.2f}); Nav2 command blocked"
            )
            self._publish_diagnostic_event(
                "NAV_GOAL_DRY_RUN",
                {"kind": kind, "goal": self._diagnostic_point(point)},
            )
            return
        if (
            bool(self.get_parameter("require_nav_path_preflight").value)
            and not self._nav_path_is_ready(point, signature)
        ):
            return
        if not self._nav_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().warning("Nav2 navigate_to_pose is unavailable")
            self._publish_diagnostic_event(
                "NAV_GOAL_NAV2_UNAVAILABLE",
                {"kind": kind, "goal": self._diagnostic_point(point)},
            )
            return
        if self._goal_signature == signature and (self._nav_goal_pending or self._nav_goal_handle is not None):
            return
        self._cancel_navigation()
        self._goal_kind, self._goal_signature = kind, signature
        self._arm_xyxy_navigation_guard(kind)
        goal = NavigateToPose.Goal()
        goal.pose = self._pose_stamped(point)
        if kind in {"COVERAGE", "XYXY_MAP_SEARCH"}:
            behavior_tree = Path(
                str(self.get_parameter("map_search_behavior_tree").value)
            ).expanduser()
            if not behavior_tree.is_file():
                self.get_logger().error(
                    "map-search no-recovery behavior tree is missing: "
                    f"{behavior_tree}"
                )
                self._nav_result = "FAILED"
                self._publish_diagnostic_event(
                    "NAV_GOAL_REJECTED_BT_MISSING",
                    {
                        "kind": kind,
                        "goal": self._diagnostic_point(point),
                        "behavior_tree": str(behavior_tree),
                    },
                )
                return
            goal.behavior_tree = str(behavior_tree)
        self._nav_goal_pending = True
        self._nav_generation += 1
        generation = self._nav_generation
        future = self._nav_client.send_goal_async(goal)
        future.add_done_callback(
            lambda result, expected=generation: self._goal_response(result, expected)
        )
        self.get_logger().info(f"Nav2 {kind}: ({point.x_m:.2f}, {point.y_m:.2f})")
        self._publish_diagnostic_event(
            "NAV_GOAL_SENT",
            {
                "kind": kind,
                "goal": self._diagnostic_point(point),
                "behavior_tree": (
                    goal.behavior_tree if goal.behavior_tree else "NAV2_DEFAULT"
                ),
            },
        )

    def _arm_xyxy_navigation_guard(self, kind: str) -> None:
        """Capture the last stationary 900/1050-Hz reference for one route."""

        self._xyxy_navigation_samples.clear()
        self._xyxy_degradation_started_at = None
        self._xyxy_goal_reference_level = None
        self._xyxy_goal_start_pose = None
        self._xyxy_goal_started_at = time.monotonic()

        if kind not in {
            "COVERAGE",
            "DOA_PROBE",
            "ENTRY",
            "INTERIOR",
            "XYXY_MAP_SEARCH",
        }:
            return
        if not self._beacon_evidence_seen:
            return

        references: list[float] = []
        if (
            kind != "XYXY_MAP_SEARCH"
            and self._last_stationary_xyxy_level is not None
        ):
            references.append(self._last_stationary_xyxy_level)
        max_age = float(
            self.get_parameter("xyxy_navigation_reference_max_age_sec").value
        )
        if (
            self._latest_xyxy_direction_level is not None
            and time.monotonic() - self._latest_xyxy_direction_at <= max_age
        ):
            references.append(self._latest_xyxy_direction_level)
        start_pose = self._robot_pose()
        if not references or start_pose is None:
            return

        # A route is considered worse only when it falls below both the last
        # settled evidence and any just-observed validated packet strength.
        self._xyxy_goal_reference_level = max(references)
        self._xyxy_goal_start_pose = start_pose
        self.get_logger().info(
            "XYXY route guard armed: "
            f"900/1050 reference={self._xyxy_goal_reference_level:.1f} dBFS"
        )

    def _nav_costmap_goal_safe(self, point: MapPoint) -> bool | None:
        """Return Nav2 costmap safety for a base-center candidate.

        ``None`` means the required costmap has not arrived yet. Nav2's
        ``costmap_raw`` topic uses ``nav2_msgs/Costmap`` with native uint8
        values, where 253--255 mean inscribed/lethal/unknown space.
        """

        costmap = self._nav_costmap
        if costmap is None:
            return None if bool(self.get_parameter("require_nav_costmap").value) else True

        frame_id = costmap.header.frame_id.lstrip("/")
        if frame_id and frame_id != self._map_frame.lstrip("/"):
            self.get_logger().warning(
                "Nav2 global costmap frame does not match map frame: "
                f"{costmap.header.frame_id!r} != {self._map_frame!r}"
            )
            return False

        metadata = costmap.metadata
        if (
            metadata.resolution <= 0.0
            or metadata.size_x <= 0
            or metadata.size_y <= 0
        ):
            return False
        origin = metadata.origin
        origin_yaw = math.atan2(
            2.0 * (origin.orientation.w * origin.orientation.z + origin.orientation.x * origin.orientation.y),
            1.0 - 2.0 * (origin.orientation.y ** 2 + origin.orientation.z ** 2),
        )
        dx = point.x_m - float(origin.position.x)
        dy = point.y_m - float(origin.position.y)
        cos_yaw, sin_yaw = math.cos(origin_yaw), math.sin(origin_yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        col = math.floor(local_x / float(metadata.resolution))
        row = math.floor(local_y / float(metadata.resolution))
        if not (
            0 <= col < int(metadata.size_x)
            and 0 <= row < int(metadata.size_y)
        ):
            return False
        index = row * int(metadata.size_x) + col
        if index >= len(costmap.data):
            return False
        cost = int(costmap.data[index])
        return cost <= int(self.get_parameter("nav_costmap_max_cost").value)

    def _nav_path_is_ready(self, point: MapPoint, signature: tuple) -> bool:
        """Request one non-driving planner preflight for the candidate goal."""

        if self._path_preflight_passed_signature == signature:
            return True
        if self._path_preflight_pending:
            return False
        if not self._path_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().warning("Nav2 compute_path_to_pose is unavailable")
            return False

        goal = ComputePathToPose.Goal()
        goal.goal = self._pose_stamped(point)
        # use_start=False makes Nav2 plan from the current robot pose, not a
        # stale pose saved while the coverage map was generated.
        goal.use_start = False
        self._path_preflight_pending = True
        self._path_preflight_signature = signature
        future = self._path_client.send_goal_async(goal)
        future.add_done_callback(
            lambda result, expected=signature: self._path_preflight_response(
                result, expected
            )
        )
        self.get_logger().info(
            f"Nav2 path preflight {signature[0]}: ({point.x_m:.2f}, {point.y_m:.2f})"
        )
        self._publish_diagnostic_event(
            "NAV_PATH_PREFLIGHT_SENT",
            {
                "kind": signature[0],
                "goal": self._diagnostic_point(point),
                "signature": list(signature),
            },
        )
        return False

    def _path_preflight_response(self, future, signature: tuple) -> None:
        try:
            handle = future.result()
        except Exception as exc:
            self._finish_path_preflight(signature, False, f"request failed: {exc}")
            return
        if not handle.accepted:
            self._finish_path_preflight(signature, False, "goal rejected")
            return
        result_future = handle.get_result_async()
        result_future.add_done_callback(
            lambda result, expected=signature: self._path_preflight_result(
                result, expected
            )
        )

    def _path_preflight_result(self, future, signature: tuple) -> None:
        try:
            action_result = future.result()
            path = action_result.result.path
            success = (
                action_result.status == GoalStatus.STATUS_SUCCEEDED
                # A start==goal request may legitimately return one pose.
                and len(path.poses) >= 1
            )
            reason = "" if success else "planner returned no reachable path"
        except Exception as exc:
            success = False
            reason = f"planner request failed: {exc}"
        self._finish_path_preflight(signature, success, reason)

    def _finish_path_preflight(
        self,
        signature: tuple,
        success: bool,
        reason: str,
    ) -> None:
        if self._path_preflight_signature != signature:
            return
        self._path_preflight_pending = False
        if success:
            self._path_preflight_passed_signature = signature
            self._publish_diagnostic_event(
                "NAV_PATH_PREFLIGHT_PASSED", {"signature": list(signature)}
            )
            return
        self._path_preflight_passed_signature = None
        self._nav_result = "FAILED"
        self.get_logger().warning(
            f"{signature[0]} goal rejected by Nav2 path preflight: {reason}"
        )
        self._publish_diagnostic_event(
            "NAV_PATH_PREFLIGHT_FAILED",
            {"signature": list(signature), "reason": reason},
        )

    def _goal_response(self, future, generation: int) -> None:
        if generation != self._nav_generation:
            return
        try:
            handle = future.result()
        except Exception as exc:
            self._nav_goal_pending = False
            self._nav_result = "FAILED"
            self.get_logger().warning(f"Nav2 goal request failed: {exc}")
            self._publish_diagnostic_event(
                "NAV_GOAL_REQUEST_FAILED", {"reason": str(exc)}
            )
            return
        if not handle.accepted:
            self._nav_goal_pending = False
            self._nav_result = "FAILED"
            self._publish_diagnostic_event("NAV_GOAL_REJECTED_BY_NAV2")
            return
        self._nav_goal_handle = handle
        self._nav_goal_pending = False
        result = handle.get_result_async()
        result.add_done_callback(
            lambda outcome, expected=generation: self._goal_result(outcome, expected)
        )

    def _goal_result(self, future, generation: int) -> None:
        if generation != self._nav_generation:
            return
        self._nav_goal_handle = None
        self._nav_goal_pending = False
        try:
            result = future.result()
            self._nav_result = "SUCCEEDED" if result.status == GoalStatus.STATUS_SUCCEEDED else "FAILED"
        except Exception:
            self._nav_result = "FAILED"
        self._publish_diagnostic_event(
            "NAV_GOAL_RESULT",
            {"result": self._nav_result, "goal_kind": self._goal_kind},
        )

    def _cancel_navigation(self) -> None:
        active = self._nav_goal_handle is not None or self._nav_goal_pending
        previous_kind = self._goal_kind
        previous_signature = self._goal_signature
        self._nav_generation += 1
        if self._nav_goal_handle is not None:
            self._nav_goal_handle.cancel_goal_async()
        self._nav_goal_handle = None
        self._nav_goal_pending = False
        self._goal_signature = None
        self._path_preflight_pending = False
        self._path_preflight_signature = None
        self._path_preflight_passed_signature = None
        self._xyxy_active_goal_index = None
        self._xyxy_active_goal_point = None
        if active:
            self._publish_diagnostic_event(
                "NAV_GOAL_CANCELLED",
                {
                    "goal_kind": previous_kind,
                    "goal_signature": list(previous_signature)
                    if previous_signature else None,
                },
            )

    def _pose_stamped(self, point: MapPoint) -> PoseStamped:
        message = PoseStamped()
        message.header.frame_id = self._map_frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose.position.x, message.pose.position.y = point.x_m, point.y_m
        message.pose.orientation.z = math.sin(point.yaw_rad / 2.0)
        message.pose.orientation.w = math.cos(point.yaw_rad / 2.0)
        return message

    def _publish_doa_enabled(self, enabled: bool) -> None:
        self._doa_enabled = bool(enabled)
        if not rclpy.ok():
            return
        message = Bool(); message.data = enabled; self._doa_enable_pub.publish(message)

    def _publish_yolo_ready(self, ready: bool) -> None:
        self._yolo_ready = bool(ready)
        if not rclpy.ok():
            return
        message = Bool(); message.data = ready; self._yolo_ready_pub.publish(message)

    def _publish_mission_complete(self, complete: bool) -> None:
        if not rclpy.ok():
            return
        message = Bool(); message.data = complete; self._mission_complete_pub.publish(message)

    def _transition(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        self._state_since = time.monotonic()
        message = String(); message.data = state; self._state_pub.publish(message)
        self.get_logger().info(f"map-evidence state -> {state}")
        self._publish_diagnostic_event("STATE_TRANSITION", {"state": state})

    def destroy_node(self):  # type: ignore[override]
        self._publish_diagnostic_event("SESSION_END")
        self._cancel_navigation()
        self._publish_doa_enabled(False)
        self._publish_yolo_ready(False)
        self._publish_mission_complete(False)
        for stream in self._streams:
            stream.close()
        return super().destroy_node()


def _acquire_single_instance_lock():
    """Prevent two search nodes on one TurtleBot PC from owning Nav2."""

    lock_path = "/tmp/beacon_tracker-map-evidence-search.lock"
    stream = open(lock_path, "w", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        print(
            "map_evidence_search_node is already running; stop the existing "
            "launch before starting another one",
            file=sys.stderr,
        )
        return None
    stream.write(f"pid={os.getpid()}\n")
    stream.flush()
    return stream


def main(args=None) -> None:
    instance_lock = _acquire_single_instance_lock()
    if instance_lock is None:
        raise SystemExit(2)
    rclpy.init(args=args)
    node = MapEvidenceSearchNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        fcntl.flock(instance_lock.fileno(), fcntl.LOCK_UN)
        instance_lock.close()


if __name__ == "__main__":
    main()
