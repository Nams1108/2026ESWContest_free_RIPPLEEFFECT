#!/usr/bin/env python3
"""Persist one complete UWB/XYXY/Nav2 evidence-search diagnostic session.

This observer never sends a Nav2 goal and never changes the state machine.
It is started with ``map_evidence_search.launch.py`` and writes from launch
until Ctrl+C into one timestamped directory.  CSV is convenient for quick
sorting; JSONL preserves full Nav2 paths and every structured state snapshot.
"""

from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path as FilePath

import rclpy
from action_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import Bool, Float32, String


TRANSIENT_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

GOAL_STATUS_NAMES = {
    0: "UNKNOWN",
    1: "ACCEPTED",
    2: "EXECUTING",
    3: "CANCELING",
    4: "SUCCEEDED",
    5: "CANCELED",
    6: "ABORTED",
}


def _round_or_none(value, digits: int = 4):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _point_payload(position, orientation=None) -> dict[str, float]:
    result = {
        "x_m": round(float(position.x), 4),
        "y_m": round(float(position.y), 4),
        "z_m": round(float(position.z), 4),
    }
    if orientation is not None:
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        result["yaw_rad"] = round(yaw, 4)
    return result


class EvidenceDiagnosticsLoggerNode(Node):
    """Write sensor events, Nav2 paths, and map-evidence decisions together."""

    def __init__(self) -> None:
        super().__init__("evidence_diagnostics_logger_node")
        self._declare_parameters()
        self._latest: dict[str, object] = {
            "state": "NO_DATA",
            "uwb_status": "NO_DATA",
            "packet_status": "NO_DATA",
            "packet_locked": None,
            "packet_outcome": "NO_DATA",
            "direction_level_dbfs": None,
            "direction_delta_db": None,
            "raw_level_dbfs": None,
            "raw_level_delta_db": None,
            "doa_angle_rad": None,
            "doa_stable": None,
            "doa_enabled": None,
            "yolo_ready": None,
            "mission_complete": None,
            "nav_goal_kind": None,
            "nav_result": None,
            "robot_map_x_m": None,
            "robot_map_y_m": None,
            "robot_map_yaw_rad": None,
        }
        self._last_direction_level: float | None = None
        self._last_raw_level: float | None = None

        self._session_dir = self._open_session()
        self._json_stream = (self._session_dir / "events.jsonl").open(
            "w", encoding="utf-8"
        )
        self._path_stream = (self._session_dir / "nav_paths.jsonl").open(
            "w", encoding="utf-8"
        )
        self._csv_stream = (self._session_dir / "timeline.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._snapshot_stream = (self._session_dir / "state_snapshots.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._csv_writer = csv.DictWriter(
            self._csv_stream,
            fieldnames=[
                "wall_time_iso", "wall_time_sec", "monotonic_sec", "event",
                "state", "nav_goal_kind", "nav_goal_x_m", "nav_goal_y_m",
                "nav_result", "robot_map_x_m", "robot_map_y_m", "robot_map_yaw_rad",
                "uwb_range_m", "uwb_status", "packet_locked", "packet_status",
                "packet_outcome", "direction_level_dbfs", "direction_quality_db",
                "direction_delta_db", "raw_level_dbfs", "raw_level_delta_db",
                "doa_angle_rad", "doa_stable", "doa_enabled", "yolo_ready",
                "mission_complete", "payload_json",
            ],
        )
        self._snapshot_writer = csv.DictWriter(
            self._snapshot_stream,
            fieldnames=[
                "wall_time_iso", "monotonic_sec", "state", "state_age_sec",
                "robot_map_x_m", "robot_map_y_m", "robot_map_yaw_rad",
                "nav_goal_kind", "nav_goal_active", "nav_goal_pending", "nav_result",
                "coverage_cycle", "coverage_index", "coverage_total_waypoints",
                "uwb_status", "uwb_range_m", "uwb_link_usable", "uwb_fresh_age_sec",
                "uwb_fresh_samples", "uwb_fresh_median_m", "uwb_hold_age_sec",
                "uwb_degraded_age_sec", "uwb_fresh_history_count",
                "xyxy_locked", "xyxy_direction_level_dbfs",
                "xyxy_direction_quality_db", "xyxy_metric_age_sec",
                "xyxy_goal_reference_level_dbfs", "xyxy_goal_samples",
                "xyxy_history_count", "xyxy_map_search_cycle",
                "xyxy_map_search_index", "xyxy_map_search_queue_size",
                "xyxy_map_search_samples", "xyxy_map_search_target_x_m",
                "xyxy_map_search_target_y_m", "evidence_records",
                "evidence_packet_metrics", "selected_zone_id", "doa_enabled",
                "yolo_ready", "mission_complete", "snapshot_json",
            ],
        )
        self._csv_writer.writeheader()
        self._snapshot_writer.writeheader()

        self._subscribe()
        self._write_manifest()
        self._record("SESSION_START", {"session_dir": str(self._session_dir)})
        self.get_logger().info(
            "evidence diagnostics recording until Ctrl+C: "
            f"{self._session_dir}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("output_dir", "~/resonator_filter/outputs")
        self.declare_parameter("session_prefix", "evidence_diagnostics")
        self.declare_parameter("diagnostic_topic", "/beacon_search/diagnostics")
        self.declare_parameter(
            "diagnostic_config_topic", "/beacon_search/diagnostics_config"
        )
        self.declare_parameter("record_full_nav_paths", True)
        self.declare_parameter("max_path_poses", 5000)

    def _open_session(self) -> FilePath:
        output_dir = FilePath(
            str(self.get_parameter("output_dir").value)
        ).expanduser()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = output_dir / (
            f"{self.get_parameter('session_prefix').value}_{stamp}"
        )
        session_dir.mkdir(parents=True, exist_ok=False)
        return session_dir

    def _write_manifest(self) -> None:
        # Humble's rclpy does not expose Node.list_parameters(), unlike
        # Jazzy.  ``_parameters`` contains every parameter declared above on
        # both platforms and preserves the effective launch-file overrides.
        names = sorted(self._parameters)
        manifest = {
            "started_at": datetime.now().astimezone().isoformat(),
            "node": self.get_name(),
            "session_dir": str(self._session_dir),
            "parameters": {
                name: self.get_parameter(name).value for name in sorted(names)
            },
            "files": {
                "events_jsonl": "events.jsonl",
                "timeline_csv": "timeline.csv",
                "state_snapshots_csv": "state_snapshots.csv",
                "nav_paths_jsonl": "nav_paths.jsonl",
            },
        }
        (self._session_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _subscribe(self) -> None:
        self.create_subscription(
            String,
            str(self.get_parameter("diagnostic_topic").value),
            self._diagnostic_callback,
            50,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("diagnostic_config_topic").value),
            self._diagnostic_config_callback,
            TRANSIENT_QOS,
        )
        self.create_subscription(Float32, "/uwb/range_m", self._uwb_range_callback, 20)
        self.create_subscription(String, "/uwb/status", self._uwb_status_callback, 20)
        self.create_subscription(Bool, "/beacon/packet_locked", self._packet_lock_callback, TRANSIENT_QOS)
        self.create_subscription(String, "/beacon/packet_status", self._packet_status_callback, TRANSIENT_QOS)
        self.create_subscription(String, "/beacon/packet_metric", self._packet_metric_callback, 50)
        self.create_subscription(String, "/beacon/packet_event", self._packet_event_callback, 50)
        self.create_subscription(Float32, "/beacon/doa_angle_rad", self._doa_angle_callback, 20)
        self.create_subscription(Bool, "/beacon/doa_stable", self._doa_stable_callback, 20)
        self.create_subscription(Bool, "/beacon/doa_enabled", self._doa_enabled_callback, TRANSIENT_QOS)
        self.create_subscription(String, "/beacon_search/state", self._state_callback, TRANSIENT_QOS)
        self.create_subscription(Bool, "/beacon_search/yolo_ready", self._yolo_ready_callback, TRANSIENT_QOS)
        self.create_subscription(Bool, "/beacon_search/mission_complete", self._mission_callback, TRANSIENT_QOS)
        self.create_subscription(PoseStamped, "/beacon_search/person_pose", self._person_pose_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self._amcl_callback, 20)
        self.create_subscription(Odometry, "/odom", self._odom_callback, qos_profile_sensor_data)
        self.create_subscription(NavPath, "/plan", self._global_plan_callback, 10)
        self.create_subscription(NavPath, "/local_plan", self._local_plan_callback, 10)
        self.create_subscription(
            GoalStatusArray,
            "/navigate_to_pose/_action/status",
            self._nav_status_callback,
            20,
        )

    def _record(self, event: str, payload: dict[str, object] | None = None) -> None:
        wall_now = datetime.now().astimezone()
        wall_epoch = time.time()
        monotonic_now = time.monotonic()
        payload = payload or {}
        entry = {
            "wall_time_iso": wall_now.isoformat(),
            "wall_time_sec": round(wall_epoch, 6),
            "monotonic_sec": round(monotonic_now, 6),
            "event": event,
            "latest": dict(self._latest),
            "payload": payload,
        }
        encoded = json.dumps(entry, separators=(",", ":"), sort_keys=True)
        self._json_stream.write(encoded + "\n")
        self._csv_writer.writerow({
            "wall_time_iso": entry["wall_time_iso"],
            "wall_time_sec": entry["wall_time_sec"],
            "monotonic_sec": entry["monotonic_sec"],
            "event": event,
            "state": self._latest.get("state"),
            "nav_goal_kind": self._latest.get("nav_goal_kind"),
            "nav_goal_x_m": self._latest.get("nav_goal_x_m"),
            "nav_goal_y_m": self._latest.get("nav_goal_y_m"),
            "nav_result": self._latest.get("nav_result"),
            "robot_map_x_m": self._latest.get("robot_map_x_m"),
            "robot_map_y_m": self._latest.get("robot_map_y_m"),
            "robot_map_yaw_rad": self._latest.get("robot_map_yaw_rad"),
            "uwb_range_m": self._latest.get("uwb_range_m"),
            "uwb_status": self._latest.get("uwb_status"),
            "packet_locked": self._latest.get("packet_locked"),
            "packet_status": self._latest.get("packet_status"),
            "packet_outcome": self._latest.get("packet_outcome"),
            "direction_level_dbfs": self._latest.get("direction_level_dbfs"),
            "direction_quality_db": self._latest.get("direction_quality_db"),
            "direction_delta_db": self._latest.get("direction_delta_db"),
            "raw_level_dbfs": self._latest.get("raw_level_dbfs"),
            "raw_level_delta_db": self._latest.get("raw_level_delta_db"),
            "doa_angle_rad": self._latest.get("doa_angle_rad"),
            "doa_stable": self._latest.get("doa_stable"),
            "doa_enabled": self._latest.get("doa_enabled"),
            "yolo_ready": self._latest.get("yolo_ready"),
            "mission_complete": self._latest.get("mission_complete"),
            "payload_json": json.dumps(payload, separators=(",", ":"), sort_keys=True),
        })
        self._json_stream.flush()
        self._csv_stream.flush()

    def _record_snapshot(self, snapshot: dict[str, object]) -> None:
        nav = snapshot.get("nav", {})
        coverage = snapshot.get("coverage", {})
        uwb = snapshot.get("uwb", {})
        xyxy = snapshot.get("xyxy", {})
        evidence = snapshot.get("evidence", {})
        handoff = snapshot.get("handoff", {})
        robot = snapshot.get("robot_map") or {}
        self._snapshot_writer.writerow({
            "wall_time_iso": datetime.now().astimezone().isoformat(),
            "monotonic_sec": round(time.monotonic(), 6),
            "state": snapshot.get("state"),
            "state_age_sec": snapshot.get("state_age_sec"),
            "robot_map_x_m": robot.get("x_m"),
            "robot_map_y_m": robot.get("y_m"),
            "robot_map_yaw_rad": robot.get("yaw_rad"),
            "nav_goal_kind": nav.get("goal_kind"),
            "nav_goal_active": nav.get("goal_active"),
            "nav_goal_pending": nav.get("goal_pending"),
            "nav_result": nav.get("result"),
            "coverage_cycle": coverage.get("cycle"),
            "coverage_index": coverage.get("index"),
            "coverage_total_waypoints": coverage.get("total_waypoints"),
            "uwb_status": uwb.get("status"),
            "uwb_range_m": uwb.get("range_m"),
            "uwb_link_usable": uwb.get("link_usable"),
            "uwb_fresh_age_sec": uwb.get("fresh_age_sec"),
            "uwb_fresh_samples": uwb.get("fresh_samples"),
            "uwb_fresh_median_m": uwb.get("fresh_median_m"),
            "uwb_hold_age_sec": uwb.get("hold_age_sec"),
            "uwb_degraded_age_sec": uwb.get("degraded_age_sec"),
            "uwb_fresh_history_count": uwb.get("fresh_history_count"),
            "xyxy_locked": xyxy.get("locked"),
            "xyxy_direction_level_dbfs": xyxy.get("direction_level_dbfs"),
            "xyxy_direction_quality_db": xyxy.get("direction_quality_db"),
            "xyxy_metric_age_sec": xyxy.get("direction_metric_age_sec"),
            "xyxy_goal_reference_level_dbfs": xyxy.get("goal_reference_level_dbfs"),
            "xyxy_goal_samples": xyxy.get("goal_samples"),
            "xyxy_history_count": xyxy.get("history_count"),
            "xyxy_map_search_cycle": xyxy.get("map_search_cycle"),
            "xyxy_map_search_index": xyxy.get("map_search_index"),
            "xyxy_map_search_queue_size": xyxy.get("map_search_queue_size"),
            "xyxy_map_search_samples": xyxy.get("map_search_samples"),
            "xyxy_map_search_target_x_m": (xyxy.get("map_search_target") or {}).get("x_m"),
            "xyxy_map_search_target_y_m": (xyxy.get("map_search_target") or {}).get("y_m"),
            "evidence_records": evidence.get("records"),
            "evidence_packet_metrics": evidence.get("packet_metrics_in_window"),
            "selected_zone_id": evidence.get("selected_zone_id"),
            "doa_enabled": handoff.get("doa_enabled"),
            "yolo_ready": handoff.get("yolo_ready"),
            "mission_complete": handoff.get("mission_complete"),
            "snapshot_json": json.dumps(snapshot, separators=(",", ":"), sort_keys=True),
        })
        self._snapshot_stream.flush()

    def _diagnostic_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            self._record("MAP_DIAGNOSTIC_MALFORMED", {"data": message.data})
            return
        event = str(payload.get("event", "UNKNOWN"))
        snapshot = payload.get("snapshot")
        if isinstance(snapshot, dict):
            self._update_from_snapshot(snapshot)
            if event == "SNAPSHOT":
                self._record_snapshot(snapshot)
        self._record(f"MAP_{event}", payload)

    def _diagnostic_config_callback(self, message: String) -> None:
        """Save the latched complete map-evidence parameter set once per run."""

        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            self._record("MAP_CONFIG_MALFORMED", {"data": message.data})
            return
        config_path = self._session_dir / "map_evidence_parameters.json"
        config_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._record("MAP_ALGORITHM_CONFIG", payload)

    def _update_from_snapshot(self, snapshot: dict[str, object]) -> None:
        nav = snapshot.get("nav") if isinstance(snapshot.get("nav"), dict) else {}
        uwb = snapshot.get("uwb") if isinstance(snapshot.get("uwb"), dict) else {}
        xyxy = snapshot.get("xyxy") if isinstance(snapshot.get("xyxy"), dict) else {}
        handoff = snapshot.get("handoff") if isinstance(snapshot.get("handoff"), dict) else {}
        robot = snapshot.get("robot_map") if isinstance(snapshot.get("robot_map"), dict) else {}
        self._latest.update({
            "state": snapshot.get("state", self._latest["state"]),
            "nav_goal_kind": nav.get("goal_kind"),
            "nav_result": nav.get("result"),
            "uwb_status": uwb.get("status", self._latest["uwb_status"]),
            "uwb_range_m": uwb.get("range_m", self._latest.get("uwb_range_m")),
            "packet_locked": xyxy.get("locked", self._latest["packet_locked"]),
            "direction_level_dbfs": xyxy.get("direction_level_dbfs", self._latest["direction_level_dbfs"]),
            "direction_quality_db": xyxy.get("direction_quality_db", self._latest.get("direction_quality_db")),
            "doa_enabled": handoff.get("doa_enabled", self._latest["doa_enabled"]),
            "yolo_ready": handoff.get("yolo_ready", self._latest["yolo_ready"]),
            "mission_complete": handoff.get("mission_complete", self._latest["mission_complete"]),
            "robot_map_x_m": robot.get("x_m", self._latest["robot_map_x_m"]),
            "robot_map_y_m": robot.get("y_m", self._latest["robot_map_y_m"]),
            "robot_map_yaw_rad": robot.get("yaw_rad", self._latest["robot_map_yaw_rad"]),
        })

    def _uwb_range_callback(self, message: Float32) -> None:
        self._latest["uwb_range_m"] = _round_or_none(message.data)
        self._record("UWB_RANGE", {"range_m": self._latest["uwb_range_m"]})

    def _uwb_status_callback(self, message: String) -> None:
        self._latest["uwb_status"] = message.data.upper()
        self._record("UWB_STATUS", {"status": self._latest["uwb_status"]})

    def _packet_lock_callback(self, message: Bool) -> None:
        self._latest["packet_locked"] = bool(message.data)
        self._record("XYXY_LOCK", {"locked": bool(message.data)})

    def _packet_status_callback(self, message: String) -> None:
        status = message.data.upper()
        self._latest["packet_status"] = status
        if status in {"SEARCH", "STARTING"}:
            self._latest["packet_outcome"] = "SEARCH"
        elif status.startswith(("ERROR", "EXITED")):
            self._latest["packet_outcome"] = "ERROR"
        self._record("XYXY_STATUS", {"status": status})

    def _packet_metric_callback(self, message: String) -> None:
        try:
            metric = json.loads(message.data)
        except json.JSONDecodeError:
            self._record("XYXY_METRIC_MALFORMED", {"data": message.data})
            return
        direction_level = _round_or_none(metric.get("direction_level_dbfs"))
        raw_level = _round_or_none(metric.get("level_dbfs"))
        self._latest["direction_delta_db"] = (
            None if direction_level is None or self._last_direction_level is None
            else round(direction_level - self._last_direction_level, 4)
        )
        self._latest["raw_level_delta_db"] = (
            None if raw_level is None or self._last_raw_level is None
            else round(raw_level - self._last_raw_level, 4)
        )
        self._latest["direction_level_dbfs"] = direction_level
        self._latest["direction_quality_db"] = _round_or_none(metric.get("direction_quality_db"))
        self._latest["raw_level_dbfs"] = raw_level
        self._latest["packet_outcome"] = "PASS"
        if direction_level is not None:
            self._last_direction_level = direction_level
        if raw_level is not None:
            self._last_raw_level = raw_level
        self._record("XYXY_METRIC", {"metric": metric})

    def _packet_event_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            self._record("XYXY_EVENT_MALFORMED", {"data": message.data})
            return
        outcome = payload.get("event")
        # ``packet_lock_node`` mirrors a PASS metric to packet_event for
        # external consumers, while /beacon/packet_metric is the canonical
        # source saved above.  Do not count one physical packet twice.
        if outcome == "METRIC":
            return
        if outcome in {"PASS", "FAIL"}:
            self._latest["packet_outcome"] = outcome
        self._record("XYXY_" + str(outcome or "EVENT"), payload)

    def _doa_angle_callback(self, message: Float32) -> None:
        self._latest["doa_angle_rad"] = _round_or_none(message.data)
        self._record("DOA_ANGLE", {"angle_rad": self._latest["doa_angle_rad"]})

    def _doa_stable_callback(self, message: Bool) -> None:
        self._latest["doa_stable"] = bool(message.data)
        self._record("DOA_STABLE", {"stable": bool(message.data)})

    def _doa_enabled_callback(self, message: Bool) -> None:
        self._latest["doa_enabled"] = bool(message.data)
        self._record("DOA_ENABLED", {"enabled": bool(message.data)})

    def _state_callback(self, message: String) -> None:
        self._latest["state"] = message.data
        self._record("STATE_TOPIC", {"state": message.data})

    def _yolo_ready_callback(self, message: Bool) -> None:
        self._latest["yolo_ready"] = bool(message.data)
        self._record("YOLO_READY", {"ready": bool(message.data)})

    def _mission_callback(self, message: Bool) -> None:
        self._latest["mission_complete"] = bool(message.data)
        self._record("MISSION_COMPLETE", {"complete": bool(message.data)})

    def _person_pose_callback(self, message: PoseStamped) -> None:
        self._record("PERSON_POSE", {
            "frame_id": message.header.frame_id,
            "pose": _point_payload(message.pose.position, message.pose.orientation),
        })

    def _amcl_callback(self, message: PoseWithCovarianceStamped) -> None:
        pose = _point_payload(message.pose.pose.position, message.pose.pose.orientation)
        self._latest.update({
            "robot_map_x_m": pose["x_m"],
            "robot_map_y_m": pose["y_m"],
            "robot_map_yaw_rad": pose["yaw_rad"],
        })
        self._record("AMCL_POSE", {"frame_id": message.header.frame_id, "pose": pose})

    def _odom_callback(self, message: Odometry) -> None:
        self._record("ODOM", {
            "frame_id": message.header.frame_id,
            "child_frame_id": message.child_frame_id,
            "pose": _point_payload(message.pose.pose.position, message.pose.pose.orientation),
            "linear_x_mps": _round_or_none(message.twist.twist.linear.x),
            "angular_z_radps": _round_or_none(message.twist.twist.angular.z),
        })

    def _nav_path_payload(self, message: NavPath) -> dict[str, object]:
        poses = message.poses
        path_length = 0.0
        for previous, current in zip(poses, poses[1:]):
            path_length += math.hypot(
                current.pose.position.x - previous.pose.position.x,
                current.pose.position.y - previous.pose.position.y,
            )
        payload: dict[str, object] = {
            "frame_id": message.header.frame_id,
            "pose_count": len(poses),
            "length_m": round(path_length, 4),
            "start": None if not poses else _point_payload(
                poses[0].pose.position, poses[0].pose.orientation
            ),
            "end": None if not poses else _point_payload(
                poses[-1].pose.position, poses[-1].pose.orientation
            ),
        }
        if bool(self.get_parameter("record_full_nav_paths").value):
            maximum = int(self.get_parameter("max_path_poses").value)
            payload["poses"] = [
                _point_payload(item.pose.position, item.pose.orientation)
                for item in poses[:maximum]
            ]
            payload["poses_truncated"] = len(poses) > maximum
        return payload

    def _record_nav_path(self, event: str, message: NavPath) -> None:
        payload = self._nav_path_payload(message)
        entry = {
            "wall_time_iso": datetime.now().astimezone().isoformat(),
            "monotonic_sec": round(time.monotonic(), 6),
            "event": event,
            "path": payload,
        }
        self._path_stream.write(json.dumps(entry, separators=(",", ":"), sort_keys=True) + "\n")
        self._path_stream.flush()
        self._record(event, payload)

    def _global_plan_callback(self, message: NavPath) -> None:
        self._record_nav_path("NAV_GLOBAL_PLAN", message)

    def _local_plan_callback(self, message: NavPath) -> None:
        self._record_nav_path("NAV_LOCAL_PLAN", message)

    def _nav_status_callback(self, message: GoalStatusArray) -> None:
        statuses = [
            {
                "goal_id": "".join(
                    f"{int(byte):02x}" for byte in item.goal_info.goal_id.uuid
                ),
                "status": int(item.status),
                "status_name": GOAL_STATUS_NAMES.get(int(item.status), "UNKNOWN"),
            }
            for item in message.status_list
        ]
        self._record("NAV_ACTION_STATUS", {"statuses": statuses})

    def destroy_node(self):  # type: ignore[override]
        try:
            self._record("SESSION_END", {"final_latest": dict(self._latest)})
        finally:
            for stream in (
                self._json_stream,
                self._path_stream,
                self._csv_stream,
                self._snapshot_stream,
            ):
                stream.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EvidenceDiagnosticsLoggerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
