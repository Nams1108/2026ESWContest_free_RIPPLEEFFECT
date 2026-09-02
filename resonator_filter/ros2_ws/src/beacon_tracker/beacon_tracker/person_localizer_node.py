#!/usr/bin/env python3
"""NUC-side depth/TF localization for Jetson YOLO detections.

The NUC owns the RealSense depth stream and all map TF.  Jetson sends only a
timestamped image bounding box; a UWB value is never substituted for camera
depth.  The resulting stable map pose is handed to map_evidence_search_node,
which remains the only node allowed to issue a Nav2 goal.
"""

from __future__ import annotations

from collections import deque
import json
import math
import time
from typing import Deque

try:
    import numpy as np
    from cv_bridge import CvBridge
except ImportError as exc:
    np = None
    CvBridge = None
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = ""

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32, String
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener


TRANSIENT_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1e9


class PersonLocalizerNode(Node):
    """Convert a Jetson person bounding box into a stable map-frame pose."""

    def __init__(self) -> None:
        super().__init__("person_localizer_node")
        self._declare_parameters()
        self._enabled = False
        self._range_m: float | None = None
        self._uwb_status = "INVALID"
        self._intrinsics: tuple[float, float, float, float] | None = None
        self._depth_frames: Deque[tuple[float, object, object]] = deque(
            maxlen=int(self.get_parameter("depth_buffer_size").value)
        )
        self._history: Deque[tuple[float, PointStamped, float]] = deque(
            maxlen=int(self.get_parameter("min_stable_detections").value)
        )
        self._last_detection_stamp: tuple[int, int] | None = None
        self._target_active = False
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._bridge = None if CvBridge is None else CvBridge()

        self._pose_pub = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("person_pose_topic").value),
            10,
        )
        self._detected_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("person_detected_topic").value),
            TRANSIENT_QOS,
        )
        # Retain the exact HTTP-facing status topic used by the teammate code.
        self._status_pub = self.create_publisher(
            String,
            str(self.get_parameter("person_status_topic").value),
            10,
        )
        self._alert_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("person_alert_topic").value),
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("enable_topic").value),
            self._enable_callback,
            TRANSIENT_QOS,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("uwb_range_topic").value),
            self._range_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("uwb_status_topic").value),
            self._status_callback,
            10,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("depth_topic").value),
            self._depth_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("detection_topic").value),
            self._detection_callback,
            10,
        )
        self._publish_detected(False, force=True)
        self._publish_alert(False)
        self._publish_status(
            False,
            None,
            None,
            None,
            "waiting for UWB gate, Jetson detection, aligned depth and CameraInfo",
        )
        if self._bridge is None or np is None:
            self.get_logger().error(
                "NUC person localizer needs cv_bridge and NumPy: " + IMPORT_ERROR
            )
        self.get_logger().info(
            "NUC person localizer ready: Jetson boxes + local aligned depth + TF "
            "-> /beacon_search/person_pose"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("enable_topic", "/beacon_search/yolo_ready")
        self.declare_parameter("uwb_range_topic", "/uwb/range_m")
        self.declare_parameter("uwb_status_topic", "/uwb/status")
        self.declare_parameter("uwb_enable_range_m", 3.0)
        self.declare_parameter(
            "detection_topic", "/beacon_search/jetson/yolo_detection"
        )
        self.declare_parameter(
            "depth_topic", "/camera/camera/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter(
            "camera_info_topic", "/camera/camera/color/camera_info"
        )
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("depth_buffer_size", 30)
        self.declare_parameter("depth_sync_tolerance_sec", 0.25)
        self.declare_parameter("min_depth_m", 0.35)
        self.declare_parameter("max_depth_m", 4.0)
        self.declare_parameter("min_stable_detections", 3)
        self.declare_parameter("max_target_spread_m", 0.45)
        self.declare_parameter("safety_distance_m", 1.0)
        self.declare_parameter("person_pose_topic", "/beacon_search/person_pose")
        self.declare_parameter("person_detected_topic", "/beacon_search/person_detected")
        self.declare_parameter(
            "person_status_topic", "/victim_detector/target_status"
        )
        self.declare_parameter("person_alert_topic", "/person_alert_signal")

    def _enable_callback(self, message: Bool) -> None:
        enabled = bool(message.data)
        if enabled == self._enabled:
            return
        self._enabled = enabled
        self._clear_target("YOLO enabled" if enabled else "YOLO disabled")

    def _range_callback(self, message: Float32) -> None:
        self._range_m = float(message.data)
        if not self._gate_open():
            self._clear_target("outside UWB vision gate")

    def _status_callback(self, message: String) -> None:
        self._uwb_status = message.data.upper()
        if not self._gate_open():
            self._clear_target("UWB status invalid")

    def _camera_info_callback(self, message: CameraInfo) -> None:
        if message.k[0] > 0.0 and message.k[4] > 0.0:
            self._intrinsics = (
                float(message.k[0]),
                float(message.k[4]),
                float(message.k[2]),
                float(message.k[5]),
            )

    def _depth_callback(self, message: Image) -> None:
        if self._bridge is None:
            return
        try:
            depth = self._bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            self._depth_frames.append((_stamp_seconds(message.header.stamp), depth, message.header))
        except Exception as exc:
            self.get_logger().warning(f"aligned depth conversion failed: {exc}")

    def _detection_callback(self, message: String) -> None:
        try:
            data = json.loads(message.data)
            stamp_data = data["stamp"]
            stamp_key = (int(stamp_data["sec"]), int(stamp_data["nanosec"]))
            enabled = bool(data.get("enabled", False))
            fresh = bool(data.get("inference_fresh", False))
            best = data.get("best")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"invalid Jetson detection message: {exc}")
            return

        if not self._gate_open() or not enabled:
            self._clear_target("Jetson/NUC YOLO gate closed")
            return
        if best is None:
            self._clear_target("no confident person")
            return

        bbox = best.get("bbox")
        confidence = float(best.get("confidence", 0.0))
        if not isinstance(bbox, list) or len(bbox) != 4:
            self._clear_target("invalid Jetson person bounding box")
            return
        if not fresh:
            self._publish_status(
                False,
                bbox,
                None,
                confidence,
                "rendered cached Jetson detection; waiting for fresh inference",
            )
            return
        # A duplicate may occur if DDS reconnects. Never fabricate another
        # stable sample from the same image.
        if stamp_key == self._last_detection_stamp:
            return
        self._last_detection_stamp = stamp_key
        if self._intrinsics is None:
            self._publish_status(False, bbox, None, confidence, "waiting for CameraInfo")
            return

        point, depth_m = self._project_to_map(
            stamp_key,
            [int(value) for value in bbox],
        )
        if point is None:
            self._publish_status(
                False, bbox, None, confidence, "no aligned depth or camera-to-map TF"
            )
            return
        self._history.append((time.monotonic(), point, confidence))
        stable = self._stable_person_point()
        if stable is None:
            self._publish_status(
                False, bbox, point, confidence, "collecting stable depth/TF detections", depth_m
            )
            self._publish_detected(False)
            self._publish_alert(False)
            return

        pose = PoseStamped()
        pose.header = stable.header
        pose.pose.position.x = stable.point.x
        pose.pose.position.y = stable.point.y
        pose.pose.position.z = stable.point.z
        pose.pose.orientation.w = 1.0
        self._pose_pub.publish(pose)
        self._publish_status(
            True, bbox, stable, confidence, "stable person map pose", depth_m
        )
        self._publish_detected(True)
        self._publish_alert(depth_m <= float(self.get_parameter("safety_distance_m").value))

    def _gate_open(self) -> bool:
        return (
            self._enabled
            and self._uwb_status in {"FRESH", "HOLD"}
            and self._range_m is not None
            and math.isfinite(self._range_m)
            and 0.0 <= self._range_m
            <= float(self.get_parameter("uwb_enable_range_m").value)
        )

    def _matching_depth(self, stamp_key: tuple[int, int]):
        target = float(stamp_key[0]) + float(stamp_key[1]) / 1e9
        if not self._depth_frames:
            return None
        best = min(self._depth_frames, key=lambda item: abs(item[0] - target))
        if abs(best[0] - target) > float(
            self.get_parameter("depth_sync_tolerance_sec").value
        ):
            return None
        return best

    def _project_to_map(self, stamp_key: tuple[int, int], bbox: list[int]):
        depth_data = self._matching_depth(stamp_key)
        if depth_data is None:
            return None, None
        _, frame, header = depth_data
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        u = int(max(0, min(width - 1, round((x1 + x2) / 2.0))))
        v = int(max(0, min(height - 1, round((y1 + y2) / 2.0))))
        half = max(2, int(min(abs(x2 - x1), abs(y2 - y1)) * 0.10))
        roi = frame[
            max(0, v - half) : min(height, v + half + 1),
            max(0, u - half) : min(width, u + half + 1),
        ]
        values = roi[np.isfinite(roi) & (roi > 0)]
        if values.size == 0:
            return None, None
        depth_m = float(np.median(values))
        if frame.dtype == np.uint16:
            depth_m /= 1000.0
        if not (
            float(self.get_parameter("min_depth_m").value)
            <= depth_m
            <= float(self.get_parameter("max_depth_m").value)
        ):
            return None, depth_m
        fx, fy, cx, cy = self._intrinsics
        point = PointStamped()
        point.header = header
        point.point.x = (u - cx) * depth_m / fx
        point.point.y = (v - cy) * depth_m / fy
        point.point.z = depth_m
        try:
            transform = self._tf_buffer.lookup_transform(
                str(self.get_parameter("map_frame").value),
                header.frame_id,
                Time.from_msg(header.stamp),
                timeout=Duration(seconds=0.10),
            )
        except TransformException as exc:
            self.get_logger().debug(f"camera-to-map transform unavailable: {exc}")
            return None, depth_m
        return do_transform_point(point, transform), depth_m

    def _stable_person_point(self):
        needed = int(self.get_parameter("min_stable_detections").value)
        if len(self._history) < needed:
            return None
        points = [item[1] for item in self._history]
        x = float(np.median([point.point.x for point in points]))
        y = float(np.median([point.point.y for point in points]))
        z = float(np.median([point.point.z for point in points]))
        spread = max(
            math.hypot(point.point.x - x, point.point.y - y) for point in points
        )
        if spread > float(self.get_parameter("max_target_spread_m").value):
            return None
        stable = PointStamped()
        stable.header.frame_id = str(self.get_parameter("map_frame").value)
        stable.header.stamp = self.get_clock().now().to_msg()
        stable.point.x, stable.point.y, stable.point.z = x, y, z
        return stable

    def _publish_detected(self, detected: bool, *, force: bool = False) -> None:
        if not force and detected == self._target_active:
            return
        self._target_active = detected
        message = Bool()
        message.data = detected
        self._detected_pub.publish(message)

    def _publish_alert(self, alert: bool) -> None:
        message = Bool()
        message.data = alert
        self._alert_pub.publish(message)

    def _publish_status(
        self,
        stable: bool,
        bbox,
        point,
        confidence: float | None,
        reason: str,
        depth_m: float | None = None,
    ) -> None:
        # Keep the legacy keys consumed by the existing HTTP frontend.
        payload = {
            "target_detected": stable,
            "bbox": [] if bbox is None else [int(value) for value in bbox],
            "person_depth_m": -1.0 if depth_m is None else round(float(depth_m), 3),
            "person_map_x": -999.0
            if point is None
            else round(float(point.point.x), 3),
            "person_map_y": -999.0
            if point is None
            else round(float(point.point.y), 3),
            "uwb_range_m": self._range_m,
            "uwb_status": self._uwb_status,
            "confidence": confidence,
            "reason": reason,
        }
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"))
        self._status_pub.publish(message)

    def _clear_target(self, reason: str) -> None:
        if not self._history and not self._target_active:
            return
        self._history.clear()
        self._last_detection_stamp = None
        self._publish_detected(False)
        self._publish_alert(False)
        self._publish_status(False, None, None, None, reason)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PersonLocalizerNode()
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
