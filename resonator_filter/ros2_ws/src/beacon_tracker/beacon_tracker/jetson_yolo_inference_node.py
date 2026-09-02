#!/usr/bin/env python3
"""Jetson-only YOLO inference for the NUC RealSense relay.

This process deliberately has no depth, TF, AMCL, Nav2 action or cmd_vel
dependency.  It preserves the team's HTTP-facing image/count topics and sends
only JSON bounding boxes back to the NUC.  The NUC localizer is the sole owner
of depth lookup and map-coordinate projection.
"""

from __future__ import annotations

import json
import math

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
except ImportError as exc:
    cv2 = None
    np = None
    YOLO = None
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = ""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32, Int32, String


TRANSIENT_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class JetsonYoloInferenceNode(Node):
    """Run person-only YOLO and retain existing HTTP video data topics."""

    def __init__(self) -> None:
        super().__init__("jetson_yolo_inference_node")
        self._declare_parameters()
        self._enabled = False
        self._range_m: float | None = None
        self._uwb_status = "INVALID"
        self._frame_counter = 0
        self._cached_detections: list[dict[str, object]] = []
        self._model = None
        self._device = "cpu"

        self._detection_pub = self.create_publisher(
            String,
            str(self.get_parameter("detection_topic").value),
            10,
        )
        self._count_pub = self.create_publisher(
            Int32,
            str(self.get_parameter("person_count_topic").value),
            10,
        )
        self._debug_pub = self.create_publisher(
            CompressedImage,
            str(self.get_parameter("debug_image_topic").value),
            # Video is live telemetry.  Dropping an old frame is preferable
            # to delaying every later frame while a network peer is busy.
            qos_profile_sensor_data,
        )
        self._legacy_yolo_pub = self.create_publisher(
            CompressedImage,
            str(self.get_parameter("legacy_yolo_image_topic").value),
            qos_profile_sensor_data,
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
            CompressedImage,
            str(self.get_parameter("input_image_topic").value),
            self._image_callback,
            qos_profile_sensor_data,
        )

        if YOLO is None or cv2 is None or np is None:
            self.get_logger().error(
                "Jetson YOLO dependencies are unavailable: " + IMPORT_ERROR
            )
        else:
            self._model = self._load_model()
        self.get_logger().info(
            "Jetson YOLO inference ready: JPEG in, HTTP image/count out, "
            "bounding boxes back to NUC"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("enable_topic", "/beacon_search/yolo_ready")
        self.declare_parameter("uwb_range_topic", "/uwb/range_m")
        self.declare_parameter("uwb_status_topic", "/uwb/status")
        self.declare_parameter("uwb_enable_range_m", 3.0)
        self.declare_parameter(
            "input_image_topic", "/beacon_search/jetson/color/compressed"
        )
        self.declare_parameter(
            "detection_topic", "/beacon_search/jetson/yolo_detection"
        )
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("device", "auto")
        self.declare_parameter("image_size", 320)
        self.declare_parameter("process_every_n_frames", 3)
        self.declare_parameter("confidence_threshold", 0.55)
        # Preserve the topics already consumed by the HTTP web monitor.
        self.declare_parameter("person_count_topic", "/detected_person_count")
        self.declare_parameter("debug_image_topic", "/victim_detector/compressed")
        self.declare_parameter(
            "legacy_yolo_image_topic", "/yolo/processed_image/compressed"
        )
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("debug_jpeg_quality", 50)

    def _load_model(self):
        try:
            model = YOLO(str(self.get_parameter("model_path").value))
            requested = str(self.get_parameter("device").value).lower()
            if requested == "auto":
                try:
                    model.to("cuda")
                    self._device = 0
                    self.get_logger().info("Jetson YOLO uses CUDA")
                except Exception:
                    self._device = "cpu"
                    self.get_logger().warning("CUDA unavailable; Jetson YOLO uses CPU")
            else:
                self._device = 0 if requested in {"cuda", "gpu", "0"} else requested
            return model
        except Exception as exc:
            self.get_logger().error(f"Jetson YOLO model load failed: {exc}")
            return None

    def _enable_callback(self, message: Bool) -> None:
        self._enabled = bool(message.data)
        if not self._enabled:
            self._cached_detections = []

    def _range_callback(self, message: Float32) -> None:
        self._range_m = float(message.data)
        if not self._gate_open():
            self._cached_detections = []

    def _status_callback(self, message: String) -> None:
        self._uwb_status = message.data.upper()
        if not self._gate_open():
            self._cached_detections = []

    def _gate_open(self) -> bool:
        return (
            self._enabled
            and self._uwb_status in {"FRESH", "HOLD"}
            and self._range_m is not None
            and math.isfinite(self._range_m)
            and 0.0 <= self._range_m
            <= float(self.get_parameter("uwb_enable_range_m").value)
        )

    def _image_callback(self, message: CompressedImage) -> None:
        if cv2 is None or np is None:
            return
        raw = np.frombuffer(message.data, dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning("Jetson received an undecodable JPEG frame")
            return

        gate_open = self._gate_open() and self._model is not None
        run_inference = False
        if gate_open:
            self._frame_counter += 1
            every = max(1, int(self.get_parameter("process_every_n_frames").value))
            run_inference = (
                self._frame_counter % every == 0 or not self._cached_detections
            )
            if run_inference:
                self._cached_detections = self._detect_people(frame)
        else:
            self._cached_detections = []
        detections = list(self._cached_detections)
        best = self._choose_target(detections, frame.shape[1])
        self._publish_count(len(detections))
        self._publish_detection(message, frame, gate_open, run_inference, detections, best)
        self._publish_http_image(frame, message, gate_open, detections, best)

    def _detect_people(self, frame) -> list[dict[str, object]]:
        try:
            results = self._model(
                frame,
                classes=[0],
                conf=float(self.get_parameter("confidence_threshold").value),
                verbose=False,
                imgsz=int(self.get_parameter("image_size").value),
                device=self._device,
            )
        except Exception as exc:
            self.get_logger().warning(f"Jetson YOLO inference failed: {exc}")
            return []
        found: list[dict[str, object]] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
                found.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "confidence": float(box.conf.cpu().item()),
                    }
                )
        return found

    @staticmethod
    def _choose_target(detections, width: int):
        if not detections:
            return None
        return max(
            detections,
            key=lambda item: float(item["confidence"])
            - 0.15
            * abs(
                ((item["bbox"][0] + item["bbox"][2]) / 2.0 - width / 2.0)
                / max(1.0, width / 2.0)
            ),
        )

    def _publish_count(self, count: int) -> None:
        message = Int32()
        message.data = count
        self._count_pub.publish(message)

    def _publish_detection(
        self,
        image: CompressedImage,
        frame,
        enabled: bool,
        inference_fresh: bool,
        detections,
        best,
    ) -> None:
        payload = {
            "stamp": {
                "sec": int(image.header.stamp.sec),
                "nanosec": int(image.header.stamp.nanosec),
            },
            "frame_id": image.header.frame_id,
            "image_width": int(frame.shape[1]),
            "image_height": int(frame.shape[0]),
            "enabled": enabled,
            "inference_fresh": inference_fresh,
            "person_count": len(detections),
            "best": None
            if best is None
            else {
                "bbox": [int(value) for value in best["bbox"]],
                "confidence": round(float(best["confidence"]), 4),
            },
        }
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"))
        self._detection_pub.publish(message)

    def _publish_http_image(self, frame, source, enabled: bool, detections, best) -> None:
        if not bool(self.get_parameter("publish_debug_image").value):
            return
        rendered = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            # The HTTP stream keeps the teammate's red real-person boxes.
            cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                rendered,
                f"PERSON {float(detection['confidence']):.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )
        distance = "--" if self._range_m is None else f"{self._range_m:.2f}m"
        label = "LOCKED" if enabled else "UNLOCKED"
        cv2.putText(
            rendered,
            f"{label} - UWB: {distance}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 255, 0) if enabled else (0, 0, 255),
            2,
        )
        success, encoded = cv2.imencode(
            ".jpg",
            rendered,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                int(self.get_parameter("debug_jpeg_quality").value),
            ],
        )
        if not success:
            return
        message = CompressedImage()
        message.header = source.header
        message.format = "jpeg"
        message.data = encoded.tobytes()
        self._debug_pub.publish(message)
        self._legacy_yolo_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JetsonYoloInferenceNode()
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
