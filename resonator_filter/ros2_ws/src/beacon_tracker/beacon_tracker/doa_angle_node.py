#!/usr/bin/env python3
"""ROS2 adapter for the repository's validated ReSpeaker DOA estimator."""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


def _load_doa_estimator():
    """Load ``doa_angle/doa.py`` without changing the user's DOA files."""

    candidates = [
        Path("/home/lee/resonator_filter"),
        Path(__file__).resolve().parents[4],
    ]
    for repository_root in candidates:
        doa_directory = repository_root / "doa_angle"
        if (doa_directory / "doa.py").exists():
            sys.path.insert(0, str(doa_directory))
            from doa import DOAEstimator  # type: ignore[import-not-found]

            return DOAEstimator
    raise ImportError("doa_angle/doa.py was not found in the repository")


DOAEstimator = _load_doa_estimator()


class DOAAngleNode(Node):
    """Publish stable DOA estimates only while the packet is locked."""

    def __init__(self) -> None:
        super().__init__("doa_angle_node")

        self.declare_parameter("packet_topic", "/beacon/packet_locked")
        self.declare_parameter("uwb_range_topic", "/uwb/range_m")
        self.declare_parameter("uwb_status_topic", "/uwb/status")
        self.declare_parameter("angle_topic", "/beacon/doa_angle_rad")
        self.declare_parameter("stable_topic", "/beacon/doa_stable")
        self.declare_parameter("metric_topic", "/beacon/doa_metric")
        self.declare_parameter("acoustic_range_m", 6.0)
        self.declare_parameter("estimate_period_sec", 1.5)
        self.declare_parameter("require_enable", False)
        self.declare_parameter("enable_topic", "/beacon/doa_enabled")

        self._packet_locked = False
        self._uwb_distance_m: float | None = None
        self._uwb_status = "INVALID"
        self._doa_enabled = not bool(self.get_parameter("require_enable").value)
        self._doa_running = False
        self._result_queue: Queue[
            tuple[float | None, str, dict[str, object] | None]
        ] = Queue()
        self._state_lock = threading.Lock()
        self._estimator = DOAEstimator()
        self._last_angle_rad: float | None = None

        self._angle_pub = self.create_publisher(
            Float32,
            str(self.get_parameter("angle_topic").value),
            10,
        )
        self._stable_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("stable_topic").value),
            10,
        )
        self._metric_pub = self.create_publisher(
            String,
            str(self.get_parameter("metric_topic").value),
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("packet_topic").value),
            self._packet_callback,
            10,
        )
        if bool(self.get_parameter("require_enable").value):
            self.create_subscription(
                Bool,
                str(self.get_parameter("enable_topic").value),
                self._enable_callback,
                10,
            )
        self.create_subscription(
            Float32,
            str(self.get_parameter("uwb_range_topic").value),
            self._uwb_range_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("uwb_status_topic").value),
            self._uwb_status_callback,
            10,
        )

        self._publish_stable(False)
        self._next_estimate_time = 0.0
        self._timer = self.create_timer(0.1, self._timer_callback)
        self.get_logger().info(
            "DOA adapter ready: waiting for packet lock"
            + (" and explicit enable" if bool(self.get_parameter("require_enable").value) else "")
            + "; "
            f"period={float(self.get_parameter('estimate_period_sec').value):.1f}s"
        )

    def _packet_callback(self, message: Bool) -> None:
        locked = bool(message.data)
        if locked == self._packet_locked:
            return

        self._packet_locked = locked
        if not locked:
            self._publish_stable(False)
            self.get_logger().info("packet unlock -> DOA marked unstable")
        else:
            self._next_estimate_time = 0.0
            self.get_logger().info("packet lock -> DOA eligibility updated")

    def _uwb_range_callback(self, message: Float32) -> None:
        self._uwb_distance_m = float(message.data)

    def _uwb_status_callback(self, message: String) -> None:
        self._uwb_status = message.data.upper()
        if not self._doa_allowed():
            self._publish_stable(False)

    def _enable_callback(self, message: Bool) -> None:
        enabled = bool(message.data)
        if enabled == self._doa_enabled:
            return
        self._doa_enabled = enabled
        self._publish_stable(False)
        self.get_logger().info(
            "map-evidence search -> DOA " + ("enabled" if enabled else "disabled")
        )

    def _doa_allowed(self) -> bool:
        acoustic_range = float(
            self.get_parameter("acoustic_range_m").value
        )
        return (
            self._doa_enabled
            and self._packet_locked
            # Map-evidence DOA uses only a currently changing, live UWB link.
            # HOLD/LOST can contain a stale last distance and must not keep an
            # acoustic probability correction active.
            and self._uwb_status == "FRESH"
            and self._uwb_distance_m is not None
            and math.isfinite(self._uwb_distance_m)
            and 0.0 <= self._uwb_distance_m <= acoustic_range
        )

    def _timer_callback(self) -> None:
        self._consume_results()

        if not self._doa_allowed() or self._doa_running:
            if not self._doa_allowed():
                self._publish_stable(False)
            return

        now = time.monotonic()
        if now < self._next_estimate_time:
            return

        self._doa_running = True
        self._next_estimate_time = now + float(
            self.get_parameter("estimate_period_sec").value
        )
        thread = threading.Thread(
            target=self._estimate_worker,
            name="doa-estimate-worker",
            daemon=True,
        )
        thread.start()

    def _estimate_worker(self) -> None:
        try:
            angle_deg = self._estimator.estimate()
            if angle_deg is None:
                self._result_queue.put((None, "no stable DOA estimate", None))
                return
            angle_rad = math.atan2(
                math.sin(math.radians(float(angle_deg))),
                math.cos(math.radians(float(angle_deg))),
            )
            diagnostics = dict(
                getattr(self._estimator, "last_diagnostics", {})
            )
            self._result_queue.put((angle_rad, "ok", diagnostics))
        except Exception as exc:  # Hardware/runtime path.
            self._result_queue.put((None, str(exc), None))
        finally:
            self._doa_running = False

    def _consume_results(self) -> None:
        while True:
            try:
                angle_rad, reason, diagnostics = self._result_queue.get_nowait()
            except Empty:
                return

            if not self._doa_allowed():
                continue
            if angle_rad is None:
                self._publish_stable(False)
                self.get_logger().warning(f"DOA invalid: {reason}")
                continue

            self._last_angle_rad = angle_rad
            angle_message = Float32()
            angle_message.data = float(angle_rad)
            self._angle_pub.publish(angle_message)
            metric_message = String()
            metric_payload = dict(diagnostics or {})
            metric_payload["angle_rad"] = float(angle_rad)
            metric_payload["monotonic_sec"] = time.monotonic()
            metric_message.data = json.dumps(metric_payload, sort_keys=True)
            self._metric_pub.publish(metric_message)
            self._publish_stable(True)
            self.get_logger().info(
                f"DOA stable: {math.degrees(angle_rad):.1f}° "
                f"({angle_rad:.3f} rad)"
            )

    def _publish_stable(self, stable: bool) -> None:
        if not rclpy.ok():
            return
        message = Bool()
        message.data = bool(stable)
        self._stable_pub.publish(message)

    def destroy_node(self):  # type: ignore[override]
        self._packet_locked = False
        self._publish_stable(False)
        self._estimator.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DOAAngleNode()
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
