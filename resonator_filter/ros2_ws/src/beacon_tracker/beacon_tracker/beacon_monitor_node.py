#!/usr/bin/env python3
"""One-line live monitor for UWB range, packet lock, and beacon DOA.

This node observes the sensor topics only.  It never sends a navigation goal
and never changes the tracker state machine.  It can therefore stay running
at every UWB range as a diagnostic terminal.
"""

from __future__ import annotations

import math
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, String


STATE_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class BeaconMonitorNode(Node):
    """Continuously show the latest UWB, packet, and DOA values."""

    def __init__(self) -> None:
        super().__init__("beacon_monitor_node")

        self.declare_parameter("refresh_sec", 0.25)
        self.declare_parameter("uwb_range_topic", "/uwb/range_m")
        self.declare_parameter("uwb_status_topic", "/uwb/status")
        self.declare_parameter("packet_topic", "/beacon/packet_locked")
        self.declare_parameter("packet_status_topic", "/beacon/packet_status")
        self.declare_parameter("packet_stale_timeout_sec", 12.0)
        self.declare_parameter("doa_angle_topic", "/beacon/doa_angle_rad")
        self.declare_parameter("doa_stable_topic", "/beacon/doa_stable")

        self._uwb_range_m: float | None = None
        self._uwb_status = "NO DATA"
        self._packet_locked: bool | None = None
        self._packet_status = "NO DATA"
        self._doa_angle_rad: float | None = None
        self._doa_stable: bool | None = None

        self._uwb_time: float | None = None
        self._packet_time: float | None = None
        self._doa_time: float | None = None

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
        self.create_subscription(
            Bool,
            str(self.get_parameter("packet_topic").value),
            self._packet_callback,
            STATE_QOS,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("packet_status_topic").value),
            self._packet_status_callback,
            STATE_QOS,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("doa_angle_topic").value),
            self._doa_angle_callback,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("doa_stable_topic").value),
            self._doa_stable_callback,
            10,
        )

        refresh_sec = float(self.get_parameter("refresh_sec").value)
        self._timer = self.create_timer(refresh_sec, self._print_status)
        self.get_logger().info(
            "monitor ready: UWB | PACKET | DOA "
            "(Ctrl+C to stop this monitor only)"
        )

    def _uwb_range_callback(self, message: Float32) -> None:
        self._uwb_range_m = float(message.data)
        self._uwb_time = time.monotonic()

    def _uwb_status_callback(self, message: String) -> None:
        self._uwb_status = message.data.upper()

    def _packet_callback(self, message: Bool) -> None:
        self._packet_locked = bool(message.data)
        self._packet_time = time.monotonic()

    def _packet_status_callback(self, message: String) -> None:
        self._packet_status = message.data.upper()
        self._packet_time = time.monotonic()

    def _doa_angle_callback(self, message: Float32) -> None:
        self._doa_angle_rad = float(message.data)
        self._doa_time = time.monotonic()

    def _doa_stable_callback(self, message: Bool) -> None:
        self._doa_stable = bool(message.data)

    @staticmethod
    def _age_text(timestamp: float | None) -> str:
        if timestamp is None:
            return "no data"
        return f"{time.monotonic() - timestamp:.1f}s ago"

    def _uwb_text(self) -> str:
        if self._uwb_range_m is None:
            return "NO DATA"
        return (
            f"{self._uwb_range_m:.2f} m "
            f"[{self._uwb_status}, {self._age_text(self._uwb_time)}]"
        )

    def _packet_text(self) -> str:
        if self._packet_locked is None:
            return "NO DATA"
        stale_timeout_sec = float(
            self.get_parameter("packet_stale_timeout_sec").value
        )
        packet_age_sec = (
            math.inf
            if self._packet_time is None
            else time.monotonic() - self._packet_time
        )
        if self._packet_locked:
            if packet_age_sec > stale_timeout_sec:
                return (
                    "STALE / last LOCKED "
                    f"[{self._age_text(self._packet_time)}]"
                )
            return f"PASS / LOCKED [{self._age_text(self._packet_time)}]"
        if self._packet_status.startswith(("ERROR", "EXITED")):
            return f"ERROR / {self._packet_status}"
        if self._packet_status == "STARTING":
            return "WAIT / STARTING"
        return f"FAIL / {self._packet_status}"

    def _doa_text(self) -> str:
        if self._doa_angle_rad is None:
            return "NO DATA"
        angle_deg = math.degrees(self._doa_angle_rad)
        stable_text = "STABLE" if self._doa_stable else "UNSTABLE"
        return (
            f"{angle_deg:+.1f} deg ({self._doa_angle_rad:+.3f} rad) "
            f"[{stable_text}, {self._age_text(self._doa_time)}]"
        )

    def _print_status(self) -> None:
        line = (
            f"UWB: {self._uwb_text()} | "
            f"PACKET: {self._packet_text()} | "
            f"DOA: {self._doa_text()}"
        )
        print(f"\r{line:<180}", end="", flush=True)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BeaconMonitorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        print()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
