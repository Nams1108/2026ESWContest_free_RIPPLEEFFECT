#!/usr/bin/env python3
"""Publish BU03 serial distance and freshness status as ROS2 topics."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32, String

try:
    from uwb_reader import (
        BU03RangeReader,
        DistanceFreshnessTracker,
        DEFAULT_BAUDRATE,
        DEFAULT_HOLD_EPSILON_M,
        DEFAULT_HOLD_TIMEOUT_SEC,
        DEFAULT_STARTUP_DELAY_SEC,
        DEFAULT_TIMEOUT_SEC,
        SerialException,
        choose_default_port,
    )
except ImportError:  # Source-tree fallback before the package is installed.
    repository_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repository_root))
    from uwb_reader import (  # type: ignore[no-redef]
        BU03RangeReader,
        DistanceFreshnessTracker,
        DEFAULT_BAUDRATE,
        DEFAULT_HOLD_EPSILON_M,
        DEFAULT_HOLD_TIMEOUT_SEC,
        DEFAULT_STARTUP_DELAY_SEC,
        DEFAULT_TIMEOUT_SEC,
        SerialException,
        choose_default_port,
    )


class UWBRangeNode(Node):
    """Poll ``AT+DISTANCE`` without hiding stale-value status."""

    def __init__(self) -> None:
        super().__init__("uwb_range_node")

        self.declare_parameter("port", choose_default_port())
        self.declare_parameter("baudrate", DEFAULT_BAUDRATE)
        self.declare_parameter("timeout_sec", DEFAULT_TIMEOUT_SEC)
        self.declare_parameter("startup_delay_sec", DEFAULT_STARTUP_DELAY_SEC)
        self.declare_parameter("poll_interval_sec", 0.2)
        self.declare_parameter("reconnect_interval_sec", 2.0)
        self.declare_parameter("hold_timeout_sec", DEFAULT_HOLD_TIMEOUT_SEC)
        self.declare_parameter("hold_epsilon_m", DEFAULT_HOLD_EPSILON_M)
        self.declare_parameter("range_topic", "/uwb/range_m")
        self.declare_parameter("status_topic", "/uwb/status")

        self._port = str(self.get_parameter("port").value)
        self._baudrate = int(self.get_parameter("baudrate").value)
        self._timeout_sec = float(self.get_parameter("timeout_sec").value)
        self._startup_delay_sec = float(
            self.get_parameter("startup_delay_sec").value
        )
        poll_interval = float(self.get_parameter("poll_interval_sec").value)
        self._reconnect_interval_sec = max(
            0.2,
            float(self.get_parameter("reconnect_interval_sec").value),
        )
        hold_timeout = float(self.get_parameter("hold_timeout_sec").value)
        hold_epsilon = float(self.get_parameter("hold_epsilon_m").value)

        self._range_pub = self.create_publisher(
            Float32,
            str(self.get_parameter("range_topic").value),
            10,
        )
        self._status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            10,
        )

        self._reader = BU03RangeReader(
            port=self._port,
            baudrate=self._baudrate,
            response_timeout_sec=self._timeout_sec,
            startup_delay_sec=self._startup_delay_sec,
        )
        self._freshness = DistanceFreshnessTracker(
            hold_timeout_sec=hold_timeout,
            epsilon_m=hold_epsilon,
        )
        self._connected = False
        self._next_reconnect_at = 0.0
        self._last_distance_m: float | None = None
        self._last_error = ""
        self._timer = self.create_timer(poll_interval, self._poll)

        self.get_logger().info(
            f"UWB range node: port={self._port} "
            f"baudrate={self._baudrate} interval={poll_interval:.3f}s"
        )

    def destroy_node(self):  # type: ignore[override]
        self._reader.close()
        return super().destroy_node()

    def _poll(self) -> None:
        if not self._connected:
            if time.monotonic() < self._next_reconnect_at:
                return
            try:
                self._reader.open()
            except (PermissionError, RuntimeError) as exc:
                self._publish_status("LOST")
                self._log_error(str(exc))
                self._schedule_reconnect()
                return
            self._connected = True
            self._last_error = ""
            self.get_logger().info("BU03 serial connection opened")

        try:
            distance_m, _response, _latency_ms = self._reader.request_distance()
        except SerialException as exc:
            # A CH340/BU03 reset can make select() report readable while a
            # subsequent read returns no bytes.  Do not let that USB event
            # terminate the ROS node; close and reopen the stable by-id port.
            self._handle_serial_disconnect(exc)
            return
        except (RuntimeError, TimeoutError) as exc:
            self._publish_status("LOST")
            self._log_error(str(exc))
            return

        freshness, _unchanged_sec = self._freshness.update(distance_m)
        self._last_distance_m = distance_m

        range_msg = Float32()
        range_msg.data = float(distance_m)
        self._range_pub.publish(range_msg)
        self._publish_status(freshness)

    def _handle_serial_disconnect(self, exc: SerialException) -> None:
        """Publish LOST and retry after a physical serial I/O failure."""

        self._reader.close()
        self._connected = False
        self._publish_status("LOST")
        self._schedule_reconnect()
        self._log_error(
            "BU03 serial I/O disconnected; "
            f"retrying {self._port} in {self._reconnect_interval_sec:.1f}s: {exc}"
        )

    def _schedule_reconnect(self) -> None:
        self._next_reconnect_at = time.monotonic() + self._reconnect_interval_sec

    def _publish_status(self, status: str) -> None:
        message = String()
        message.data = status
        self._status_pub.publish(message)

    def _log_error(self, message: str) -> None:
        if message == self._last_error:
            return
        self._last_error = message
        self.get_logger().error(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UWBRangeNode()
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
