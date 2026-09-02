#!/usr/bin/env python3
"""Bridge the validated XYXY decoder's lock state into ROS2.

The existing decoder remains the source of truth for packet classification.
This node translates only its explicit LOCK/UNLOCK events and does not infer
a lock from audio level.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String


LOCK_MARKER = ">>> LOCK = BEACON_1 <<<"
UNLOCK_MARKER = ">>> UNLOCK = BEACON_1 <<<"
KEEPALIVE_MARKER = ">>> LOCK 유지 / keepalive refreshed <<<"
PACKET_METRIC_MARKER = "@@BEACON_PACKET_METRIC@@"
PACKET_RESULT_PATTERN = re.compile(
    r"\[PACKET RESULT\]\s+PASS=(?P<passes>\d+)\s+"
    r"ERASE=(?P<erases>\d+)\s+WRONG=(?P<wrongs>\d+)\s+"
    r"->\s+(?P<outcome>VALID|REJECT)"
)


STATE_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def lock_event_from_line(line: str) -> bool | None:
    """Return the lock state represented by one decoder output line."""

    if LOCK_MARKER in line:
        return True
    if UNLOCK_MARKER in line:
        return False
    return None


def packet_metric_from_line(line: str) -> dict[str, object] | None:
    """Parse a metric emitted only for a fully validated XYXY packet.

    ``direction_*`` is optional to preserve compatibility with a running
    decoder from before the 900/1050 direction metric was added.
    """

    if PACKET_METRIC_MARKER not in line:
        return None
    _, _, payload = line.partition(PACKET_METRIC_MARKER)
    try:
        metric = json.loads(payload.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(metric, dict):
        return None
    if not all(key in metric for key in ("level_dbfs", "quality_db", "raw_dbfs")):
        return None
    return metric


def packet_result_from_line(line: str) -> dict[str, object] | None:
    """Parse both accepted and rejected decoder candidates for diagnostics."""

    match = PACKET_RESULT_PATTERN.search(line)
    if match is None:
        return None
    outcome = match.group("outcome")
    return {
        "event": "PASS" if outcome == "VALID" else "FAIL",
        "decoder_outcome": outcome,
        "followup_passes": int(match.group("passes")),
        "followup_erases": int(match.group("erases")),
        "followup_wrongs": int(match.group("wrongs")),
    }


class PacketLockNode(Node):
    """Run ``beacon1_lock_final.py`` and publish its lock state."""

    def __init__(self) -> None:
        super().__init__("packet_lock_node")

        default_script = Path.home() / "resonator_filter" / "beacon1_lock_final.py"
        self.declare_parameter("detector_script", str(default_script))
        self.declare_parameter("python_executable", sys.executable)
        self.declare_parameter("poll_interval_sec", 0.1)
        self.declare_parameter("packet_topic", "/beacon/packet_locked")
        self.declare_parameter("status_topic", "/beacon/packet_status")
        self.declare_parameter("metric_topic", "/beacon/packet_metric")
        self.declare_parameter("event_topic", "/beacon/packet_event")

        script = Path(str(self.get_parameter("detector_script").value)).expanduser()
        python_executable = str(self.get_parameter("python_executable").value)
        poll_interval = float(self.get_parameter("poll_interval_sec").value)

        self._packet_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("packet_topic").value),
            STATE_QOS,
        )
        self._status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            STATE_QOS,
        )
        self._metric_pub = self.create_publisher(
            String,
            str(self.get_parameter("metric_topic").value),
            20,
        )
        self._event_pub = self.create_publisher(
            String,
            str(self.get_parameter("event_topic").value),
            50,
        )
        self._locked = False
        self._last_exit_code = None
        self._process: subprocess.Popen[str] | None = None
        self._output_queue: Queue[str] = Queue()
        self._reader_thread: threading.Thread | None = None
        self._shutting_down = False

        if not script.exists():
            self.get_logger().error(f"packet detector script not found: {script}")
        else:
            try:
                self._process = subprocess.Popen(
                    [python_executable, "-u", str(script)],
                    cwd=str(script.parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                self._reader_thread = threading.Thread(
                    target=self._read_detector_output,
                    args=(self._process,),
                    name="packet-detector-output-reader",
                    daemon=True,
                )
                self._reader_thread.start()
            except OSError as exc:
                self.get_logger().error(f"packet detector start failed: {exc}")

        self._timer = self.create_timer(poll_interval, self._poll_detector)
        self._publish_lock(False)
        self._publish_status("STARTING" if self._process else "ERROR")
        self.get_logger().info(
            f"packet lock bridge ready: script={script} "
            f"topic={self.get_parameter('packet_topic').value}"
        )

    def destroy_node(self):  # type: ignore[override]
        self._stop_detector()
        return super().destroy_node()

    def _poll_detector(self) -> None:
        process = self._process
        if process is None:
            return

        self._drain_detector_output()

        return_code = process.poll()
        if return_code is not None:
            reader_thread = self._reader_thread
            if reader_thread is not None:
                reader_thread.join(timeout=0.2)
            self._drain_detector_output()
            self._publish_lock(False)
            self._publish_status(f"EXITED:{return_code}")
            if not self._shutting_down and self._last_exit_code != return_code:
                self._last_exit_code = return_code
                self.get_logger().error(
                    f"packet detector stopped with exit code {return_code}"
                )
            self._process = None
            self._reader_thread = None

    def _read_detector_output(self, process: subprocess.Popen[str]) -> None:
        """Read every decoder line without relying on ``select`` buffering.

        ``TextIOWrapper`` may already hold complete lines in its own buffer
        while ``select`` reports no bytes at the OS file descriptor. A blocking
        reader thread avoids dropping delayed LOCK/UNLOCK lines in that case.
        """

        stdout = process.stdout
        if stdout is None:
            return

        try:
            for line in iter(stdout.readline, ""):
                self._output_queue.put(line.rstrip())
        except (OSError, ValueError) as exc:
            self._output_queue.put(f"[packet reader error] {exc}")

    def _drain_detector_output(self) -> None:
        while True:
            try:
                line = self._output_queue.get_nowait()
            except Empty:
                return
            self._handle_detector_line(line)

    def _handle_detector_line(self, line: str) -> None:
        packet_result = packet_result_from_line(line)
        if packet_result is not None:
            packet_result["monotonic_sec"] = round(time.monotonic(), 6)
            packet_result["locked_before_result"] = self._locked
            self._publish_packet_event(packet_result)
            return

        metric = packet_metric_from_line(line)
        if metric is not None:
            message = String()
            message.data = json.dumps(metric, sort_keys=True)
            self._metric_pub.publish(message)
            self._publish_packet_event({
                "event": "METRIC",
                "monotonic_sec": round(time.monotonic(), 6),
                "locked": self._locked,
                "metric": metric,
            })
            return

        event = lock_event_from_line(line)
        if event is not None:
            self._publish_lock(event)
            self._publish_status("LOCKED" if event else "SEARCH")
            self._publish_packet_event({
                "event": "LOCK" if event else "UNLOCK",
                "monotonic_sec": round(time.monotonic(), 6),
                "locked": event,
            })
            return

        # The decoder emits LOCK only once.  Each later valid XYXY packet has
        # this keepalive line instead, so forward it as a fresh LOCKED update
        # for the ROS monitor and for consumers that supervise freshness.
        if KEEPALIVE_MARKER in line:
            self._publish_lock(True)
            self._publish_status("LOCKED")
            self._publish_packet_event({
                "event": "LOCK_KEEPALIVE",
                "monotonic_sec": round(time.monotonic(), 6),
                "locked": True,
            })
            return

        if "error" in line.lower() or "cannot" in line.lower():
            self.get_logger().warning(line)
            self._publish_packet_event({
                "event": "DECODER_ERROR",
                "monotonic_sec": round(time.monotonic(), 6),
                "line": line,
            })

    def _publish_lock(self, locked: bool) -> None:
        self._locked = bool(locked)
        message = Bool()
        message.data = self._locked
        self._packet_pub.publish(message)

    def _publish_status(self, status: str) -> None:
        message = String()
        message.data = status
        self._status_pub.publish(message)

    def _publish_packet_event(self, payload: dict[str, object]) -> None:
        """Keep PASS/FAIL candidates observable without changing detection."""

        message = String()
        message.data = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        self._event_pub.publish(message)

    def _stop_detector(self) -> None:
        self._shutting_down = True
        process = self._process
        if process is None:
            self._process = None
            return

        if process.poll() is not None:
            reader_thread = self._reader_thread
            if reader_thread is not None:
                reader_thread.join(timeout=0.5)
            self._drain_detector_output()
            self._process = None
            self._reader_thread = None
            return

        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
        finally:
            reader_thread = self._reader_thread
            if reader_thread is not None:
                reader_thread.join(timeout=0.5)
            self._drain_detector_output()
            self._process = None
            self._reader_thread = None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PacketLockNode()
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
