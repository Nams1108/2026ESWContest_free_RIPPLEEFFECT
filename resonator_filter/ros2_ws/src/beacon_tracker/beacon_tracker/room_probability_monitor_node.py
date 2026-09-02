#!/usr/bin/env python3
"""Terminal monitor for live map-section beacon probabilities.

The search node owns navigation and publishes the JSON state.  This node is
read-only: it only renders that state, so it is safe to run on a second
terminal on either the TurtleBot PC or a Remote PC in the same ROS domain.
"""

from __future__ import annotations

import json
import sys

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


TRANSIENT_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class RoomProbabilityMonitorNode(Node):
    """Render one sorted probability table whenever search evidence changes."""

    def __init__(self) -> None:
        super().__init__("room_probability_monitor_node")
        self.declare_parameter("topic", "/beacon_search/room_probabilities")
        self.declare_parameter("clear_terminal", True)
        self._last_payload = ""
        self.create_subscription(
            String,
            str(self.get_parameter("topic").value),
            self._callback,
            TRANSIENT_QOS,
        )
        self.get_logger().info(
            "room-probability monitor ready: waiting for map-derived sections"
        )

    def _callback(self, message: String) -> None:
        if message.data == self._last_payload:
            return
        try:
            payload = json.loads(message.data)
            sections = payload["sections"]
            if not isinstance(sections, list):
                raise ValueError("sections is not a list")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            self.get_logger().warning("ignored malformed room probability update")
            return
        self._last_payload = message.data

        if bool(self.get_parameter("clear_terminal").value):
            sys.stdout.write("\033[2J\033[H")
        state = str(payload.get("state", "UNKNOWN"))
        confidence = float(payload.get("confidence_percent", 0.0))
        valid_uwb = int(payload.get("valid_uwb_waypoints", 0))
        valid_xyxy = int(payload.get("valid_xyxy_waypoints", 0))
        observed = int(payload.get("observed_sections", 0))
        total = int(payload.get("total_sections", len(sections)))
        print(
            "Map section beacon probability (relative evidence ranking)\n"
            f"state={state} | data confidence={confidence:.1f}% | "
            f"UWB waypoints={valid_uwb} | XYXY-valid={valid_xyxy} | "
            f"observed sections={observed}/{total}\n"
        )
        print(
            f"{'section':>7}  {'beacon':>8}  {'UWB fit':>8}  "
            f"{'near m':>7}  {'XYXY 900/1050':>14}  {'area m²':>8}  status"
        )
        for section in sorted(
            sections,
            key=lambda item: float(item.get("probability", 0.0)),
            reverse=True,
        ):
            fit = _number_or_dash(section.get("uwb_fit_cost"), 2)
            near = _number_or_dash(section.get("uwb_proximity_cost_m"), 2)
            audio = _number_or_dash(
                section.get(
                    "xyxy_direction_level_dbfs",
                    section.get("xyxy_level_dbfs"),
                ),
                1,
            )
            selected = "SELECTED" if section.get("selected") else ""
            print(
                f"{int(section.get('section_id', -1)):>7}  "
                f"{float(section.get('probability_percent', 0.0)):>7.1f}%  "
                f"{fit:>8}  {near:>7}  {audio:>14}  "
                f"{float(section.get('area_m2', 0.0)):>8.2f}  {selected}"
            )
        print(
            "\n주의: 수치는 UWB·XYXY 증거를 섹션 사이에서 정규화한 상대 순위입니다. "
            "UWB NLOS/반사음 때문에 data confidence가 충분히 높아져도 단독 확정에는 사용하지 않습니다.",
            flush=True,
        )


def _number_or_dash(value: object, digits: int) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoomProbabilityMonitorNode()
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
