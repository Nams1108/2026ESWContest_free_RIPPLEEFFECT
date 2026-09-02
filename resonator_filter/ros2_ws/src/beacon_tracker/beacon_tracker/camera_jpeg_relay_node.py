#!/usr/bin/env python3
"""Relay the NUC-owned RealSense colour stream to the Jetson as JPEG.

The RealSense USB device, depth stream, CameraInfo and TF remain on the NUC.
Only this bounded-size colour stream crosses the network to the Jetson.  The
original ROS header is preserved so Jetson detections can be matched back to a
local aligned-depth frame on the NUC.
"""

from __future__ import annotations

import time

try:
    import cv2
    from cv_bridge import CvBridge
except ImportError as exc:
    cv2 = None
    CvBridge = None
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = ""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image


class CameraJpegRelayNode(Node):
    """Encode the local RealSense colour image once for remote inference."""

    def __init__(self) -> None:
        super().__init__("camera_jpeg_relay_node")
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter(
            "jetson_color_topic", "/beacon_search/jetson/color/compressed"
        )
        self.declare_parameter("jpeg_quality", 50)
        self.declare_parameter("max_fps", 10.0)

        self._pub = self.create_publisher(
            CompressedImage,
            str(self.get_parameter("jetson_color_topic").value),
            # Image transport is intentionally best-effort: a slow Jetson
            # must receive the newest frame, not a reliable backlog.
            qos_profile_sensor_data,
        )
        self._max_fps = max(0.0, float(self.get_parameter("max_fps").value))
        self._last_publish_at = 0.0
        if cv2 is None or CvBridge is None:
            self._bridge = None
            self.get_logger().error(
                "camera relay needs cv_bridge and OpenCV: " + IMPORT_ERROR
            )
            return

        self._bridge = CvBridge()
        self.create_subscription(
            Image,
            str(self.get_parameter("color_topic").value),
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            "NUC RealSense JPEG relay ready: "
            f"{self.get_parameter('color_topic').value} -> "
            f"{self.get_parameter('jetson_color_topic').value}"
        )

    def _image_callback(self, image: Image) -> None:
        if self._bridge is None:
            return
        # Do not use NUC CPU for JPEG encoding while the remote inference
        # node is absent.  The local web monitor still receives the raw
        # RealSense topic independently.
        if self._pub.get_subscription_count() == 0:
            return
        now = time.monotonic()
        if (
            self._max_fps > 0.0
            and now - self._last_publish_at < 1.0 / self._max_fps
        ):
            return
        self._last_publish_at = now
        try:
            frame = self._bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
            success, encoded = cv2.imencode(
                ".jpg",
                frame,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    int(self.get_parameter("jpeg_quality").value),
                ],
            )
        except Exception as exc:
            self.get_logger().warning(f"JPEG relay encode failed: {exc}")
            return
        if not success:
            self.get_logger().warning("JPEG relay encode returned no data")
            return
        message = CompressedImage()
        message.header = image.header
        message.format = "jpeg"
        message.data = encoded.tobytes()
        self._pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraJpegRelayNode()
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
