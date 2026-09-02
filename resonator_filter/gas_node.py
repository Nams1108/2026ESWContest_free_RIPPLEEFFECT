#!/usr/bin/env python3

import serial
import rclpy

from rclpy.node import Node
from std_msgs.msg import Float32, String


class GasSensorNode(Node):

    def __init__(self):
        super().__init__("gas_sensor_node")

        # Arduino Uno와 USB 시리얼 연결
        self.serial_port = serial.Serial(
            "/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_43439353536351200361-if00",
            9600,
            timeout=1
        )

        # 가스 농도(ppm) 전송
        self.gas_pub = self.create_publisher(
            Float32,
            "/gas_value",
            10
        )

        # 위험 상태(Safe / Warning / Danger) 전송
        self.risk_pub = self.create_publisher(
            String,
            "/gas_risk",
            10
        )

        # 0.1초마다 Arduino 데이터 확인
        self.timer = self.create_timer(
            0.1,
            self.read_gas
        )

        self.get_logger().info(
            "Gas Sensor Node Started"
        )


    def read_gas(self):

        try:
            line = (
                self.serial_port
                .readline()
                .decode("utf-8", errors="ignore")
                .strip()
            )

            # Arduino 측정 데이터만 처리
            if not line.startswith("$GAS,"):
                return

            # 예:
            # $GAS,90s,30,0.3ppm,Safe
            parts = line.split(",")

            if len(parts) < 5:
                return

            ppm = float(
                parts[3].replace("ppm", "")
            )

            risk = parts[4]

            # ppm 전송
            ppm_msg = Float32()
            ppm_msg.data = ppm
            self.gas_pub.publish(ppm_msg)

            # 위험 상태 전송
            risk_msg = String()
            risk_msg.data = risk
            self.risk_pub.publish(risk_msg)

            self.get_logger().info(
                f"Gas: {ppm:.1f} ppm / {risk}"
            )

        except Exception as e:
            self.get_logger().error(
                f"Gas serial error: {e}"
            )


def main(args=None):

    rclpy.init(args=args)

    node = GasSensorNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node.serial_port.is_open:
            node.serial_port.close()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
