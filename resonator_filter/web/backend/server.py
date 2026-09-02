import asyncio
from collections import Counter, deque
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import BatteryState, CompressedImage, Image
from std_msgs.msg import Int32, String, Float32, Bool


# ============================================================
# 기본 경로 및 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
PROJECT_DIR = BASE_DIR.parents[1]

# 제출 저장소의 맵을 기본 사용합니다. 운영 환경에서 다른 맵을 사용할 때는
# BEACON_MAP_YAML=/absolute/path/map.yaml 로 덮어쓸 수 있습니다.
MAP_YAML_PATH = Path(
    os.environ.get(
        "BEACON_MAP_YAML",
        str(PROJECT_DIR / "maps" / "beacon.yaml"),
    )
).expanduser()


# ============================================================
# 구조 대상 중복 감지 설정
# ============================================================

# 같은 사람으로 판단할 거리
# 기존 구조 대상 발견 위치 기준 1.5m 이내
# YOLO rescue-target confirmation settings.
# The count is accepted only after repeated matching detections
# during one /beacon_search/yolo_ready session.
YOLO_COUNT_WINDOW = 7
YOLO_COUNT_MIN_SAMPLES = 5
YOLO_COUNT_MIN_AGREEMENT = 4



# ============================================================
# 전체 시스템 데이터
# ============================================================

system_data = {

    "robot_status": {
        "status": "운행 대기",
        "battery": 100,
        "x": 0.0,
        "y": 0.0,
        "yaw": 0.0,
    },

    "robot": {
        "connected": False,
        "state": "READY",
        "x": 0.0,
        "y": 0.0,
        "yaw_deg": 0.0,
        "battery": 100,
    },

    "camera": {
        "connected": False,
        "stream_url": "/video_feed",
    },

    "yolo": {
        "connected": False,
        "detected": False,
        "count": 0,
        "confidence": None,
        "pose": "-",
        "alert": "",
        # Raw current-frame count is kept in "count".  The web rescue
        # target list uses only the confirmed session count below.
        "active": False,
        "count_stable": False,
        "rescue_target_count": 0,
    },

    "gas": {
        "connected": False,
        "value": None,
        "unit": "raw",
        "risk": "연결 대기",
    },

    "beacon": {
        "connected": False,
        "detected": False,
        "id": None,
        "direction_deg": None,
        "tracking": False,
        # Validated XYXY packet bridge (/beacon/packet_*).
        "packet": {
            "status": "STARTING",
            "locked": False,
            # Full four-tone weak-link metric. It remains the conservative
            # XYXY identity/packet-health display.
            "level_dbfs": None,
            "quality_db": None,
            # Published only after a full XYXY packet passes. This uses the
            # SIPEED direction tones: X/900 Hz and Y/1050 Hz.
            "direction_level_dbfs": None,
            "direction_quality_db": None,
            "raw_dbfs": {},
            "scores_db": {},
            "last_valid_at": None,
        },
    },

    "uwb": {
        "connected": False,
        "model": "BU03",
        "distance_m": None,
    },

    "map": {
        "connected": False,
        "image_url": "/map_image",
        "resolution": None,
        "origin": None,
        "width": None,
        "height": None,
    },

    # 발견된 구조 대상
    "victims": [],

    # TurtleBot 이동 경로
    "path": [],

    # 이벤트 로그
    "events": [],
}


# ============================================================
# 이벤트 기록 함수
# ============================================================

def add_event(message: str):

    time_str = time.strftime("%H:%M:%S")

    formatted_msg = f"[{time_str}] {message}"

    if (
        not system_data["events"]
        or system_data["events"][0] != formatted_msg
    ):

        system_data["events"].insert(
            0,
            formatted_msg
        )

        system_data["events"] = (
            system_data["events"][:50]
        )


add_event(
    "웹 모니터링 서버가 시작되었습니다."
)


# ============================================================
# 지도 불러오기
# ============================================================

map_image_path: Optional[Path] = None


def load_map():

    global map_image_path

    try:

        if not MAP_YAML_PATH.exists():

            print(
                f"[MAP] YAML 파일 없음: {MAP_YAML_PATH}"
            )

            add_event(
                "지도 YAML 파일을 찾을 수 없습니다."
            )

            return


        with MAP_YAML_PATH.open(
            "r",
            encoding="utf-8"
        ) as f:

            map_config = yaml.safe_load(f)


        image_name = map_config.get("image")
        resolution = map_config.get("resolution")
        origin = map_config.get("origin")


        if not image_name:

            raise ValueError(
                "map yaml에 image 항목이 없습니다."
            )


        image_path = Path(image_name)


        if not image_path.is_absolute():

            image_path = (
                MAP_YAML_PATH.parent
                /
                image_path
            )


        map_image_path = image_path.resolve()


        image = cv2.imread(
            str(map_image_path),
            cv2.IMREAD_GRAYSCALE
        )


        if image is None:

            raise FileNotFoundError(
                f"지도 이미지 읽기 실패: {map_image_path}"
            )


        height, width = image.shape[:2]


        system_data["map"]["connected"] = True
        system_data["map"]["resolution"] = resolution
        system_data["map"]["origin"] = origin
        system_data["map"]["width"] = width
        system_data["map"]["height"] = height


        print("[MAP] 지도 연결 성공")

        print(
            f"[MAP] image: {map_image_path}, "
            f"size: {width} x {height}"
        )


        add_event(
            "실제 지도를 정상적으로 불러왔습니다."
        )


    except Exception as e:

        print(
            "[MAP] 지도 로딩 실패:",
            e
        )

        add_event(
            f"지도 로딩 오류: {e}"
        )


load_map()


# ============================================================
# ROS2 Bridge Node
# ============================================================

class WebMonitoringNode(Node):

    def __init__(self):

        super().__init__(
            "rescue_web_monitor"
        )


        self.bridge = CvBridge()

        # Keep one pre-encoded latest image.  The old stream generator copied
        # and re-encoded the same frame ~33 times per second per browser.
        self.latest_jpeg = None
        self._frame_sequence = 0
        self.frame_condition = threading.Condition()
        self._last_raw_web_frame_at = 0.0
        self._raw_web_interval_sec = 1.0 / 10.0

        # Counts are scoped to one /beacon_search/yolo_ready session.
        # They intentionally do not use robot position as an identity key:
        # movement was the cause of duplicate PERSON entries.
        self._yolo_active = False
        self._yolo_count_samples = deque(maxlen=YOLO_COUNT_WINDOW)
        self._confirmed_rescue_count = 0


        # 이동 경로 저장용
        self.last_path_x = None
        self.last_path_y = None


       


        qos_profile = QoSProfile(

            reliability=
                QoSReliabilityPolicy.BEST_EFFORT,

            history=
                QoSHistoryPolicy.KEEP_LAST,

            depth=1,

        )


        # Packet state is latched by packet_lock_node.  Transient-local
        # reception makes the web page immediately show the latest state
        # even when it starts after the acoustic packet decoder.
        packet_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ====================================================
        # 1. 위치 및 배터리 구독
        # ====================================================

        self.create_subscription(

            PoseWithCovarianceStamped,

            "/amcl_pose",

            self.pose_callback,

            qos_profile,

        )


        self.create_subscription(

            BatteryState,

            "/battery_state",

            self.battery_callback,

            10,

        )

        self.create_subscription(

            Float32,

            "/gas_value",

            self.gas_callback,

            10,

        )

        # ====================================================
        # 2. 원본 카메라 영상
        #
        # ★ 기존 코드 유지
        # YOLO 미작동 시 원본 영상 사용
        # ====================================================

        self.create_subscription(

            Image,

            "/camera/camera/color/image_raw",

            self.camera_callback,

            qos_profile,

        )


        # ====================================================
        # 3. YOLO
        #
        # ★ 기존 토픽 그대로 유지
        # ====================================================

        self.create_subscription(

            Int32,

            "/detected_person_count",

            self.yolo_count_callback,

            10,

        )


        # Map-evidence search publishes this as a transient state.  It
        # defines the start and end of one rescue-target counting session.
        self.create_subscription(
            Bool,
            "/beacon_search/yolo_ready",
            self.yolo_ready_callback,
            packet_qos,
        )


        self.create_subscription(

            String,

            "/person_alert_signal",

            self.yolo_alert_callback,

            10,

        )


        self.create_subscription(

            CompressedImage,

            "/yolo/processed_image/compressed",

            self.yolo_compressed_callback,

            qos_profile,

        )


        # ====================================================
        # 4. Validated XYXY packet bridge
        # ====================================================

        self.create_subscription(
            String,
            "/beacon/packet_status",
            self.packet_status_callback,
            packet_qos,
        )

        self.create_subscription(
            Bool,
            "/beacon/packet_locked",
            self.packet_lock_callback,
            packet_qos,
        )

        self.create_subscription(
            String,
            "/beacon/packet_metric",
            self.packet_metric_callback,
            10,
        )

        # ====================================================
        # 5. 이벤트 토픽
        # ====================================================

        self.create_subscription(

            String,

            "/detected_event",

            self.event_callback,

            10,

        )


        self.get_logger().info(
            "Rescue Web Monitor ROS2 Node Started."
        )


        add_event(
            "ROS2 모니터링 노드가 시작되었습니다."
        )


    # ========================================================
    # TurtleBot 위치
    # ========================================================

    def pose_callback(self, msg):

        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation


        x = float(position.x)
        y = float(position.y)


        siny_cosp = 2.0 * (
            orientation.w * orientation.z
            +
            orientation.x * orientation.y
        )


        cosy_cosp = 1.0 - 2.0 * (
            orientation.y * orientation.y
            +
            orientation.z * orientation.z
        )


        yaw = math.atan2(
            siny_cosp,
            cosy_cosp
        )


        yaw_deg = math.degrees(
            yaw
        )


        first_conn = (
            not system_data["robot"]["connected"]
        )


        system_data["robot"]["connected"] = True

        system_data["robot"]["x"] = round(
            x,
            3
        )

        system_data["robot"]["y"] = round(
            y,
            3
        )

        system_data["robot"]["yaw_deg"] = round(
            yaw_deg,
            1
        )


        system_data["robot_status"]["x"] = round(
            x,
            3
        )

        system_data["robot_status"]["y"] = round(
            y,
            3
        )

        system_data["robot_status"]["yaw"] = round(
            yaw,
            3
        )


        if first_conn:

            add_event(
                "TurtleBot 위치(AMCL) 수신 시작"
            )


        # ====================================================
        # TurtleBot 이동 경로 저장
        #
        # 5cm 이상 움직였을 때 새로운 위치 저장
        # ====================================================

        if (

            self.last_path_x is None

            or

            math.hypot(

                x - self.last_path_x,

                y - self.last_path_y

            ) >= 0.05

        ):

            system_data["path"].append({

                "x": round(x, 3),

                "y": round(y, 3),

            })


            self.last_path_x = x
            self.last_path_y = y


            # 너무 많은 좌표가 쌓이는 것을 방지
            if len(system_data["path"]) > 2000:

                system_data["path"] = (
                    system_data["path"][-2000:]
                )


    # ========================================================
    # 배터리
    # ========================================================

    def battery_callback(self, msg):

        pct = max(

            0.0,

            min(
                100.0,
                float(msg.percentage)
            )

        )


        system_data["robot"]["battery"] = round(
            pct,
            1
        )


        system_data["robot_status"]["battery"] = round(
            pct,
            1
        )


    # ========================================================
    # 가스 센서
    # ========================================================

    def gas_callback(self, msg):

        gas_value = float(msg.data)

        system_data["gas"]["connected"] = True
        system_data["gas"]["value"] = round(
            gas_value,
            2
        )
        system_data["gas"]["unit"] = "ppm"
        system_data["gas"]["risk"] = "측정 중"



    # ========================================================
    # 원본 카메라
    #
    # ★ 기존 연결 방식 유지
    # ========================================================

    def _cache_jpeg_frame(self, jpeg: bytes) -> None:
        """Store one encoded frame and wake waiting HTTP clients."""

        if not jpeg:
            return
        with self.frame_condition:
            self.latest_jpeg = jpeg
            self._frame_sequence += 1
            self.frame_condition.notify_all()

    def camera_callback(self, msg):

        # The raw stream is only the fallback before Jetson debug images
        # arrive.  It is encoded once at a bounded rate, not in every HTTP
        # client loop.
        if not system_data["yolo"]["connected"]:
            now = time.monotonic()
            if now - self._last_raw_web_frame_at < self._raw_web_interval_sec:
                return
            self._last_raw_web_frame_at = now

            try:
                frame = self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="bgr8"
                )
                success, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 55]
                )
                if not success:
                    return
                self._cache_jpeg_frame(encoded.tobytes())
                system_data["camera"]["connected"] = True

            except Exception as e:
                self.get_logger().error(
                    f"Camera frame encode error: {e}"
                )


    # ========================================================
    # YOLO 사람 감지
    #
    # ★ 여기만 중복 판정 알고리즘 수정
    # ========================================================

    def yolo_ready_callback(self, msg: Bool):
        """Start a fresh final-count session only when map search enables YOLO."""
        ready = bool(msg.data)
        system_data["yolo"]["active"] = ready
        if ready == self._yolo_active:
            return

        self._yolo_active = ready
        self._yolo_count_samples.clear()
        system_data["yolo"]["count_stable"] = False

        if ready:
            # A new selected-room scan starts a new rescue count.  The
            # previous session remains visible until this point.
            self._confirmed_rescue_count = 0
            system_data["yolo"]["rescue_target_count"] = 0
            system_data["victims"] = []
            add_event("YOLO \uad6c\uc870 \ub300\uc0c1 \ud655\uc815 \uacc4\uc218 \uc2dc\uc791")
            self.get_logger().info(
                "YOLO rescue-target count session started; previous victims cleared"
            )
            return

        add_event(
            f"YOLO \uad6c\uc870 \ub300\uc0c1 \ud655\uc815 \uacc4\uc218 \uc885\ub8cc: {self._confirmed_rescue_count}\uba85"
        )
        self.get_logger().info(
            f"YOLO rescue-target count session stopped: "
            f"confirmed={self._confirmed_rescue_count}"
        )

    def yolo_count_callback(self, msg):
        """Store raw YOLO count and promote only stable session evidence."""
        count = max(0, int(msg.data))
        yolo = system_data["yolo"]
        yolo["connected"] = True
        yolo["count"] = count
        yolo["detected"] = count > 0

        # Detection frames outside the selected-room YOLO phase must not
        # create rescue targets in the web page.
        if not self._yolo_active:
            return

        if count > 0:
            system_data["robot"]["state"] = "PERSON_FOUND"
            system_data["robot_status"]["status"] = "\ub300\uc0c1 \uac10\uc9c0"
        else:
            system_data["robot"]["state"] = "SEARCHING"
            system_data["robot_status"]["status"] = "\ud0d0\uc0c9 \uc911"

        self._yolo_count_samples.append(count)
        if len(self._yolo_count_samples) < YOLO_COUNT_MIN_SAMPLES:
            return

        stable_count, agreement = Counter(
            self._yolo_count_samples
        ).most_common(1)[0]
        yolo["count_stable"] = agreement >= YOLO_COUNT_MIN_AGREEMENT
        if not yolo["count_stable"]:
            return

        # Keep the highest stable simultaneous count observed in this one
        # session. Brief occlusion must not make a confirmed target disappear,
        # and repeated frames can never append duplicates.
        if stable_count <= self._confirmed_rescue_count:
            return

        self._confirmed_rescue_count = stable_count
        yolo["rescue_target_count"] = stable_count
        has_pose = bool(system_data["robot"]["connected"])
        current_x = round(float(system_data["robot"]["x"]), 3) if has_pose else None
        current_y = round(float(system_data["robot"]["y"]), 3) if has_pose else None
        gas_value = system_data["gas"]["value"] if system_data["gas"]["connected"] else None
        now = time.strftime("%H:%M:%S")

        # Replace instead of append: this list is the final confirmed count
        # for the active YOLO session, not a per-frame detection history.
        system_data["victims"] = [
            {
                "id": f"PERSON_{index:02d}",
                "time": now,
                "x": current_x,
                "y": current_y,
                "gas": gas_value,
                "status": "\uad6c\uc870 \ud544\uc694",
            }
            for index in range(1, stable_count + 1)
        ]
        add_event(
            f"YOLO \ud655\uc815 \uad6c\uc870 \ub300\uc0c1: {stable_count}\uba85 "
            f"({agreement}/{len(self._yolo_count_samples)} \ud504\ub808\uc784 \uc77c\uce58)"
        )
        self.get_logger().info(
            f"[CONFIRMED RESCUE TARGETS] count={stable_count} "
            f"agreement={agreement}/{len(self._yolo_count_samples)}"
        )


    # ========================================================
    # YOLO Alert
    # ========================================================

    def yolo_alert_callback(self, msg):

        system_data["yolo"]["alert"] = (
            msg.data
        )

    # ========================================================
    # YOLO 처리 영상
    #
    # ★ 중요
    #
    # 현재 카메라 + YOLO 영상이 잘 작동하는 부분이라
    # 기존 코드 그대로 유지
    # ========================================================

    def yolo_compressed_callback(
        self,
        msg: CompressedImage
    ):

        try:

            # The Jetson node already emits JPEG.  Do not decode/re-encode
            # it on the NUC before serving the exact same bytes by HTTP.
            is_jpeg = (
                "jpeg" in str(msg.format).lower()
                or bytes(msg.data[:2]) == b"\xff\xd8"
            )
            if is_jpeg:
                self._cache_jpeg_frame(bytes(msg.data))
            else:
                # Compatibility fallback for a legacy non-JPEG publisher.
                np_arr = np.frombuffer(msg.data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("invalid compressed YOLO image")
                success, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 55]
                )
                if not success:
                    return
                self._cache_jpeg_frame(encoded.tobytes())


            system_data["yolo"]["connected"] = True

            system_data["camera"]["connected"] = True


        except Exception as e:

            self.get_logger().error(
                f"YOLO compressed decode error: {e}"
            )


    # ========================================================
    # Validated XYXY packet callbacks
    # ========================================================

    def packet_status_callback(self, msg: String):

        status = str(msg.data).strip().upper() or "STARTING"
        beacon = system_data["beacon"]
        packet = beacon["packet"]
        previous = packet["status"]

        packet["status"] = status
        beacon["connected"] = True

        if status == "LOCKED":
            packet["locked"] = True
            beacon["detected"] = True
            beacon["tracking"] = True
            beacon["id"] = "BEACON_1"
        elif status in {"SEARCH", "STARTING", "ERROR"}:
            packet["locked"] = False
            beacon["detected"] = False
            beacon["tracking"] = False
            if status != "LOCKED":
                beacon["id"] = None

        if status != previous:
            add_event(f"비콘 패킷 상태: {status}")

    def packet_lock_callback(self, msg: Bool):

        locked = bool(msg.data)
        beacon = system_data["beacon"]
        packet = beacon["packet"]
        previous = bool(packet["locked"])

        packet["locked"] = locked
        beacon["connected"] = True
        beacon["detected"] = locked
        beacon["tracking"] = locked
        beacon["id"] = "BEACON_1" if locked else None

        if locked:
            packet["status"] = "LOCKED"
        elif packet["status"] == "LOCKED":
            packet["status"] = "SEARCH"

        if locked != previous:
            add_event(
                "비콘 XYXY 패킷 LOCKED"
                if locked
                else "비콘 XYXY 패킷 UNLOCKED"
            )

    def packet_metric_callback(self, msg: String):

        try:
            metric = json.loads(msg.data)
            level_dbfs = float(metric["level_dbfs"])
            quality_db = float(metric["quality_db"])
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            self.get_logger().warning("Malformed /beacon/packet_metric ignored")
            return

        if not math.isfinite(level_dbfs) or not math.isfinite(quality_db):
            return

        # Direction fields are optional so the web monitor remains compatible
        # while an older packet_lock_node is still running during deployment.
        direction_level_dbfs = None
        direction_quality_db = None
        try:
            candidate_level = float(metric["direction_level_dbfs"])
            candidate_quality = float(metric["direction_quality_db"])
            if math.isfinite(candidate_level) and math.isfinite(candidate_quality):
                direction_level_dbfs = candidate_level
                direction_quality_db = candidate_quality
        except (TypeError, ValueError, KeyError):
            pass

        packet = system_data["beacon"]["packet"]
        packet["level_dbfs"] = round(level_dbfs, 2)
        packet["quality_db"] = round(quality_db, 2)
        packet["direction_level_dbfs"] = (
            round(direction_level_dbfs, 2)
            if direction_level_dbfs is not None
            else None
        )
        packet["direction_quality_db"] = (
            round(direction_quality_db, 2)
            if direction_quality_db is not None
            else None
        )
        packet["raw_dbfs"] = metric.get("raw_dbfs", {})
        packet["scores_db"] = metric.get("scores_db", {})
        packet["last_valid_at"] = time.strftime("%H:%M:%S")
        system_data["beacon"]["connected"] = True

    # ========================================================
    # 기타 이벤트
    # ========================================================

    def event_callback(
        self,
        msg: String
    ):

        add_event(
            msg.data
        )


# ============================================================
# ROS2 Thread
# ============================================================

ros_node: Optional[
    WebMonitoringNode
] = None


def ros_spin():

    global ros_node


    rclpy.init()


    ros_node = (
        WebMonitoringNode()
    )


    try:

        rclpy.spin(
            ros_node
        )


    except Exception as e:

        print(
            "[ROS] spin error:",
            e
        )


    finally:

        if ros_node is not None:

            ros_node.destroy_node()


        if rclpy.ok():

            rclpy.shutdown()


ros_thread = threading.Thread(

    target=ros_spin,

    daemon=True

)


ros_thread.start()


# ============================================================
# FastAPI Server
# ============================================================

app = FastAPI(
    title="TurtleBot Rescue Monitoring System"
)


if FRONTEND_DIR.exists():

    app.mount(

        "/static",

        StaticFiles(
            directory=str(
                FRONTEND_DIR
            )
        ),

        name="static"

    )


# ============================================================
# 메인 페이지
# ============================================================

@app.get("/")
async def root():

    index_file = (
        FRONTEND_DIR
        /
        "index.html"
    )


    if not index_file.exists():

        return JSONResponse(

            {
                "error":
                    f"index.html not found: {index_file}"
            },

            status_code=404

        )


    return FileResponse(
        str(index_file)
    )


# ============================================================
# 시스템 상태 API
# ============================================================

@app.get("/status")
@app.get("/api/system-status")
async def status():

    return system_data


# ============================================================
# 실제 ROS 지도 이미지
# ============================================================

@app.get("/map_image")
async def map_image():

    if (
        map_image_path is None
        or
        not map_image_path.exists()
    ):

        return JSONResponse(

            {
                "error":
                    "Map file not found"
            },

            status_code=404

        )


    image = cv2.imread(

        str(map_image_path),

        cv2.IMREAD_GRAYSCALE

    )


    if image is None:

        return JSONResponse(

            {
                "error":
                    "Map load failed"
            },

            status_code=500

        )


    success, encoded = (
        cv2.imencode(
            ".png",
            image
        )
    )


    if not success:

        return JSONResponse(

            {
                "error":
                    "Map encode failed"
            },

            status_code=500

        )


    return StreamingResponse(

        iter([
            encoded.tobytes()
        ]),

        media_type="image/png"

    )


# ============================================================
# Camera Stream
#
# ★ 기존 방식 유지
# ============================================================

def camera_stream():

    last_sequence = -1
    while True:

        node = ros_node
        if node is None:
            time.sleep(0.1)
            continue

        # Wait for a genuinely new frame. This prevents the HTTP generator
        # from re-encoding one stale image in a tight loop while YOLO runs.
        with node.frame_condition:
            has_new_frame = node.frame_condition.wait_for(
                lambda: (
                    node.latest_jpeg is not None
                    and node._frame_sequence != last_sequence
                ),
                timeout=1.0,
            )
            if not has_new_frame:
                continue
            jpeg = node.latest_jpeg
            last_sequence = node._frame_sequence

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg
            + b"\r\n"
        )


# ============================================================
# Video Feed
#
# ★ 기존 방식 유지
# ============================================================

@app.get("/video_feed")
def video_feed():

    return StreamingResponse(

        camera_stream(),

        media_type=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),

    )


# ============================================================
# WebSocket
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()


    print(
        "[WEB] Browser websocket connected"
    )


    try:

        while True:

            await websocket.send_text(

                json.dumps(

                    system_data,

                    ensure_ascii=False

                )

            )


            await asyncio.sleep(
                0.2
            )


    except (
        WebSocketDisconnect,
        Exception
    ):

        print(
            "[WEB] Browser websocket disconnected"
        )
