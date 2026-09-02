# GitHub 다운로드·실행·코드 읽기 가이드

이 문서는 프로젝트를 처음 보는 사용자가 저장소를 내려받고, 실행 환경을 준비하고, 각 Python 파일이 어디에 사용되는지 확인하기 위한 안내서입니다.

## 1. 먼저 알아야 할 실행 구조

프로그램은 한 파일을 실행하는 구조가 아니라 다음 세 실행 묶음으로 구성됩니다.

| 실행 장치 | 담당 기능 | 주 실행 파일 |
|---|---|---|
| NUC | TurtleBot, Nav2, UWB, XYXY, DOA, 지도 탐색, RealSense, 로그 | `map_evidence_search.launch.py` |
| Jetson Orin | YOLO 사람 추론 | `jetson_yolo_inference.launch.py` |
| NUC 웹 서버 | 지도·센서·카메라 웹 모니터링 | `web/backend/server.py` |

중앙 상태머신인 `map_evidence_search_node.py`만 비콘 탐색 목적지를 Nav2에 전달합니다. Jetson 코드는 Nav2와 모터를 제어하지 않습니다.

## 2. GitHub에서 다운로드하기

### Git 사용

NUC 최종 설정은 `/home/human/resonator_filter` 경로를 기준으로 하므로 실제 장비에서는 해당 경로에 clone하는 방법이 가장 간단합니다.

```bash
git clone https://github.com/<OWNER>/<REPOSITORY>.git \
  /home/human/resonator_filter

cd /home/human/resonator_filter
```

이미 clone한 저장소를 갱신할 때만 다음 명령을 사용합니다.

```bash
cd /home/human/resonator_filter
git pull
```

### ZIP 사용

GitHub의 `Code → Download ZIP`을 선택한 뒤 압축을 해제합니다. 실제 NUC에서 전체 구동할 경우 최종 폴더가 다음 경로가 되도록 배치합니다.

```text
/home/human/resonator_filter
```

다른 위치에 설치하면 다음 두 경로 의존성을 수정해야 합니다.

- `ros2_ws/src/beacon_tracker/config/waffle.yaml`의 `detector_script`
- `beacon1_lock_final.py`의 `FILTER_FILE`

## 3. 처음 읽을 파일

다음 순서로 보면 전체 구조를 빠르게 이해할 수 있습니다.

1. [`README.md`](../README.md): 설치와 전체 터미널 실행 순서
2. [`FINAL_SYSTEM_OVERVIEW.md`](FINAL_SYSTEM_OVERVIEW.md): 모든 알고리즘의 연결 관계
3. [`map_evidence_search_node.py`](../ros2_ws/src/beacon_tracker/beacon_tracker/map_evidence_search_node.py): 전체 상태머신
4. [`map_evidence.py`](../ros2_ws/src/beacon_tracker/beacon_tracker/map_evidence.py): 지도 구역화와 확률 계산
5. [`beacon1_lock_final.py`](../beacon1_lock_final.py): XYXY 패킷 검증
6. [`doa.py`](../doa_angle/doa.py): MUSIC·SRP-PHAT·TDOA
7. [`jetson_yolo_inference_node.py`](../ros2_ws/src/beacon_tracker/beacon_tracker/jetson_yolo_inference_node.py): Jetson 사람 검출

## 4. 소스만 확인하는 스모크 테스트

하드웨어가 없어도 Python 문법과 UWB parser를 확인할 수 있습니다.

```bash
cd /home/human/resonator_filter

python3 - <<'PY'
import ast
from pathlib import Path

files = list(Path('.').rglob('*.py'))
for path in files:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
print(f'Python syntax OK: {len(files)} files')
PY

python3 uwb_reader.py --self-test
```

ROS 패키지 빌드까지 확인하려면 다음을 실행합니다.

```bash
source /opt/ros/humble/setup.bash
source /home/human/turtlebot3_ws/install/setup.bash

cd /home/human/resonator_filter/ros2_ws
colcon build --symlink-install --packages-select beacon_tracker
source install/setup.bash

ros2 pkg executables beacon_tracker
```

## 5. NUC 설치와 터미널 실행 순서

### 5.1 의존성 설치

```bash
sudo apt update
sudo apt install -y \
  python3-serial python3-numpy python3-scipy python3-opencv python3-yaml \
  python3-pip \
  ros-humble-cv-bridge ros-humble-navigation2 ros-humble-nav2-bringup

python3 -m pip install --user fastapi uvicorn
```

NUC의 ROS 2 Humble 시스템 Python과 충돌할 수 있으므로 `numpy`, `setuptools`, `opencv-python`을 전역 pip로 강제 업그레이드하지 않습니다.

### 5.2 빌드

```bash
source /opt/ros/humble/setup.bash
source /home/human/turtlebot3_ws/install/setup.bash

cd /home/human/resonator_filter/ros2_ws
colcon build --symlink-install --packages-select beacon_tracker
source install/setup.bash
```

### 5.3 모든 NUC ROS 터미널의 공통 환경

각각의 새 터미널을 열 때 다음 환경을 먼저 적용합니다.

```bash
source /opt/ros/humble/setup.bash
source /home/human/turtlebot3_ws/install/setup.bash
source /home/human/resonator_filter/ros2_ws/install/setup.bash

export TURTLEBOT3_MODEL=waffle
export ROS_DOMAIN_ID=30
unset ROS_LOCALHOST_ONLY
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

### 5.4 터미널 1: TurtleBot bringup

```bash
ros2 launch turtlebot3_bringup robot.launch.py
```

확인:

```bash
ros2 topic echo /odom --once
ros2 topic hz /scan
```

### 5.5 터미널 2: RealSense

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  align_depth.enable:=true
```

확인:

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
```

### 5.6 터미널 3: Nav2와 RViz

```bash
ros2 launch turtlebot3_navigation2 navigation2.launch.py \
  map:=/home/human/resonator_filter/maps/beacon.yaml \
  params_file:=/home/human/resonator_filter/nav2_config/waffle_resonator_humble.yaml \
  use_sim_time:=False
```

RViz에서 `2D Pose Estimate`를 지정한 뒤 확인합니다.

```bash
ros2 action list | grep navigate_to_pose
ros2 lifecycle get /bt_navigator
```

### 5.7 터미널 4: 비콘 탐색 통합 launch

```bash
ros2 launch beacon_tracker map_evidence_search.launch.py \
  search_enabled:=true \
  launch_yolo:=true
```

이 명령은 다음 NUC 노드를 한 번에 시작합니다.

- `uwb_range_node`
- `packet_lock_node`
- `doa_angle_node`
- `map_evidence_search_node`
- `evidence_diagnostics_logger_node`
- `camera_jpeg_relay_node`
- `person_localizer_node`

중복 실행 여부를 확인합니다.

```bash
ros2 node list | sort
ros2 node list | grep -E 'uwb|packet|doa|map_evidence|camera_jpeg|person_localizer'
```

### 5.8 터미널 5: 센서 모니터

```bash
ros2 run beacon_tracker beacon_monitor_node
```

### 5.9 터미널 6: 구역 확률 모니터

```bash
ros2 run beacon_tracker room_probability_monitor_node
```

### 5.10 터미널 7: 웹 서버

```bash
cd /home/human/resonator_filter/web/backend
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

접속 주소:

```text
http://<NUC_IP>:8000
```

### 5.11 터미널 8: 가스 센서

```bash
python3 /home/human/resonator_filter/gas_node.py
```

## 6. Jetson Orin 준비와 실행

Jetson에도 최소한 `ros2_ws/src/beacon_tracker` 패키지가 있어야 합니다. Jetson 사용자 계정은 NUC의 `human`과 다를 수 있으므로 Jetson에서는 홈 디렉터리 기준 경로를 사용합니다.

```bash
git clone https://github.com/<OWNER>/<REPOSITORY>.git \
  ~/resonator_filter

python3 -m pip install ultralytics==8.4.135

source /opt/ros/humble/setup.bash
cd ~/resonator_filter/ros2_ws
colcon build --symlink-install --packages-select beacon_tracker
source install/setup.bash

export TURTLEBOT3_MODEL=waffle
export ROS_DOMAIN_ID=30
unset ROS_LOCALHOST_ONLY
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 launch beacon_tracker jetson_yolo_inference.launch.py
```

`yolov8n.pt`가 없고 인터넷이 연결돼 있으면 최초 실행 시 자동 다운로드됩니다. 오프라인 환경에서는 모델을 미리 준비합니다.

```bash
python3 - <<'PY'
from ultralytics import YOLO
YOLO('yolov8n.pt')
print('YOLO model ready')
PY
```

Jetson 연결 확인:

```bash
ros2 topic hz /beacon_search/jetson/color/compressed
ros2 topic echo /beacon_search/jetson/yolo_detection
```

## 7. Python 파일별 역할과 중요 코드

아래 파일명은 GitHub에서 클릭하면 실제 소스로 이동합니다. `직접 실행하지 않음`으로 표시된 파일은 다른 노드가 import하는 라이브러리입니다.

### 센서 원본 처리

#### [`beacon1_lock_final.py`](../beacon1_lock_final.py)

- 사용 위치: NUC의 `packet_lock_node.py`가 자식 프로세스로 실행합니다.
- 역할: SIPEED 공진관 마이크로 `X=(500+900Hz)`, `Y=(650+1050Hz)`의 XYXY 순서와 음량을 검증합니다.
- 중요 코드: `CHANNEL_MAP`은 주파수별 공진관 채널, `classify_slot()`은 PASS/WRONG/ERASE, `finish_candidate()`는 패킷 승인, `register_valid_packet()`은 LOCK 상태, `emit_valid_packet_metric()`은 지도용 음량 JSON을 담당합니다.
- 주의: `@@BEACON_PACKET_METRIC@@` 등 표준 출력 marker는 ROS bridge가 parsing하므로 한쪽만 바꾸면 안 됩니다.
- 단독 확인:

```bash
cd /home/human/resonator_filter
python3 -u beacon1_lock_final.py
```

#### [`uwb_reader.py`](../uwb_reader.py)

- 사용 위치: `uwb_range_node.py`가 import합니다.
- 역할: BU03에 `AT+DISTANCE`를 보내 거리 parsing, timeout, 통계와 FRESH/HOLD 판정을 수행합니다.
- 중요 코드: `parse_distance()`, `DistanceFreshnessTracker`, `BU03RangeReader.request_distance()`, `summarize()`를 봅니다.
- 하드웨어 없는 parser 검사:

```bash
python3 /home/human/resonator_filter/uwb_reader.py --self-test
```

- 실제 거리 20회 검사:

```bash
python3 /home/human/resonator_filter/uwb_reader.py \
  --port /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \
  --count 20 --interval 0.2
```

#### [`gas_node.py`](../gas_node.py)

- 사용 위치: NUC의 Arduino 가스 센서용 독립 ROS 노드입니다.
- 역할: `$GAS,...` 시리얼 프레임을 `/gas_value`, `/gas_risk`로 발행합니다.
- 중요 코드: `GasSensorNode.__init__()`의 by-id 포트와 `read_gas()`의 frame parsing을 확인합니다.
- 실행:

```bash
python3 /home/human/resonator_filter/gas_node.py
```

### ReSpeaker DOA

#### [`doa_angle/audio.py`](../doa_angle/audio.py)

- 사용 위치: `doa.py`의 `DOAEstimator`가 import합니다.
- 역할: ReSpeaker의 16kHz·6채널 S16_LE 오디오를 `arecord`로 읽어 NumPy 배열로 변환합니다.
- 중요 코드: `DOAAudioReader.start()`, `read()`, `stop()`과 실제 raw mic 채널 1~4 선택을 확인합니다.
- 직접 실행하지 않으며 통합 launch의 `doa_angle_node`를 통해 사용합니다.

#### [`doa_angle/config.py`](../doa_angle/config.py)

- 사용 위치: `doa.py`의 모든 신호처리 임계값을 제공합니다.
- 역할: 마이크 좌표, 활성 tone, MUSIC·SRP-PHAT·TDOA 가중치, SNR과 군집 기준을 관리합니다.
- 중요 코드: `DOA_ACTIVE_TONES_HZ`, `DOA_FUSION_*`, `DOA_MEASUREMENTS`, `DOA_SIGN`을 확인합니다.
- 채널 시간 보정은 `doa_calibration.json`, 로봇 장착 yaw는 `waffle.yaml`에서 별도로 관리합니다.

#### [`doa_angle/doa.py`](../doa_angle/doa.py)

- 사용 위치: NUC의 `doa_angle_node.py`가 `DOAEstimator`를 import합니다.
- 역할: 활성 900/1050Hz를 선택하고 MUSIC 후보를 SRP-PHAT와 6쌍 TDOA로 검증한 뒤 두 frame 각도를 군집화합니다.
- 중요 코드: `select_active_doa_tone_window()`, `music_tone_spectrum()`, `extract_circular_music_peaks()`, `srp_phat_doa()`, `score_tdoa_candidate()`, `DOAEstimator.estimate()`를 순서대로 봅니다.
- 결과 확인:

```bash
ros2 topic echo /beacon/doa_metric
ros2 topic echo /beacon/doa_stable
```

### ROS 센서 bridge와 모니터

#### [`uwb_range_node.py`](../ros2_ws/src/beacon_tracker/beacon_tracker/uwb_range_node.py)

- 사용 위치: NUC 통합 launch가 실행합니다.
- 역할: `uwb_reader.py`를 ROS로 연결하고 `/uwb/range_m`, `/uwb/status`를 발행합니다.
- 중요 코드: `UWBRangeNode.__init__()`의 parameter/topic, `_poll()`의 거리 요청·상태 발행·재연결 처리를 확인합니다.
- 단독 실행:

```bash
ros2 run beacon_tracker uwb_range_node --ros-args \
  --params-file /home/human/resonator_filter/ros2_ws/src/beacon_tracker/config/waffle.yaml
```

#### [`packet_lock_node.py`](../ros2_ws/src/beacon_tracker/beacon_tracker/packet_lock_node.py)

- 사용 위치: NUC 통합 launch가 실행합니다.
- 역할: `beacon1_lock_final.py`의 stdout을 LOCK/status/metric/event ROS 토픽으로 변환합니다.
- 중요 코드: `lock_event_from_line()`, `packet_metric_from_line()`, `packet_result_from_line()`과 자식 프로세스 종료 처리를 확인합니다.
- 단독 실행:

```bash
ros2 run beacon_tracker packet_lock_node --ros-args \
  --params-file /home/human/resonator_filter/ros2_ws/src/beacon_tracker/config/waffle.yaml
```

#### [`doa_angle_node.py`](../ros2_ws/src/beacon_tracker/beacon_tracker/doa_angle_node.py)

- 사용 위치: NUC 통합 launch가 실행합니다.
- 역할: XYXY LOCK, UWB FRESH≤6m, 명시적 enable을 확인하고 별도 worker에서 DOA를 계산합니다.
- 중요 코드: `_load_doa_estimator()`, packet/UWB/enable callback, worker 시작·완료와 stale angle 폐기를 확인합니다.
- 실행은 관련 입력을 함께 만드는 통합 launch를 권장합니다.

#### [`beacon_monitor_node.py`](../ros2_ws/src/beacon_tracker/beacon_tracker/beacon_monitor_node.py)

- 사용 위치: 현장 운영자가 별도 NUC 터미널에서 실행합니다.
- 역할: UWB·XYXY·DOA 최신값과 stale 시간을 한 줄로 표시하며 제어 명령은 만들지 않습니다.
- 실행:

```bash
ros2 run beacon_tracker beacon_monitor_node
```

#### [`room_probability_monitor_node.py`](../ros2_ws/src/beacon_tracker/beacon_tracker/room_probability_monitor_node.py)

- 사용 위치: 현장 운영자가 별도 NUC 터미널에서 실행합니다.
- 역할: 자동 구역별 상대 비콘 확률, UWB fit, 거리와 XYXY 음량을 표로 표시합니다.
- 중요 코드: room probability JSON callback과 `_number_or_dash()` 출력 formatting을 확인합니다.
- 실행:

```bash
ros2 run beacon_tracker room_probability_monitor_node
```

#### [`evidence_diagnostics_logger_node.py`](../ros2_ws/src/beacon_tracker/beacon_tracker/evidence_diagnostics_logger_node.py)

- 사용 위치: NUC 통합 launch가 기본으로 실행합니다.
- 역할: 시작부터 Ctrl+C까지 상태, 센서, Nav2 goal/path와 내부 변수를 같은 timestamp로 기록합니다.
- 중요 코드: subscription 구성, snapshot callback, event/path writer와 종료 시 flush/manifest 처리를 확인합니다.
- 별도 실행은 일반적으로 필요 없으며 통합 launch에서 `launch_diagnostics:=true`를 사용합니다.

### 지도 탐색과 확률

#### [`map_evidence.py`](../ros2_ws/src/beacon_tracker/beacon_tracker/map_evidence.py)

- 사용 위치: `map_evidence_search_node.py`가 import하는 ROS 비의존 계산 라이브러리입니다.
- 역할: 지도 구역·portal·coverage 생성, UWB/XYXY/DOA 점수와 방 선택을 계산합니다.
- 중요 코드: `build_topology()`, `closed_space_first_coverage_waypoints()`, `estimate_zone_probabilities()`, `decide_room_entry()`, `assess_doa_open_space()`를 봅니다.
- 직접 실행하지 않습니다. import 확인:

```bash
python3 -c 'from beacon_tracker import map_evidence; print("map_evidence import OK")'
```

#### [`map_evidence_search_node.py`](../ros2_ws/src/beacon_tracker/beacon_tracker/map_evidence_search_node.py)

- 사용 위치: NUC 통합 launch의 중앙 상태머신입니다.
- 역할: 지도, costmap, TF, UWB, XYXY, DOA와 사람 pose를 받아 Nav2 goal, 센서 enable과 임무 완료를 결정합니다.
- 중요 코드: `_declare_parameters()`에서 전체 조건, map/costmap callback, evidence 수집·평가, goal preflight/send/result, UWB 장애 복구, DOA probe, YOLO gate와 victim approach 부분을 확인합니다.
- 이 파일이 비콘 탐색 Nav2 goal을 소유하므로 같은 노드를 중복 실행하면 안 됩니다.
- 실행:

```bash
ros2 launch beacon_tracker map_evidence_search.launch.py \
  search_enabled:=true launch_yolo:=true
```

#### [`survey_policy.py`](../ros2_ws/src/beacon_tracker/beacon_tracker/survey_policy.py)

- 사용 위치: 중앙 상태머신의 초기 evidence 수집 종료 조건입니다.
- 역할: 단발성 UWB spike가 아니라 최근 거리 window의 시간·sample·sigma·증가량과 공동 evidence 수로 방 선택 시작 여부를 판단합니다.
- 중요 코드: `assess_survey_departure()`, `assess_survey_gate()`를 확인합니다.
- 직접 실행하지 않습니다.

#### [`recovery_policy.py`](../ros2_ws/src/beacon_tracker/beacon_tracker/recovery_policy.py)

- 사용 위치: 중앙 상태머신의 UWB HOLD/LOST 복구 판정입니다.
- 역할: 연속 FRESH 복구, 정지 복구의 물리적 일관성과 반복 XYXY peak를 판정합니다.
- 중요 코드: `assess_continuous_fresh_recovery()`, `assess_stationary_uwb_recovery()`, `assess_repeated_xyxy_peak()`를 확인합니다.
- 직접 실행하지 않습니다.

#### [`uwb_trilateration.py`](../ros2_ws/src/beacon_tracker/beacon_tracker/uwb_trilateration.py)

- 사용 위치: 지도 상태머신의 거리 집계와 실험적 anchor 위치 추정 보조 모듈입니다.
- 역할: median/MAD outlier 제거와 weighted Huber 2D 위치 추정을 수행합니다.
- 중요 코드: `aggregate_ranges()`, `estimate_anchor_position()`과 비공선 관측 검사를 확인합니다.
- 단일 UWB 추정 좌표를 Nav2 goal로 직접 사용하지 않습니다.

### 카메라와 Jetson YOLO

#### [`camera_jpeg_relay_node.py`](../ros2_ws/src/beacon_tracker/beacon_tracker/camera_jpeg_relay_node.py)

- 사용 위치: NUC 통합 launch에서 `launch_yolo=true`일 때 실행합니다.
- 역할: RealSense RGB를 JPEG quality 50, 최대 10fps로 압축하여 Jetson에 전달합니다.
- 중요 코드: image callback의 FPS 제한, `cv_bridge`, JPEG encoding과 원본 header 보존을 확인합니다.
- 토픽 확인:

```bash
ros2 topic hz /beacon_search/jetson/color/compressed
```

#### [`jetson_yolo_inference_node.py`](../ros2_ws/src/beacon_tracker/beacon_tracker/jetson_yolo_inference_node.py)

- 사용 위치: Jetson 전용 launch가 실행합니다.
- 역할: `yolo_ready`, UWB gate와 NUC JPEG를 받아 person class만 검출하고 bbox·count·HTTP 호환 영상을 발행합니다.
- 중요 코드: `_gate_open()`, `_detect_people()`, `_choose_target()`, `_publish_detection()`, `_publish_http_image()`를 확인합니다.
- 실행:

```bash
ros2 launch beacon_tracker jetson_yolo_inference.launch.py
```

#### [`person_localizer_node.py`](../ros2_ws/src/beacon_tracker/beacon_tracker/person_localizer_node.py)

- 사용 위치: NUC 통합 launch에서 `launch_yolo=true`일 때 실행합니다.
- 역할: Jetson bbox와 NUC aligned depth·CameraInfo·TF를 결합해 사람의 안정된 map pose를 발행합니다.
- 중요 코드: timestamp depth matching, bbox ROI median depth, pinhole 3D 변환, TF lookup과 최근 3회 spread 검사를 확인합니다.
- 결과 확인:

```bash
ros2 topic echo /beacon_search/person_pose
ros2 topic echo /beacon_search/person_detected
```

### Launch와 Python 패키징

#### [`map_evidence_search.launch.py`](../ros2_ws/src/beacon_tracker/launch/map_evidence_search.launch.py)

- 사용 위치: NUC의 주 launch입니다.
- 역할: UWB, XYXY, DOA, 상태머신, 로그와 선택적 카메라 relay/localizer를 한 번씩 실행합니다.
- 중요 코드: `_validate_boolean_launch_arguments()`와 `generate_launch_description()`의 노드 목록·parameter file 전달을 확인합니다.
- 주요 인자: `search_enabled`, `launch_yolo`, `launch_diagnostics`, `params_file`

#### [`jetson_yolo_inference.launch.py`](../ros2_ws/src/beacon_tracker/launch/jetson_yolo_inference.launch.py)

- 사용 위치: Jetson의 주 launch입니다.
- 역할: `waffle.yaml`을 읽어 `jetson_yolo_inference_node` 하나만 실행합니다.
- 주요 인자: `params_file`

#### [`setup.py`](../ros2_ws/src/beacon_tracker/setup.py)

- 사용 위치: `colcon build`가 ROS 실행 파일과 launch/config 설치 규칙을 읽습니다.
- 중요 코드: `data_files`와 `console_scripts`입니다. 새 ROS Python 노드를 추가하면 여기에도 entry point를 등록해야 합니다.
- 확인:

```bash
ros2 pkg executables beacon_tracker
```

#### [`beacon_tracker/__init__.py`](../ros2_ws/src/beacon_tracker/beacon_tracker/__init__.py)

- 사용 위치: Python이 폴더를 `beacon_tracker` 패키지로 인식할 때 사용합니다.
- 역할: 실행 알고리즘은 없으며 package marker입니다.
- 직접 실행하지 않습니다.

### 웹 서버

#### [`web/backend/server.py`](../web/backend/server.py)

- 사용 위치: NUC의 독립 HTTP/WebSocket 서버입니다.
- 역할: ROS의 로봇 위치, 배터리, 가스, 카메라, UWB, XYXY, YOLO와 구조 대상 수를 웹 frontend에 전달합니다.
- 중요 코드: `load_map()`, `WebMonitoringNode`, `/status`, `/map_image`, `/video_feed`, `/ws` route와 YOLO session count 처리를 확인합니다.
- 실행:

```bash
cd /home/human/resonator_filter/web/backend
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

## 8. Python 이외의 중요 파일

| 파일 | 역할 | 수정 시점 |
|---|---|---|
| [`waffle.yaml`](../ros2_ws/src/beacon_tracker/config/waffle.yaml) | 센서 경로와 탐색·DOA·YOLO 임계값 | 포트, 장착 위치, 판단 기준 변경 |
| [`waffle_resonator_humble.yaml`](../nav2_config/waffle_resonator_humble.yaml) | Waffle footprint, planner/controller/costmap | 로봇 외형 또는 Nav2 튜닝 |
| [`navigate_to_pose_no_recovery.xml`](../ros2_ws/src/beacon_tracker/config/navigate_to_pose_no_recovery.xml) | 탐색 goal 실패 시 spin/backup 반복 억제 | Nav2 behavior 변경 |
| [`fastdds_dynamic_memory.xml`](../ros2_ws/src/beacon_tracker/config/fastdds_dynamic_memory.xml) | Fast DDS payload memory 설정 | DDS history payload 오류 대응 |
| [`doa_calibration.json`](../doa_angle/doa_calibration.json) | ReSpeaker 채널 시간 지연 보정 | 마이크 장비 변경·재보정 |
| [`filter_bank_500_650_900_1050.npz`](../outputs/filter_bank_500_650_900_1050.npz) | SIPEED 네 주파수 필터 계수 | sample rate·필터 설계 변경 |
| [`beacon_sound_code.ino`](../firmware/beacon_sound_code.ino) | XYXY 음향 비콘 송신 펌웨어 | tone 또는 송신 timing 변경 |

## 9. 정상 동작 확인 명령

```bash
# 핵심 노드
ros2 node list | grep -E 'uwb|packet|doa|map_evidence|camera_jpeg|person|yolo'

# 센서
ros2 topic echo /uwb/status
ros2 topic echo /uwb/range_m
ros2 topic echo /beacon/packet_status
ros2 topic echo /beacon/packet_metric

# 상태머신과 확률
ros2 topic echo /beacon_search/state
ros2 topic echo /beacon_search/room_probabilities
ros2 action list | grep navigate_to_pose

# DOA와 YOLO
ros2 topic echo /beacon/doa_enabled
ros2 topic echo /beacon/doa_metric
ros2 topic echo /beacon_search/yolo_ready
ros2 topic echo /beacon_search/person_pose
ros2 topic echo /beacon_search/mission_complete
```

## 10. 자주 발생하는 문제

| 증상 | 우선 확인 |
|---|---|
| `Package 'beacon_tracker' not found` | workspace 빌드 후 `source ros2_ws/install/setup.bash` 여부 |
| UWB 노드 종료 | by-id 포트, `dialout` 권한, 중복 포트 접근, BU03 전원 재연결 |
| XYXY가 STARTING | SIPEED 장치명, 48kHz·8채널, `beacon1_lock_final.py` 자식 프로세스 |
| DOA가 계속 NO DATA | XYXY LOCK, UWB FRESH≤6m, `/beacon/doa_enabled`, 개방 공간·정지 gate |
| YOLO가 실행되지 않음 | UWB FRESH 2초·3m 이내, 최근 VALID XYXY, Jetson의 `yolov8n.pt`와 영상 토픽 |
| Nav2 goal이 거부됨 | AMCL 초기 위치, lifecycle active, global costmap, map/path preflight |
| 같은 장소 반복 | 최근 goal tabu, zone cooldown, UWB HOLD/LOST와 XYXY 음량 로그 |

현장 문제는 `outputs/evidence_diagnostics_*` 세션의 `timeline.csv`, `events.jsonl`, `state_snapshots.csv`를 같은 timestamp 기준으로 비교합니다.
