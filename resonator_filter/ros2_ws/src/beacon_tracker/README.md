# beacon_tracker

대회용 TurtleBot3 Waffle/NUC 통합 ROS 2 패키지입니다. 전체 설치·실행 순서는 저장소 최상위의 [`README.md`](../../../README.md)를 따릅니다.

주 실행 명령:

```bash
ros2 launch beacon_tracker map_evidence_search.launch.py \
  search_enabled:=true \
  launch_yolo:=true
```

Jetson Orin 실행 명령:

```bash
ros2 launch beacon_tracker jetson_yolo_inference.launch.py
```

운영 모니터:

```bash
ros2 run beacon_tracker beacon_monitor_node
ros2 run beacon_tracker room_probability_monitor_node
```

이 제출본은 Waffle/Humble 최종 구동 경로만 포함합니다. Burger 상태머신, 라이다 self-filter, 극좌표 실험, UWB 단독 YOLO gate 테스트는 포함하지 않습니다.
