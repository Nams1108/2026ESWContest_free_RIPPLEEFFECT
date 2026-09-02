"""
============================================================
config.py

[역할]
비콘 탐색 → DOA 정렬 → Sipeed LOCK → 추적 전진에 필요한
설정값을 한곳에서 관리합니다.

중요
------------------------------------------------------------
아래 값은 여기서 다시 정의하지 않습니다.

- 500 / 650 / 900 / 1050 Hz
- XYXY
- -36 dB threshold
- 10 dB pair margin
- LOCK / UNLOCK 조건

위 값들은 beacon1_lock_final.py의 담당 영역입니다.

이 config.py는 다음 전체 흐름을 관리합니다.

ReSpeaker XYXY 사전 검출
→ DOA 정렬
→ Sipeed beacon1_lock_final.py
→ LOCK + DOA 추적
→ TurtleBot 회전/정지/전진 명령
============================================================
"""

from pathlib import Path


# ============================================================
# 비콘 LOCK 프로그램
# ============================================================

#
# 실행 결과인 LOCK / UNLOCK만 받아옵니다.
#
# 만약 파일 위치가 바뀌면 이 경로만 변경하세요.
BEACON_LOCK_SCRIPT = (
    Path.home()
    / "resonator_filter"
    / "beacon1_lock_final.py"
)

# 비콘 ID
BEACON_ID = "BEACON_1"


# ============================================================
# ROS2 Topic
# ============================================================

# 팀원분 알고리즘의 LOCK 여부를 ROS2로 전달할 Topic
BEACON_LOCK_TOPIC = "/beacon/locked"

# ReSpeaker에서 XYXY 반복 패턴이 일정 신뢰도 이상 검출됐는지
BEACON_PATTERN_TOPIC = "/beacon/pattern_detected"

# 사전 검출 신뢰도(0.0~1.0)
BEACON_PATTERN_CONFIDENCE_TOPIC = "/beacon/pattern_confidence"

# TurtleBot 제어 모듈과의 경계
ROTATE_COMMAND_TOPIC = "/beacon/rotate_command_deg"
FORWARD_COMMAND_TOPIC = "/beacon/forward_command"
ALIGNED_TOPIC = "/beacon/aligned"

# 정렬 완료 이후에만 lock final을 시작하도록 하는 허가 신호
LOCK_ENABLE_TOPIC = "/beacon/lock_enable"

# 현재 검출된 비콘 ID
BEACON_ID_TOPIC = "/beacon/id"

# 사람이 발견됐는지 YOLO에서 나중에 받을 Topic
PERSON_DETECTED_TOPIC = "/person_detected"


# ============================================================
# MicArray A - DOA용
# ============================================================

# ★★★★★ 수정 포인트 ★★★★★
#
# 이것은 사용하는 MicArray B가 아닙니다.
#
# MicArray B = 공진관 + 패킷 검증
# MicArray A = 공진관 없음 + 방향 추정
#
# 현재 연결된 ReSpeaker 4 Mic Array의 ALSA 장치입니다.
#
# arecord --dump-hw-params 결과:
#   S16_LE / 16 kHz / 6 channels
#
# 패턴 검출기와 DOA가 동시에 ReSpeaker를 읽을 수 있도록 dsnoop을
# 사용합니다. hw:는 한 번에 한 프로세스만 장치를 열 수 있습니다.
DOA_DEVICE = "dsnoop:CARD=ArrayUAC10,DEV=0"


# 샘플링 주파수
#
# ReSpeaker 4 Mic Array USB firmware는 16 kHz / 6 channels를 사용합니다.
DOA_SAMPLE_RATE = 16000


# ★★★★★ 수정 포인트 ★★★★★
#
# ReSpeaker의 6채널 구성:
#   0 = processed audio for ASR
#   1~4 = 4개 raw microphone 채널
#   5 = playback reference
#
# arecord는 전체 6채널로 열고, DOA 계산에는 raw mic 4개만 사용합니다.
DOA_TOTAL_CHANNELS = 6

DOA_CHANNELS = [1, 2, 3, 4]

# ============================================================
# 마이크 위치
# ============================================================

# ReSpeaker USB Mic Array 하드웨어 도면 기준 좌표입니다.
#
# 도면 기준:
#   DOA 0°   = 보드 오른쪽
#   DOA 90°  = 보드 위쪽
#   DOA 180° = 보드 왼쪽
#   DOA 270° = 보드 아래쪽
#
# MIC1~MIC4는 각각 우상단/좌상단/좌하단/우하단에 있습니다.
# ReSpeaker 공식 예제의 MIC_DISTANCE_4=0.08127 m를 대향 마이크
# 사이의 지름으로 사용합니다.
#
# 단위: meter
MIC_DIAMETER_M = 0.08127
MIC_RADIUS_M = MIC_DIAMETER_M / 2.0

MIC_DIAGONAL_OFFSET_M = MIC_RADIUS_M / (2.0 ** 0.5)

MIC_POSITIONS_M = [
    ( MIC_DIAGONAL_OFFSET_M,  MIC_DIAGONAL_OFFSET_M),  # MIC1
    (-MIC_DIAGONAL_OFFSET_M,  MIC_DIAGONAL_OFFSET_M),  # MIC2
    (-MIC_DIAGONAL_OFFSET_M, -MIC_DIAGONAL_OFFSET_M),  # MIC3
    ( MIC_DIAGONAL_OFFSET_M, -MIC_DIAGONAL_OFFSET_M),  # MIC4
]

# ``doa_calibrate.py``가 생성하는 ReSpeaker별 보정값입니다. 이 파일에는
# 채널의 고정 위상 지연만 저장하며, Git에는 올리지 않는 장비별 파일입니다.
DOA_CALIBRATION_FILE = (
    Path.home()
    / "resonator_filter"
    / "doa_angle"
    / "doa_calibration.json"
)


# 공기 중 음속
SOUND_SPEED = 343.0


# ============================================================
# DOA 측정
# ============================================================

# 한 번 방향 추정할 때 사용하는 오디오 길이
DOA_BLOCK_SEC = 0.25

# Sipeed packet LOCK 뒤 ReSpeaker는 900 Hz만으로 DOA를 두 번 측정합니다.
# XYXY 순서와 네 주파수 검증은 공진관 Sipeed packet detector가 담당합니다.
DOA_MEASUREMENTS = 2
DOA_MIN_VALID_MEASUREMENTS = 2

# ReSpeaker DOA 전용 XYX gate 설정입니다.
#
# X 검증에는 감쇠가 큰 500 Hz를 쓰지 않고 900 Hz만 사용합니다.
# Y는 X1과 X2 사이에 비콘 구간이 있었는지만 느슨하게 확인합니다.
# 이 gate는 공진관 마이크의 최종 XYXY packet LOCK을 대체하지 않습니다.
# False: LOCK 뒤 900 Hz DOA 두 번만 측정합니다. XYX gate 설정은 향후
# 비교 실험을 위해 남겨 두되 기본 운용에서는 사용하지 않습니다.
DOA_REQUIRE_XYX_SEQUENCE = False
DOA_SEQUENCE_FRAME_SEC = 0.020
DOA_SEQUENCE_X_900_MIN_DBFS = -50.0
# Sipeed가 이미 전체 XYXY 패킷을 LOCK한 뒤의 보조 gate입니다. ReSpeaker의
# Y는 패킷 검증용이 아니라 X1과 X2를 분리하기 위한 표식이므로, 650/1050
# 중 하나가 이 값보다 크면 Y가 "있다"고만 판단합니다.
DOA_SEQUENCE_Y_PRESENCE_MIN_DBFS = -65.0
DOA_SEQUENCE_X_SYMBOL_MARGIN_DB = 3.0
DOA_SEQUENCE_X_FRAMES_REQUIRED = 7
# X1의 140 ms 분석창 뒤, 같은 X slot을 Y로 오인하지 않기 위한 대기입니다.
DOA_SEQUENCE_Y_MIN_DELAY_SEC = 0.120
DOA_SEQUENCE_Y_FRAMES_REQUIRED = 2
DOA_PAIR_MAX_ANGLE_DIFF_DEG = 18.0

# XYXY packet의 tone/guard 또는 packet gap에서 시작해도,
# 유효한 비콘 블록이 들어올 때까지 이 시간 동안 재시도합니다.
# beacon1_lock_final.py 기준 packet period는 약 2.4초입니다.
DOA_ESTIMATE_TIMEOUT_SEC = 8.0


# 비콘 신호가 존재하는 저주파 영역 중심으로 분석
#
# 패킷은 500~1050 Hz 영역에 있으므로
# DOA에서도 해당 음향 영역 주변만 활용하도록 설정.
DOA_BAND_LOW = 450
DOA_BAND_HIGH = 1150

# 기존 단일 tone TDOA 호환 설정
#
# 이 값은 레거시 phase-slope/XYX 보조 함수에서만 사용합니다. 현재 기본
# DOA 경로는 아래 SRP-PHAT tone 집합을 사용합니다.
#
# 패킷의 XYXY 판정과 Sipeed LOCK은 그대로 네 주파수를 모두 사용합니다.
# Sipeed가 이미 LOCK한 뒤에만 DOA가 시작되므로, 먼 거리에서 약해지는
# 동반 500 Hz를 DOA 샘플마다 다시 강제하지 않습니다. 이 값은 DOA에만
# 적용됩니다.
DOA_REFERENCE_TONES_HZ = (900.0,)

# ReSpeaker DOA의 활성 symbol은 high tone 하나로 구분합니다.
#
# SIPEED가 이미 XYXY 전체 패킷을 LOCK한 상태에서만 이 판단을 하므로,
# ReSpeaker는 감쇠가 큰 500/650 Hz를 다시 packet 검증에 쓰지 않습니다.
# X slot에서는 900 Hz, Y slot에서는 1050 Hz만 사용합니다.
DOA_ACTIVE_TONES_HZ = {
    "X": 900.0,
    "Y": 1050.0,
}
DOA_ACTIVE_TONE_MIN_DBFS = -55.0
DOA_ACTIVE_TONE_MARGIN_DB = 2.5

# 한 packet slot 안에 들어가는 깨끗한 활성 tone 분석 창입니다. 아래 MUSIC,
# SRP-PHAT, TDOA는 반드시 동일한 창과 동일한 활성 tone을 사용합니다.
DOA_ACTIVE_TONE_WINDOW_SEC = 0.128

# SRP-PHAT helper의 기본값입니다. runtime은 활성 X 또는 Y 하나만 넘깁니다.
DOA_SRP_PHAT_TONES_HZ = (900.0, 1050.0)
DOA_SRP_PHAT_TONE_HALF_BAND_HZ = 25.0
DOA_SRP_PHAT_ANGLE_STEP_DEG = 1.0
DOA_SRP_PHAT_MIN_CONFIDENCE = 1.5

# 활성 tone narrow-band MUSIC 설정입니다. 4개 raw microphone covariance에서
# direct-path 한 개 source를 가정하고 상위 후보만 SRP/TDOA와 교차 검증합니다.
DOA_MUSIC_SOURCE_COUNT = 1
DOA_MUSIC_SNAPSHOT_SEC = 0.016
DOA_MUSIC_DIAGONAL_LOADING = 0.03
DOA_MUSIC_PEAK_COUNT = 3
DOA_MUSIC_PEAK_SEPARATION_DEG = 20.0
DOA_MUSIC_MIN_PEAK_RATIO = 1.20
# Reject a reflected/ambiguous spectrum unless the strongest local maximum is
# at least 3 dB above the second candidate. This is intentionally stricter
# than the legacy peak-to-background ratio above.
DOA_MUSIC_MIN_TOP_PEAK_MARGIN_DB = 3.0

# The selected 900/1050-Hz tone must stand above adjacent non-beacon bins.
DOA_ACTIVE_TONE_MIN_SNR_DB = 6.0

# MUSIC 후보별 통합 점수 = MUSIC + SRP-PHAT + six-pair tone-TDOA support.
# 세 방법을 한 각도로 강제하지 않고, 후보마다 6개 pair residual/inlier를
# 비교해 반사음 후보를 탈락시킵니다.
DOA_FUSION_MUSIC_WEIGHT = 0.35
DOA_FUSION_SRP_WEIGHT = 0.30
DOA_FUSION_TDOA_WEIGHT = 0.35
DOA_FUSION_MIN_SCORE = 0.55
DOA_FUSION_MIN_TDOA_INLIERS = 3
# MUSIC's selected candidate must also be the SRP-PHAT direction to within
# this angle. Six-pair TDOA support is checked separately below.
DOA_FUSION_MAX_SRP_CANDIDATE_DIFF_DEG = 20.0

# 900 Hz 기준음의 최소 세기와 분석 창 길이입니다. 128 ms 창은 250 ms
# tone 내부에 들어가므로 tone 경계/guard가 TDOA 위상에 섞이는 것을 줄입니다.
# packet LOCK 뒤 유효 900 Hz 실측치(-51~-53 dBFS)를 수용합니다. 전체
# band 기준(-55 dBFS)과 TDOA·두 번 각도 일치 검증은 그대로 유지합니다.
DOA_REFERENCE_TONE_MIN_DBFS = -55.0
DOA_REFERENCE_WINDOW_SEC = 0.128

# False: Sipeed packet LOCK 이후에는 기준음 자체만으로 DOA 창을 고릅니다.
# True로 바꾸면 기존처럼 매 DOA 창마다 X/Y의 두 성분을 모두 요구합니다.
DOA_REQUIRE_SYMBOL_PAIR = False

# DOA는 무음/잡음에서 각도를 만들어내면 안 됩니다. 900 Hz 구간 자체는
# processed gate가 -45 dBFS 이상일 때만 선택하므로, raw 채널은 위상 계산에
# 필요한 최소 수준까지만 허용합니다. 비콘 OFF 실측(-66~-67 dBFS)은 제외됩니다.
DOA_MIN_SIGNAL_DBFS = -65.0

# 마이크별 TDOA가 하나의 입사 방향으로 설명되지 않으면 폐기합니다.
DOA_MAX_DIRECTION_RESIDUAL = 0.45

# 모든 TDOA를 MIC1 하나에만 비교하면, 기준 마이크 하나가 가려지거나
# 반사음 영향을 크게 받을 때 세 쌍이 한꺼번에 실패합니다. 아래 방식은
# 4개 마이크의 6개 쌍을 모두 비교하고, 하나의 방향과 맞지 않는 쌍은
# 버린 뒤 남은 쌍으로 방향을 계산합니다.
#
# 20 mm는 보정 후 실제 실내 반사 환경의 900 Hz 위상 흔들림을 1차로
# 수용하는 값입니다. 최소 3개 마이크 쌍 일치와 XYX 두 각도 검증은
# 그대로 유지하므로, 단일 반사음이나 두 쌍만 유효한 측정은 폐기합니다.
DOA_USE_ROBUST_ALL_PAIRS = True
DOA_PAIR_INLIER_TOLERANCE_M = 0.020
DOA_MIN_INLIER_PAIRS = 3

# 기존 다중 샘플 방식의 호환 설정입니다. 기본 경로는 위 XYX gate의
# 두 X 구간 각도 일치 검증을 사용합니다.
DOA_OUTLIER_TOLERANCE_DEG = 18.0
DOA_MAX_ANGLE_SPREAD_DEG = 18.0


# ============================================================
# ReSpeaker XYXY 사전 검출
# ============================================================

# 500/650/900/1050 Hz 에너지와 XYXY 시간 순서를 사용합니다.
# MP3 전체 파형을 직접 비교하지 않고, 거리/음량/잔향에 덜 민감한
# 주파수 특징과 반복 주기를 비교합니다.
PATTERN_FRAME_SEC = 0.020
PATTERN_FFT_SIZE = 1024
PATTERN_TONE_HALF_BAND_HZ = 35.0
PATTERN_MIN_COMPONENT_DBFS = -48.0
PATTERN_PAIR_MARGIN_DB = 6.0

PATTERN_SEARCH_WINDOW_FRAMES = 7
PATTERN_SEARCH_X_VOTES = 5
PATTERN_SLOT_SEC = 0.350
PATTERN_SLOT_WINDOW_HALF_SEC = 0.140
PATTERN_SLOT_PASS_VOTES = 3
# 사전 검출은 Sipeed final보다 넓게 받아 후보를 만들고,
# 최종 LOCK은 Sipeed의 더 엄격한 판정으로 확정합니다.
PATTERN_SLOT_MAX_OPPOSITE_VOTES = 3
PATTERN_SLOT_PASS_LEAD = 2
PATTERN_SLOT_WRONG_VOTES = 3
PATTERN_SLOT_WRONG_LEAD = 2
PATTERN_FOLLOWUP_PASSES_REQUIRED = 2
PATTERN_MAX_WRONG_SLOTS = 0

# 단일 packet이 아니라 반복 packet을 요구합니다.
PATTERN_PACKETS_TO_DETECT = 2
PATTERN_INTERVAL_MIN_SEC = 1.8
PATTERN_INTERVAL_MAX_SEC = 3.2
PATTERN_TIMEOUT_SEC = 5.0
# packet FSM이 경계 프레임을 한 번 놓쳐도, X/Y 음 자체가 계속
# 들어오는 동안에는 사전 검출 상태를 유지합니다.
PATTERN_ACTIVITY_TIMEOUT_SEC = 2.5


# ============================================================
# 상태 전이 / 추적 조건
# ============================================================

# DOA 결과는 ReSpeaker 기준 상대각입니다.
ALIGNMENT_TOLERANCE_DEG = 8.0
TRACKING_TOLERANCE_DEG = 12.0
TRACKING_EXIT_TOLERANCE_DEG = 18.0
ALIGNMENT_STABLE_MEASUREMENTS = 2
DOA_RECHECK_SEC = 1.5

# 실제 TurtleBot 제어기가 나중에 이 topic을 publish합니다.
# True가 들어오기 전에는 Sipeed lock final을 시작하지 않습니다.


# ★★★★★ 방향이 좌우 반대로 나오면 이것만 변경 ★★★★★
#
# 실제 0° 위치(보드 오른쪽)에서 측정했을 때 약 180° 반대 방향으로
# 안정적으로 출력되어 부호를 반전합니다.
DOA_SIGN = -1.0


# ============================================================
# Navigation
# ============================================================

GLOBAL_FRAME = "map"
ROBOT_FRAME = "base_link"

NAV2_ACTION_NAME = "navigate_to_pose"


# 한 번 DOA 측정 후 이동할 거리
#
# 너무 길게 잡으면 방향 오차가 크게 누적됩니다.
# 처음 실험은 1.0~1.5 m 권장.
STEP_DISTANCE = 1.5


# Goal 도착 후 다시 방향 측정
REMEASURE_AFTER_GOAL = True


# ============================================================
# 안전 설정
# ============================================================

# 비콘 LOCK이 풀리면 현재 Navigation Goal 취소
STOP_ON_UNLOCK = True

# YOLO에서 사람이 발견되면 이동 중지
STOP_ON_PERSON_DETECTED = True


# ============================================================
# DEBUG
# ============================================================

DEBUG = True
