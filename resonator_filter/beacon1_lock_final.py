#!/usr/bin/env python3

import json
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.signal import sosfilt


# ============================================================
# AUDIO
# ============================================================

DEVICE = "hw:CARD=MicArray,DEV=0"

FS = 48000
CHANNELS = 8

FRAME_MS = 20
FRAME_SEC = FRAME_MS / 1000.0
FRAME_SAMPLES = int(FS * FRAME_SEC)


# ============================================================
# RESONATOR CHANNELS
# ============================================================

TARGETS = [500, 650, 900, 1050]

CHANNEL_MAP = {
    500: 2,      # ALSA CH3
    650: 0,      # ALSA CH1
    900: 5,      # ALSA CH6
    1050: 4,     # ALSA CH5
}

FILTER_FILE = (
    Path.home()
    / "resonator_filter"
    / "outputs"
    / "filter_bank_500_650_900_1050.npz"
)

CAL_REF_DBFS = {
    500: -16.43,
    650: -15.67,
    900: -10.15,
    1050: -20.67,
}


# ============================================================
# PHYSICAL SYMBOLS / THRESHOLDS
# ============================================================

# X = 500 + 900 Hz
# Y = 650 + 1050 Hz
BEACON_CODE = "XYXY"

# 지금까지 검증한 값은 그대로 유지
MIN_COMPONENT_SCORE_DB = -36.0
MIN_PAIR_MARGIN_DB = 10.0


# ============================================================
# TRANSMITTER TIMING
# ============================================================

TONE_SEC = 0.250
GUARD_SEC = 0.100
SLOT_SEC = TONE_SEC + GUARD_SEC       # 0.350 s


# ============================================================
# FIRST-X SEARCH
# ============================================================

# 첫 X는 엄격하게 찾는다.
SEARCH_WINDOW_FRAMES = 7              # 140 ms
SEARCH_X_VOTES = 5                    # 5/7


# ============================================================
# SLOT CLASSIFICATION
#
# 첫 X를 잡은 뒤 Y / X / Y는 예상 시간창에서 판정한다.
#
# PASS:
#   expected >= 3 votes
#   opposite <= 2 votes
#   expected - opposite >= 2
#
# WRONG:
#   opposite >= 3 votes
#   opposite - expected >= 2
#
# 그 외:
#   ERASE
#
# v5에서는 ERASE 1개까지 허용한다.
# 즉 X1 + 나머지 3 slot 중 최소 2개가 PASS이면 packet 인정.
# WRONG이 하나라도 있으면 packet reject.
# ============================================================

SLOT_WINDOW_HALF_SEC = 0.140

SLOT_PASS_VOTES = 3
SLOT_MAX_OPPOSITE_FOR_PASS = 2
SLOT_PASS_LEAD = 2

SLOT_WRONG_VOTES = 3
SLOT_WRONG_LEAD = 2

# X1은 이미 sync 성공으로 확정됐으므로,
# Y1 / X2 / Y2 중 최소 2개 PASS
FOLLOWUP_PASSES_REQUIRED = 2

# WRONG symbol은 허용하지 않음
MAX_WRONG_SLOTS = 0


# ============================================================
# FINAL LOCK STATE MACHINE
#
# TX timing:
#   XYXY packet = 4 * 0.350 s = 1.4 s
#   packet gap  = 1.0 s
#   nominal packet period = 2.4 s
#
# LOCK acquisition:
#   - 2 VALID packets are required.
#   - The two packets must arrive with a plausible packet-to-packet
#     interval. This prevents two unrelated VALID events many seconds
#     apart from producing LOCK.
#
# LOCK hold:
#   - Once LOCKED, any subsequent VALID packet refreshes keepalive.
#   - If no VALID packet is received for UNLOCK_TIMEOUT_SEC,
#     LOCK is released.
# ============================================================

PACKETS_TO_LOCK = 2

PACKET_GAP_SEC = 1.0
NOMINAL_PACKET_PERIOD_SEC = (
    4 * SLOT_SEC
    + PACKET_GAP_SEC
)                                       # 2.4 s

LOCK_INTERVAL_MIN_SEC = 1.8
LOCK_INTERVAL_MAX_SEC = 3.2

UNLOCK_TIMEOUT_SEC = 10.0

# One compact line is emitted only after a fully validated XYXY packet.  The
# ROS bridge consumes it as evidence; it never changes the production LOCK
# state machine.  Keep this marker stable because packet_lock_node parses it.
PACKET_METRIC_MARKER = "@@BEACON_PACKET_METRIC@@"


# ============================================================
# FILTER LOAD
# ============================================================

if not FILTER_FILE.exists():
    raise FileNotFoundError(
        f"Filter bank not found: {FILTER_FILE}"
    )

bank = np.load(FILTER_FILE)

filters = {
    freq: bank[f"sos_{freq}"]
    for freq in TARGETS
}

filter_states = {}

for freq in TARGETS:
    sections = filters[freq].shape[0]

    filter_states[freq] = np.zeros(
        (sections, 2),
        dtype=np.float64
    )


# ============================================================
# UTILS
# ============================================================

def rms_db(x):
    rms = np.sqrt(np.mean(x ** 2))

    if rms <= 0:
        return -120.0

    return 20.0 * np.log10(rms)


def read_exact(stream, nbytes):
    chunks = []
    received = 0

    while received < nbytes:
        chunk = stream.read(nbytes - received)

        if not chunk:
            break

        chunks.append(chunk)
        received += len(chunk)

    return b"".join(chunks)


# ============================================================
# ARECORD
# ============================================================

cmd = [
    "arecord",
    "-q",
    "-D", DEVICE,
    "-t", "raw",
    "-f", "S16_LE",
    "-r", str(FS),
    "-c", str(CHANNELS),
]

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    bufsize=0
)

BYTES_PER_FRAME = (
    FRAME_SAMPLES
    * CHANNELS
    * 2
)


# ============================================================
# SEARCH STATE
# ============================================================

search_votes = deque(
    maxlen=SEARCH_WINDOW_FRAMES
)


# ============================================================
# PACKET STATE
# ============================================================

candidate_active = False
candidate_t0 = None

# follow-up slot:
# 1 = Y1
# 2 = X2
# 3 = Y2
next_slot_index = 1

slot_expected_votes = 0
slot_opposite_votes = 0

followup_passes = 0
followup_erases = 0
followup_wrongs = 0

candidate_count = 0
candidate_rejects = 0
candidate_tone_levels = {
    "X": {500: [], 900: []},
    "Y": {650: [], 1050: []},
}
candidate_tone_scores = {
    "X": {500: [], 900: []},
    "Y": {650: [], 1050: []},
}


# ============================================================
# LOCK STATE
# ============================================================

last_valid_packet_time = None

locked = False
total_valid_packets = 0

# LOCK acquisition state
acquire_streak = 0
last_acquire_packet_time = None

lock_count = 0
unlock_count = 0


def expected_symbol_for_slot(slot_index):
    return {
        1: "Y",
        2: "X",
        3: "Y",
    }[slot_index]


def reset_slot_votes():
    global slot_expected_votes
    global slot_opposite_votes

    slot_expected_votes = 0
    slot_opposite_votes = 0


def clear_candidate():
    global candidate_active
    global candidate_t0
    global next_slot_index
    global followup_passes
    global followup_erases
    global followup_wrongs
    global candidate_tone_levels
    global candidate_tone_scores

    candidate_active = False
    candidate_t0 = None
    next_slot_index = 1

    followup_passes = 0
    followup_erases = 0
    followup_wrongs = 0
    candidate_tone_levels = {
        "X": {500: [], 900: []},
        "Y": {650: [], 1050: []},
    }
    candidate_tone_scores = {
        "X": {500: [], 900: []},
        "Y": {650: [], 1050: []},
    }

    reset_slot_votes()
    search_votes.clear()


def start_candidate(now, scores, levels, x_score, y_score):
    global candidate_active
    global candidate_t0
    global next_slot_index
    global candidate_count
    global followup_passes
    global followup_erases
    global followup_wrongs
    global candidate_tone_levels
    global candidate_tone_scores

    candidate_count += 1

    candidate_active = True
    candidate_t0 = now
    next_slot_index = 1

    followup_passes = 0
    followup_erases = 0
    followup_wrongs = 0
    candidate_tone_levels = {
        "X": {500: [levels[500]], 900: [levels[900]]},
        "Y": {650: [], 1050: []},
    }
    candidate_tone_scores = {
        "X": {500: [scores[500]], 900: [scores[900]]},
        "Y": {650: [], 1050: []},
    }

    reset_slot_votes()

    print()
    print(
        f"[SYNC #{candidate_count}] X1 acquired"
        f" | X={x_score:.1f}"
        f" Y={y_score:.1f}"
        f" | 500={scores[500]:.1f}"
        f" 900={scores[900]:.1f}"
    )


def record_candidate_frame(frame_symbol, scores, levels):
    """Accumulate only the tone pair active in each validated packet frame."""

    if frame_symbol == "X":
        for freq in (500, 900):
            candidate_tone_levels["X"][freq].append(levels[freq])
            candidate_tone_scores["X"][freq].append(scores[freq])
    elif frame_symbol == "Y":
        for freq in (650, 1050):
            candidate_tone_levels["Y"][freq].append(levels[freq])
            candidate_tone_scores["Y"][freq].append(scores[freq])


def emit_valid_packet_metric(now):
    """Print identity and direction metrics for one fully valid XYXY packet.

    ``level_dbfs`` remains the four-tone weak-link value. It is deliberately
    conservative and is retained for diagnosing packet identity. The
    resonator direction experiment showed that 500/650 Hz can be attenuated
    independently of the front-facing 900/1050 Hz channels, so a separate
    direction metric is published without weakening the XYXY validation.
    """

    level_medians = {}
    score_medians = {}
    for symbol, frequencies in (("X", (500, 900)), ("Y", (650, 1050))):
        for freq in frequencies:
            levels = candidate_tone_levels[symbol][freq]
            scores = candidate_tone_scores[symbol][freq]
            if not levels or not scores:
                return
            level_medians[freq] = float(np.median(levels))
            score_medians[freq] = float(np.median(scores))

    payload = {
        "monotonic_sec": round(float(now), 6),
        # Weak-link values make a loud partial packet unable to look strong.
        "level_dbfs": round(min(level_medians.values()), 3),
        "quality_db": round(min(score_medians.values()), 3),
        # This is emitted only after the full XYXY packet above is valid.
        # Use the weaker of X/900 and Y/1050 so one strong symbol cannot make
        # a reflection or partial packet look directional. Unlike
        # ``level_dbfs``, it intentionally does not include 500/650 Hz.
        "direction_level_dbfs": round(
            min(level_medians[900], level_medians[1050]), 3
        ),
        "direction_quality_db": round(
            min(score_medians[900], score_medians[1050]), 3
        ),
        "raw_dbfs": {str(freq): round(level_medians[freq], 3) for freq in TARGETS},
        "scores_db": {str(freq): round(score_medians[freq], 3) for freq in TARGETS},
    }
    print(f"{PACKET_METRIC_MARKER} {json.dumps(payload, sort_keys=True)}")


def slot_center_time(slot_index):
    return (
        candidate_t0
        + slot_index * SLOT_SEC
    )


def classify_slot(expected_votes, opposite_votes):
    if (
        expected_votes >= SLOT_PASS_VOTES
        and
        opposite_votes <= SLOT_MAX_OPPOSITE_FOR_PASS
        and
        expected_votes - opposite_votes >= SLOT_PASS_LEAD
    ):
        return "PASS"

    if (
        opposite_votes >= SLOT_WRONG_VOTES
        and
        opposite_votes - expected_votes >= SLOT_WRONG_LEAD
    ):
        return "WRONG"

    return "ERASE"


def register_valid_packet(now):
    global total_valid_packets
    global last_valid_packet_time
    global locked

    global acquire_streak
    global last_acquire_packet_time

    global lock_count

    total_valid_packets += 1

    print()
    print(
        ">>> VALID XYXY PACKET <<<"
    )
    print(
        f"total valid packets = "
        f"{total_valid_packets}"
    )

    # --------------------------------------------------------
    # Already LOCKED:
    # Any valid packet refreshes keepalive.
    # We do not require perfect 2.4 s periodicity while locked,
    # because one or more packets may be missed under interference.
    # --------------------------------------------------------

    if locked:
        last_valid_packet_time = now

        print(
            ">>> LOCK 유지 / keepalive refreshed <<<"
        )
        print()

        return

    # --------------------------------------------------------
    # SEARCH / LOCK ACQUISITION
    # --------------------------------------------------------

    if last_acquire_packet_time is None:
        acquire_streak = 1
        last_acquire_packet_time = now

        print(
            f"[LOCK CHECK] first valid packet "
            f"= {acquire_streak}/{PACKETS_TO_LOCK}"
        )
        print()

        return

    dt = (
        now
        - last_acquire_packet_time
    )

    interval_ok = (
        LOCK_INTERVAL_MIN_SEC
        <= dt
        <= LOCK_INTERVAL_MAX_SEC
    )

    if interval_ok:
        acquire_streak += 1

        print(
            f"[LOCK CHECK] packet interval "
            f"= {dt:.3f} s -> PASS"
        )

    else:
        # Current valid packet becomes the new first packet.
        acquire_streak = 1

        print(
            f"[LOCK CHECK] packet interval "
            f"= {dt:.3f} s -> RESET"
        )
        print(
            f"             allowed "
            f"{LOCK_INTERVAL_MIN_SEC:.1f}"
            f"~{LOCK_INTERVAL_MAX_SEC:.1f} s"
        )

    last_acquire_packet_time = now

    print(
        f"[LOCK CHECK] acquire streak "
        f"= {acquire_streak}/{PACKETS_TO_LOCK}"
    )

    if acquire_streak >= PACKETS_TO_LOCK:
        locked = True
        last_valid_packet_time = now
        lock_count += 1

        # acquisition state no longer needed while locked
        acquire_streak = 0
        last_acquire_packet_time = None

        print()
        print(
            "##############################"
        )
        print(
            ">>> LOCK = BEACON_1 <<<"
        )
        print(
            "##############################"
        )

    print()


def finish_candidate(now):
    global candidate_rejects

    packet_ok = (
        followup_passes >= FOLLOWUP_PASSES_REQUIRED
        and
        followup_wrongs <= MAX_WRONG_SLOTS
    )

    print(
        f"[PACKET RESULT] "
        f"PASS={followup_passes} "
        f"ERASE={followup_erases} "
        f"WRONG={followup_wrongs}"
        f" -> "
        f"{'VALID' if packet_ok else 'REJECT'}"
    )

    if packet_ok:
        emit_valid_packet_metric(now)
        register_valid_packet(now)

    else:
        candidate_rejects += 1

    clear_candidate()


# ============================================================
# START
# ============================================================

print()
print("=" * 86)
print("BEACON 1 : FINAL LOCK DECODER")
print("=" * 86)
print("X = 500 + 900 Hz")
print("Y = 650 + 1050 Hz")
print()
print(f"Beacon code          : {BEACON_CODE}")
print(f"Frame                : {FRAME_MS} ms")
print(
    f"SEARCH X vote        : "
    f"{SEARCH_X_VOTES}/"
    f"{SEARCH_WINDOW_FRAMES}"
)
print(
    f"Component threshold  : "
    f"{MIN_COMPONENT_SCORE_DB:.1f} dB"
)
print(
    f"Pair margin          : "
    f"{MIN_PAIR_MARGIN_DB:.1f} dB"
)
print(
    f"Slot                 : "
    f"{SLOT_SEC*1000:.0f} ms"
)
print(
    f"Expected-slot window : "
    f"±{SLOT_WINDOW_HALF_SEC*1000:.0f} ms"
)
print(
    f"Slot PASS            : "
    f"expected>={SLOT_PASS_VOTES}, "
    f"opposite<={SLOT_MAX_OPPOSITE_FOR_PASS}, "
    f"lead>={SLOT_PASS_LEAD}"
)
print(
    f"Packet acceptance    : "
    f"X1 + >= {FOLLOWUP_PASSES_REQUIRED}/3 follow-up PASS, "
    f"WRONG <= {MAX_WRONG_SLOTS}"
)
print(
    f"LOCK acquire         : "
    f"{PACKETS_TO_LOCK} VALID packets"
)
print(
    f"Packet interval      : "
    f"{LOCK_INTERVAL_MIN_SEC:.1f}"
    f"~{LOCK_INTERVAL_MAX_SEC:.1f} s "
    f"(nominal {NOMINAL_PACKET_PERIOD_SEC:.1f} s)"
)
print(
    f"LOCK timeout         : "
    f"{UNLOCK_TIMEOUT_SEC:.1f} s"
)
print()
print("Ctrl+C 종료")
print("=" * 86)
print()


try:
    while True:
        raw_bytes = read_exact(
            proc.stdout,
            BYTES_PER_FRAME
        )

        if len(raw_bytes) != BYTES_PER_FRAME:
            print("Audio stream 종료")
            break

        raw = np.frombuffer(
            raw_bytes,
            dtype="<i2"
        ).reshape(
            -1,
            CHANNELS
        )

        # ====================================================
        # CLIPPING
        # ====================================================

        rail_hits = 0

        for freq in TARGETS:
            ch = CHANNEL_MAP[freq]

            rail_hits += int(
                np.sum(
                    np.abs(
                        raw[:, ch].astype(np.int32)
                    ) >= 32760
                )
            )

        data = (
            raw.astype(np.float64)
            / 32768.0
        )

        # ====================================================
        # BAND SCORES
        # ====================================================

        scores = {}
        levels = {}

        for freq in TARGETS:
            ch = CHANNEL_MAP[freq]

            y, filter_states[freq] = sosfilt(
                filters[freq],
                data[:, ch],
                zi=filter_states[freq]
            )

            level = rms_db(y)
            levels[freq] = level

            scores[freq] = (
                level
                - CAL_REF_DBFS[freq]
            )

        x_score = min(
            scores[500],
            scores[900]
        )

        y_score = min(
            scores[650],
            scores[1050]
        )

        frame_symbol = None

        if (
            x_score >= MIN_COMPONENT_SCORE_DB
            and
            x_score - y_score >= MIN_PAIR_MARGIN_DB
            and
            rail_hits == 0
        ):
            frame_symbol = "X"

        elif (
            y_score >= MIN_COMPONENT_SCORE_DB
            and
            y_score - x_score >= MIN_PAIR_MARGIN_DB
            and
            rail_hits == 0
        ):
            frame_symbol = "Y"

        now = time.monotonic()

        # ====================================================
        # SEARCH FIRST X
        # ====================================================

        if not candidate_active:
            search_votes.append(
                frame_symbol
            )

            if (
                search_votes.count("X")
                >= SEARCH_X_VOTES
            ):
                start_candidate(
                    now,
                    scores,
                    levels,
                    x_score,
                    y_score
                )

                search_votes.clear()

        # ====================================================
        # PACKET-SYNCHRONIZED FOLLOW-UP SLOTS
        # ====================================================

        else:
            record_candidate_frame(
                frame_symbol,
                scores,
                levels,
            )
            expected = expected_symbol_for_slot(
                next_slot_index
            )

            opposite = (
                "Y"
                if expected == "X"
                else "X"
            )

            center = slot_center_time(
                next_slot_index
            )

            window_start = (
                center
                - SLOT_WINDOW_HALF_SEC
            )

            window_end = (
                center
                + SLOT_WINDOW_HALF_SEC
            )

            if (
                window_start
                <= now
                <= window_end
            ):
                if frame_symbol == expected:
                    slot_expected_votes += 1

                elif frame_symbol == opposite:
                    slot_opposite_votes += 1

            if now > window_end:
                result = classify_slot(
                    slot_expected_votes,
                    slot_opposite_votes
                )

                if result == "PASS":
                    followup_passes += 1

                elif result == "ERASE":
                    followup_erases += 1

                else:
                    followup_wrongs += 1

                print(
                    f"[SLOT {next_slot_index+1}] "
                    f"expect={expected}"
                    f" votes={slot_expected_votes}"
                    f" opposite={slot_opposite_votes}"
                    f" -> {result}"
                )

                if next_slot_index == 3:
                    finish_candidate(now)

                else:
                    next_slot_index += 1
                    reset_slot_votes()

        # ====================================================
        # LOCK TIMEOUT
        # ====================================================

        if (
            locked
            and
            last_valid_packet_time is not None
            and
            now - last_valid_packet_time
            > UNLOCK_TIMEOUT_SEC
        ):
            locked = False
            unlock_count += 1

            acquire_streak = 0
            last_acquire_packet_time = None
            last_valid_packet_time = None

            clear_candidate()

            print()
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            print(
                ">>> UNLOCK = BEACON_1 <<<"
            )
            print(
                "reason: VALID packet timeout"
            )
            print(
                "state : SEARCH"
            )
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            print()


except KeyboardInterrupt:
    print()
    print("Decoder 종료")
    print(
        f"Candidate count           = "
        f"{candidate_count}"
    )
    print(
        f"Candidate rejects         = "
        f"{candidate_rejects}"
    )
    print(
        f"Final total valid packets = "
        f"{total_valid_packets}"
    )
    print(
        f"LOCK acquisitions        = "
        f"{lock_count}"
    )
    print(
        f"UNLOCK timeouts          = "
        f"{unlock_count}"
    )
    print(
        f"Final state              = "
        f"{'LOCKED' if locked else 'SEARCH'}"
    )


finally:
    proc.terminate()

    try:
        proc.wait(timeout=2)

    except subprocess.TimeoutExpired:
        proc.kill()
