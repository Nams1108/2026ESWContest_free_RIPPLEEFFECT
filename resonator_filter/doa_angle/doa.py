"""
============================================================
doa.py

[역할]

MicArray A의 여러 마이크 사이 시간차(TDOA)를 이용하여
비콘 소리가 들어오는 방향(DOA)을 계산합니다.

LOCK 이후에만 실행되는 것을 전제로 합니다.

현재 방식
------------------------------------------------------------
1. SIPEED의 완전한 XYXY packet LOCK 확인(ROS adapter 담당)
2. ReSpeaker raw CH1~CH4에서 현재 X(900 Hz) / Y(1050 Hz) 선택
3. 활성 tone narrow-band MUSIC 상위 방향 후보 계산
4. 각 후보를 SRP-PHAT 및 6개 mic-pair TDOA residual로 교차 채점
5. 서로 독립적인 symbol/frame 방향만 원형 각도 군집
6. 일치하는 결과만 최종 DOA로 사용, 나머지는 NO DATA

주의
------------------------------------------------------------
PDF에는 MicArray A의 실제 마이크 간격/배치가 없습니다.

따라서 config.py의 MIC_POSITIONS_M 값은
실제 하드웨어 확인 후 반드시 수정해야 합니다.
============================================================
"""

import json
import math
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from scipy.signal import (
    butter,
    sosfiltfilt,
)

from audio import DOAAudioReader

from config import (
    DOA_SAMPLE_RATE,
    DOA_CHANNELS,
    MIC_POSITIONS_M,
    DOA_CALIBRATION_FILE,
    SOUND_SPEED,
    DOA_BLOCK_SEC,
    DOA_MEASUREMENTS,
    DOA_ESTIMATE_TIMEOUT_SEC,
    DOA_REQUIRE_XYX_SEQUENCE,
    DOA_SEQUENCE_FRAME_SEC,
    DOA_SEQUENCE_X_900_MIN_DBFS,
    DOA_SEQUENCE_Y_PRESENCE_MIN_DBFS,
    DOA_SEQUENCE_X_SYMBOL_MARGIN_DB,
    DOA_SEQUENCE_X_FRAMES_REQUIRED,
    DOA_SEQUENCE_Y_MIN_DELAY_SEC,
    DOA_SEQUENCE_Y_FRAMES_REQUIRED,
    DOA_PAIR_MAX_ANGLE_DIFF_DEG,
    DOA_BAND_LOW,
    DOA_BAND_HIGH,
    DOA_REFERENCE_TONES_HZ,
    DOA_REFERENCE_TONE_MIN_DBFS,
    DOA_REFERENCE_WINDOW_SEC,
    DOA_ACTIVE_TONES_HZ,
    DOA_ACTIVE_TONE_MIN_DBFS,
    DOA_ACTIVE_TONE_MARGIN_DB,
    DOA_ACTIVE_TONE_WINDOW_SEC,
    DOA_SRP_PHAT_TONES_HZ,
    DOA_SRP_PHAT_TONE_HALF_BAND_HZ,
    DOA_SRP_PHAT_ANGLE_STEP_DEG,
    DOA_SRP_PHAT_MIN_CONFIDENCE,
    DOA_MUSIC_SOURCE_COUNT,
    DOA_MUSIC_SNAPSHOT_SEC,
    DOA_MUSIC_DIAGONAL_LOADING,
    DOA_MUSIC_PEAK_COUNT,
    DOA_MUSIC_PEAK_SEPARATION_DEG,
    DOA_MUSIC_MIN_PEAK_RATIO,
    DOA_MUSIC_MIN_TOP_PEAK_MARGIN_DB,
    DOA_ACTIVE_TONE_MIN_SNR_DB,
    DOA_FUSION_MUSIC_WEIGHT,
    DOA_FUSION_SRP_WEIGHT,
    DOA_FUSION_TDOA_WEIGHT,
    DOA_FUSION_MIN_SCORE,
    DOA_FUSION_MIN_TDOA_INLIERS,
    DOA_FUSION_MAX_SRP_CANDIDATE_DIFF_DEG,
    DOA_REQUIRE_SYMBOL_PAIR,
    DOA_SIGN,
    DOA_MIN_SIGNAL_DBFS,
    DOA_MAX_DIRECTION_RESIDUAL,
    DOA_USE_ROBUST_ALL_PAIRS,
    DOA_PAIR_INLIER_TOLERANCE_M,
    DOA_MIN_INLIER_PAIRS,
    DOA_MIN_VALID_MEASUREMENTS,
    DOA_OUTLIER_TOLERANCE_DEG,
    DOA_MAX_ANGLE_SPREAD_DEG,
    PATTERN_MIN_COMPONENT_DBFS,
    PATTERN_PAIR_MARGIN_DB,
    DEBUG,
)


_BEACON_TONES_HZ = (500.0, 650.0, 900.0, 1050.0)


def load_channel_phase_offsets(path=DOA_CALIBRATION_FILE):
    """Load per-channel delay offsets measured by ``doa_calibrate.py``.

    The first raw microphone is the reference and therefore has a fixed zero
    offset. No calibration file means the estimator remains usable for
    synthetic tests, but live DOA should be calibrated before navigation.
    """

    path = Path(path).expanduser()
    offsets = np.zeros(len(DOA_CHANNELS), dtype=np.float64)
    metadata = {}

    if not path.exists():
        return offsets, metadata

    try:
        with path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        loaded = np.asarray(
            metadata["channel_offsets_sec"], dtype=np.float64
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"DOA calibration 파일을 읽을 수 없습니다: {exc}")

    if loaded.shape != offsets.shape or not np.all(np.isfinite(loaded)):
        raise ValueError("DOA calibration channel_offsets_sec 형식이 잘못되었습니다.")

    loaded = loaded - loaded[0]
    return loaded, metadata


def apply_channel_phase_offsets(pair_taus, channel_offsets_sec):
    """Remove fixed per-channel delays from ``(ref, signal, tau)`` triples."""

    offsets = np.asarray(channel_offsets_sec, dtype=np.float64)
    if offsets.shape != (len(DOA_CHANNELS),):
        raise ValueError("DOA channel offset 개수가 마이크 채널 수와 다릅니다.")

    corrected = []
    for reference_index, signal_index, tau in pair_taus:
        corrected_tau = float(tau) - (
            offsets[signal_index] - offsets[reference_index]
        )
        corrected.append((reference_index, signal_index, corrected_tau))
    return corrected


# ============================================================
# BPF 생성
# ============================================================

_DOA_FILTER = butter(
    4,
    [
        DOA_BAND_LOW,
        DOA_BAND_HIGH,
    ],
    btype="bandpass",
    fs=DOA_SAMPLE_RATE,
    output="sos",
)


def bandpass(data):
    """
    DOA 계산 전에 Beacon 신호가 존재하는
    저주파 영역을 중심으로 filtering.
    """

    return sosfiltfilt(
        _DOA_FILTER,
        data,
        axis=0,
    )


# ============================================================
# GCC-PHAT
# ============================================================

def gcc_phat(
    signal,
    reference,
    fs,
    max_tau=None,
):
    """
    두 마이크의 도착 시간차(TDOA)를 구합니다.

    반환값:
        delay(sec)
    """

    n = (
        len(signal)
        + len(reference)
    )

    nfft = 1

    while nfft < n:
        nfft *= 2


    sig_fft = np.fft.rfft(
        signal,
        n=nfft,
    )

    ref_fft = np.fft.rfft(
        reference,
        n=nfft,
    )


    cross = (
        sig_fft
        * np.conj(ref_fft)
    )


    magnitude = np.abs(
        cross
    )

    cross = cross / np.maximum(
        magnitude,
        1e-15,
    )


    correlation = np.fft.irfft(
        cross,
        n=nfft,
    )


    max_shift = (
        nfft // 2
    )

    if max_tau is not None:

        max_shift = min(
            int(fs * max_tau),
            max_shift,
        )


    correlation = np.concatenate(
        (
            correlation[-max_shift:],
            correlation[:max_shift + 1],
        )
    )


    shift = (
        np.argmax(
            np.abs(
                correlation
            )
        )
        - max_shift
    )


    tau = (
        shift
        / float(fs)
    )


    return tau


def srp_phat_doa(
    audio,
    fs,
    mic_pos,
    *,
    tones_hz=DOA_SRP_PHAT_TONES_HZ,
    tone_half_band_hz=DOA_SRP_PHAT_TONE_HALF_BAND_HZ,
    angle_step_deg=DOA_SRP_PHAT_ANGLE_STEP_DEG,
    sound_speed=SOUND_SPEED,
    sign=1.0,
    channel_offsets_sec=None,
):
    """Estimate DOA with a Go2-style SRP-PHAT score search.

    ``audio`` contains only the four raw ReSpeaker channels.  The search is
    restricted to the configured beacon tone neighborhoods because the
    current beacon is narrow-band.  All six microphone pairs contribute to
    every candidate angle, avoiding the single wrapped-phase TDOA decision
    used by the previous runtime path.

    ``channel_offsets_sec`` keeps the existing calibration file useful.  A
    fixed delay difference is removed from each pair's cross spectrum before
    its PHAT score is accumulated.

    Returns ``(angle_deg, confidence, scores)``.  Confidence is a robust
    peak-to-background ratio and is not a probability.
    """

    audio = np.asarray(audio, dtype=np.float64)
    positions = np.asarray(mic_pos, dtype=np.float64)
    if audio.ndim != 2:
        raise ValueError("SRP-PHAT 입력은 samples x microphones 배열이어야 합니다.")
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("SRP-PHAT 마이크 위치는 N x 2 배열이어야 합니다.")
    if audio.shape[1] != positions.shape[0] or audio.shape[1] < 2:
        raise ValueError("SRP-PHAT 오디오 채널과 마이크 위치 개수가 다릅니다.")
    if fs <= 0.0 or tone_half_band_hz <= 0.0 or angle_step_deg <= 0.0:
        raise ValueError("SRP-PHAT 주파수/각도 설정이 잘못되었습니다.")

    if channel_offsets_sec is None:
        offsets = np.zeros(audio.shape[1], dtype=np.float64)
    else:
        offsets = np.asarray(channel_offsets_sec, dtype=np.float64)
        if offsets.shape != (audio.shape[1],) or not np.all(np.isfinite(offsets)):
            raise ValueError("SRP-PHAT 채널 보정값 형식이 잘못되었습니다.")

    audio = audio - np.mean(audio, axis=0, keepdims=True)
    window = np.hanning(audio.shape[0])
    nfft = 1
    while nfft < audio.shape[0]:
        nfft *= 2
    spectrum = np.fft.rfft(audio * window[:, None], n=nfft, axis=0)
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / float(fs))

    tone_mask = np.zeros(frequencies.shape, dtype=bool)
    for tone in tones_hz:
        tone_mask |= np.abs(frequencies - float(tone)) <= tone_half_band_hz
    if not np.any(tone_mask):
        raise ValueError("SRP-PHAT 비콘 tone 대역이 FFT에 없습니다.")

    # Reject silence before PHAT normalization.  500 Hz is intentionally not
    # part of the configured DOA mask, so its attenuation cannot invalidate a
    # valid 900/1050 Hz DOA after Sipeed has already locked the packet.
    tone_amplitude = 2.0 * np.max(
        np.abs(spectrum[tone_mask, :]),
        axis=0,
    ) / max(float(np.sum(window)), 1e-12)
    tone_dbfs = 20.0 * math.log10(max(float(np.max(tone_amplitude)), 1e-12))
    if tone_dbfs < DOA_MIN_SIGNAL_DBFS:
        raise ValueError(
            "SRP-PHAT 비콘 tone 신호가 너무 약합니다. "
            f"signal={tone_dbfs:.1f}dBFS"
        )

    frequencies = frequencies[tone_mask]
    spectrum = spectrum[tone_mask, :]
    if frequencies.size < 2:
        raise ValueError("SRP-PHAT에 필요한 tone bin이 부족합니다.")

    angles = np.arange(-180.0, 180.0, float(angle_step_deg))
    directions = np.column_stack((
        np.cos(np.deg2rad(angles)),
        np.sin(np.deg2rad(angles)),
    ))
    scores = np.zeros(angles.shape, dtype=np.float64)
    frequency_count = float(frequencies.size)

    for first, second in combinations(range(audio.shape[1]), 2):
        # X_second * conj(X_first) has phase
        # -w*(delay_second-delay_first).  The geometric steering delay below
        # uses the same second-minus-first convention.
        cross = spectrum[:, second] * np.conj(spectrum[:, first])
        offset_difference = offsets[second] - offsets[first]
        if abs(float(offset_difference)) > 1e-15:
            cross *= np.exp(
                1j * 2.0 * np.pi * frequencies * float(offset_difference)
            )
        phat = cross / np.maximum(np.abs(cross), 1e-15)

        baseline = positions[second] - positions[first]
        delays = (directions @ baseline) / float(sound_speed)
        steering = np.exp(
            1j * 2.0 * np.pi * frequencies[:, None] * delays[None, :]
        )
        pair_score = np.real(phat[:, None] * steering).sum(axis=0)
        scores += pair_score / frequency_count

    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])
    background = np.delete(scores, best_index)
    median = float(np.median(background))

    # Go2에서 사용하던 peak-to-background 형태의 confidence입니다.
    # 좁은 대역 비콘은 SRP score가 넓은 봉우리로 형성되므로 MAD 기반
    # z-score보다 이 비율이 실제 peak 분리를 더 안정적으로 나타냅니다.
    confidence = best_score / max(abs(median), 1e-9)
    if confidence < float(DOA_SRP_PHAT_MIN_CONFIDENCE):
        raise ValueError(
            "SRP-PHAT peak confidence가 낮습니다. "
            f"confidence={confidence:.2f}, "
            f"minimum={DOA_SRP_PHAT_MIN_CONFIDENCE:.2f}"
        )

    angle = float(angles[best_index]) * float(sign)
    angle = (angle + 180.0) % 360.0 - 180.0
    return angle, float(confidence), scores


def _tone_level_dbfs(samples, fs, tone_hz, *, nfft=4096):
    """Return the median raw-microphone level near one beacon tone."""

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] < 64:
        raise ValueError("DOA tone level 입력 형식이 잘못되었습니다.")

    window = np.hanning(samples.shape[0])
    spectrum = np.abs(
        np.fft.rfft(samples * window[:, None], n=int(nfft), axis=0)
    )
    frequencies = np.fft.rfftfreq(int(nfft), d=1.0 / float(fs))
    mask = np.abs(frequencies - float(tone_hz)) <= (
        DOA_SRP_PHAT_TONE_HALF_BAND_HZ
    )
    if not np.any(mask):
        raise ValueError("DOA tone 대역이 FFT에 없습니다.")

    amplitude = (
        2.0 * np.max(spectrum[mask], axis=0)
        / max(float(np.sum(window)), 1e-12)
    )
    return 20.0 * math.log10(
        max(float(np.median(amplitude)), 1e-12)
    )


def _tone_snr_db(samples, fs, tone_hz, *, nfft=4096):
    """Estimate active-tone SNR against nearby non-beacon FFT bins."""

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] < 64:
        raise ValueError("DOA tone SNR 입력 형식이 잘못되었습니다.")
    window = np.hanning(samples.shape[0])
    spectrum = np.abs(
        np.fft.rfft(samples * window[:, None], n=int(nfft), axis=0)
    )
    frequencies = np.fft.rfftfreq(int(nfft), d=1.0 / float(fs))
    signal_mask = np.abs(frequencies - float(tone_hz)) <= float(
        DOA_SRP_PHAT_TONE_HALF_BAND_HZ
    )
    noise_mask = (
        (frequencies >= DOA_BAND_LOW)
        & (frequencies <= DOA_BAND_HIGH)
        & ~signal_mask
    )
    for beacon_tone_hz in _BEACON_TONES_HZ:
        noise_mask &= np.abs(frequencies - beacon_tone_hz) > 40.0
    if not np.any(signal_mask) or np.count_nonzero(noise_mask) < 8:
        raise ValueError("DOA tone SNR 분석 bin이 부족합니다.")

    signal = np.max(spectrum[signal_mask], axis=0)
    noise = np.median(spectrum[noise_mask], axis=0)
    channel_snr = 20.0 * np.log10(
        np.maximum(signal, 1e-12) / np.maximum(noise, 1e-12)
    )
    return float(np.median(channel_snr))


def select_active_doa_tone_window(samples, fs):
    """Find a clean X(900 Hz) or Y(1050 Hz) ReSpeaker window.

    SIPEED owns complete XYXY packet authentication.  This helper only picks
    the active symbol for ReSpeaker spatial processing, so it deliberately
    does not demand 500 Hz or 650 Hz again.
    """

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError("DOA 오디오 배열 형식이 잘못되었습니다.")

    window_size = min(
        max(int(round(DOA_ACTIVE_TONE_WINDOW_SEC * float(fs))), 256),
        samples.shape[0],
    )
    if window_size < 256:
        raise ValueError("활성 DOA tone을 고르기 위한 오디오가 짧습니다.")

    best = None
    last_levels = None
    step = max(window_size // 2, 1)
    for start in range(0, samples.shape[0] - window_size + 1, step):
        frame = samples[start:start + window_size]
        levels = {
            symbol: _tone_level_dbfs(frame, fs, tone)
            for symbol, tone in DOA_ACTIVE_TONES_HZ.items()
        }
        last_levels = levels
        symbol = max(levels, key=levels.get)
        other_symbol = "Y" if symbol == "X" else "X"
        level = float(levels[symbol])
        margin = level - float(levels[other_symbol])
        if (
            level < float(DOA_ACTIVE_TONE_MIN_DBFS)
            or margin < float(DOA_ACTIVE_TONE_MARGIN_DB)
        ):
            continue

        candidate = (level, margin, symbol, start, window_size, levels)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    if best is None:
        if last_levels is None:
            raise ValueError("활성 DOA tone 분석 창을 만들 수 없습니다.")
        raise ValueError(
            "활성 X/Y DOA tone을 분리하지 못했습니다. "
            f"900={last_levels['X']:.1f}dBFS, "
            f"1050={last_levels['Y']:.1f}dBFS"
        )

    level, margin, symbol, start, window_size, levels = best
    return (
        str(symbol),
        float(DOA_ACTIVE_TONES_HZ[symbol]),
        float(level),
        float(margin),
        slice(int(start), int(start + window_size)),
        levels,
    )


def music_tone_spectrum(
    audio,
    fs,
    mic_pos,
    tone_hz,
    *,
    angle_step_deg=DOA_SRP_PHAT_ANGLE_STEP_DEG,
    source_count=DOA_MUSIC_SOURCE_COUNT,
    snapshot_sec=DOA_MUSIC_SNAPSHOT_SEC,
    diagonal_loading=DOA_MUSIC_DIAGONAL_LOADING,
    sound_speed=SOUND_SPEED,
    channel_offsets_sec=None,
):
    """Return the narrow-band MUSIC spectrum for exactly one active tone.

    Each short snapshot is complex-demodulated at ``tone_hz`` before building
    the four-channel covariance.  The per-channel calibration offsets are
    applied in the same phase convention as ``srp_phat_doa``.
    """

    audio = np.asarray(audio, dtype=np.float64)
    positions = np.asarray(mic_pos, dtype=np.float64)
    if audio.ndim != 2 or audio.shape[0] < 256:
        raise ValueError("MUSIC 입력은 충분히 긴 samples x microphones 배열이어야 합니다.")
    if positions.shape != (audio.shape[1], 2):
        raise ValueError("MUSIC 오디오 채널과 마이크 위치 개수가 다릅니다.")
    if not (1 <= int(source_count) < audio.shape[1]):
        raise ValueError("MUSIC source count 설정이 잘못되었습니다.")

    if channel_offsets_sec is None:
        offsets = np.zeros(audio.shape[1], dtype=np.float64)
    else:
        offsets = np.asarray(channel_offsets_sec, dtype=np.float64)
        if offsets.shape != (audio.shape[1],) or not np.all(np.isfinite(offsets)):
            raise ValueError("MUSIC 채널 보정값 형식이 잘못되었습니다.")

    snapshot_size = min(
        max(int(round(float(snapshot_sec) * float(fs))), 128),
        audio.shape[0],
    )
    if snapshot_size < 128:
        raise ValueError("MUSIC covariance snapshot이 너무 짧습니다.")

    window = np.hanning(snapshot_size)
    local_time = np.arange(snapshot_size, dtype=np.float64) / float(fs)
    mixer = np.exp(-1j * 2.0 * np.pi * float(tone_hz) * local_time)
    snapshots = []
    step = max(snapshot_size // 2, 1)
    phase_calibration = np.exp(
        1j * 2.0 * np.pi * float(tone_hz) * offsets
    )
    for start in range(0, audio.shape[0] - snapshot_size + 1, step):
        frame = audio[start:start + snapshot_size]
        phasor = np.sum(
            (frame - np.mean(frame, axis=0, keepdims=True))
            * (window * mixer)[:, None],
            axis=0,
        ) / max(float(np.sum(window)), 1e-12)
        snapshots.append(phasor * phase_calibration)

    if len(snapshots) < 2:
        raise ValueError("MUSIC covariance snapshot이 부족합니다.")

    snapshot_matrix = np.asarray(snapshots, dtype=np.complex128).T
    covariance = (
        snapshot_matrix @ snapshot_matrix.conj().T
    ) / float(snapshot_matrix.shape[1])
    trace = float(np.real(np.trace(covariance)))
    if trace <= 1e-15:
        raise ValueError("MUSIC 활성 tone 에너지가 없습니다.")
    covariance += (
        float(diagonal_loading) * trace / float(audio.shape[1])
    ) * np.eye(audio.shape[1], dtype=np.complex128)

    _eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    noise_space = eigenvectors[:, :audio.shape[1] - int(source_count)]

    angles = np.arange(-180.0, 180.0, float(angle_step_deg))
    directions = np.column_stack((
        np.cos(np.deg2rad(angles)),
        np.sin(np.deg2rad(angles)),
    ))
    delays = (directions @ positions.T) / float(sound_speed)
    steering = np.exp(-1j * 2.0 * np.pi * float(tone_hz) * delays)
    denominator = np.sum(
        np.abs(noise_space.conj().T @ steering.T) ** 2,
        axis=0,
    )
    spectrum = 1.0 / np.maximum(denominator, 1e-15)
    return angles, np.asarray(spectrum, dtype=np.float64)


def extract_circular_music_peaks(
    angles,
    spectrum,
    *,
    count=DOA_MUSIC_PEAK_COUNT,
    min_separation_deg=DOA_MUSIC_PEAK_SEPARATION_DEG,
):
    """Return spatial-spectrum local maxima with circular peak suppression."""

    angles = np.asarray(angles, dtype=np.float64)
    spectrum = np.asarray(spectrum, dtype=np.float64)
    if angles.ndim != 1 or spectrum.shape != angles.shape or angles.size < 3:
        raise ValueError("MUSIC peak 입력 형식이 잘못되었습니다.")
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("MUSIC spectrum에 유한하지 않은 값이 있습니다.")

    local_maxima = [
        index
        for index in range(angles.size)
        if spectrum[index] >= spectrum[(index - 1) % angles.size]
        and spectrum[index] >= spectrum[(index + 1) % angles.size]
    ]
    if not local_maxima:
        local_maxima = [int(np.argmax(spectrum))]

    selected = []
    for index in sorted(local_maxima, key=lambda item: spectrum[item], reverse=True):
        if all(
            abs(circular_angle_difference_deg(angles[index], angles[other]))
            >= float(min_separation_deg)
            for other in selected
        ):
            selected.append(index)
        if len(selected) >= int(count):
            break

    return [
        (float(angles[index]), float(spectrum[index]))
        for index in selected
    ]


def _normalized_candidate_scores(values):
    """Map a score vector to a robust 0..1 scale for score fusion."""

    values = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(values, (10.0, 90.0))
    if high - low <= 1e-12:
        return np.full(values.shape, 0.5, dtype=np.float64)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def collect_tone_pair_taus(audio, fs, tone_hz, mic_pos, channel_offsets_sec):
    """Measure corrected narrow-band TDOAs for every available microphone pair."""

    audio = np.asarray(audio, dtype=np.float64)
    positions = np.asarray(mic_pos, dtype=np.float64)
    raw_pair_taus = []
    errors = []
    for first, second in combinations(range(audio.shape[1]), 2):
        max_tau = np.linalg.norm(positions[second] - positions[first]) / SOUND_SPEED
        try:
            tau = tone_phase_tdoa(
                audio[:, second],
                audio[:, first],
                fs,
                tone_hz,
                max_tau=max_tau,
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        raw_pair_taus.append((first, second, tau))

    corrected = apply_channel_phase_offsets(raw_pair_taus, channel_offsets_sec)
    if len(corrected) < int(DOA_FUSION_MIN_TDOA_INLIERS):
        detail = errors[0] if errors else "유효한 pair가 없습니다."
        raise ValueError(
            "활성 tone TDOA 마이크 쌍이 부족합니다: " + detail
        )
    return corrected


def score_tdoa_candidate(
    candidate_angle_deg,
    pair_taus,
    mic_pos,
    *,
    sign=1.0,
    tolerance_m=DOA_PAIR_INLIER_TOLERANCE_M,
):
    """Return six-pair TDOA inlier support for a physical MUSIC candidate."""

    positions = np.asarray(mic_pos, dtype=np.float64)
    direction = np.array((
        math.cos(math.radians(float(candidate_angle_deg))),
        math.sin(math.radians(float(candidate_angle_deg))),
    ))
    residuals = []
    for first, second, tau in pair_taus:
        expected_m = float(np.dot(positions[second] - positions[first], direction))
        observed_m = SOUND_SPEED * float(tau) * float(sign)
        residuals.append(abs(expected_m - observed_m))

    residuals = np.asarray(residuals, dtype=np.float64)
    inliers = residuals <= float(tolerance_m)
    inlier_count = int(np.count_nonzero(inliers))
    median_residual_m = float(np.median(residuals[inliers])) if inlier_count else math.inf
    support = (
        0.60 * (float(inlier_count) / float(len(residuals)))
        + 0.40 * max(0.0, 1.0 - median_residual_m / float(tolerance_m))
        if inlier_count
        else 0.0
    )
    return float(support), inlier_count, median_residual_m, residuals


# ============================================================
# Narrow-band TDOA
# ============================================================

def phase_slope_tdoa(
    signal,
    reference,
    fs,
    f_low,
    f_high,
    max_tau=None,
    nperseg=2048,
    nfft=16384,
):
    """
    좁은 대역 비콘 신호의 TDOA를 주파수별 위상 기울기로 계산합니다.

    비콘은 500~1050 Hz의 좁은 대역 신호이고 ReSpeaker 마이크 간
    최대 지연은 약 2~4 sample 수준입니다. 이 경우 정수 sample peak만
    찾는 GCC-PHAT은 지연을 0 sample로 양자화하기 쉽습니다.

    교차 스펙트럼의 위상은

        phase(f) = -2π f τ

    관계를 가지므로, 주파수에 대한 위상 기울기에서 τ를 구합니다.
    여러 프레임의 교차 스펙트럼을 평균하여 배경 잡음의 영향을 줄입니다.
    """

    signal = np.asarray(signal, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if signal.shape != reference.shape:
        raise ValueError("TDOA 입력 신호 길이가 다릅니다.")

    if signal.ndim != 1:
        raise ValueError("TDOA 입력은 1차원 신호여야 합니다.")

    nperseg = min(
        int(nperseg),
        signal.size,
    )

    if nperseg < 256:
        raise ValueError("TDOA 계산에 필요한 오디오 길이가 짧습니다.")

    nfft = max(
        int(nfft),
        nperseg,
    )

    window = np.hanning(nperseg)
    step = max(
        nperseg // 2,
        1,
    )

    cross_sum = None
    power_signal = None
    power_reference = None
    frame_count = 0

    for start in range(
        0,
        signal.size - nperseg + 1,
        step,
    ):

        signal_frame = (
            signal[start:start + nperseg]
            - np.mean(signal[start:start + nperseg])
        )

        reference_frame = (
            reference[start:start + nperseg]
            - np.mean(reference[start:start + nperseg])
        )

        signal_fft = np.fft.rfft(
            signal_frame * window,
            n=nfft,
        )

        reference_fft = np.fft.rfft(
            reference_frame * window,
            n=nfft,
        )

        cross = (
            signal_fft
            * np.conj(reference_fft)
        )

        if cross_sum is None:
            cross_sum = np.zeros_like(cross)
            power_signal = np.zeros(cross.size)
            power_reference = np.zeros(cross.size)

        cross_sum += cross
        power_signal += np.abs(signal_fft) ** 2
        power_reference += np.abs(reference_fft) ** 2
        frame_count += 1

    if frame_count == 0:
        raise ValueError("TDOA 프레임을 만들 수 없습니다.")

    frequencies = np.fft.rfftfreq(
        nfft,
        d=1.0 / fs,
    )

    in_band = (
        (frequencies >= f_low)
        & (frequencies <= f_high)
    )

    if not np.any(in_band):
        raise ValueError("TDOA 분석 대역에 FFT bin이 없습니다.")

    magnitude = np.sqrt(
        power_signal
        * power_reference
    )

    coherence = np.abs(cross_sum) / np.maximum(
        magnitude,
        1e-15,
    )

    band_magnitude = magnitude[in_band]
    maximum_magnitude = np.max(band_magnitude)

    if maximum_magnitude <= 1e-15:
        raise ValueError("TDOA 분석 대역의 신호가 너무 약합니다.")

    # 강한 상관 성분만 사용합니다. 이렇게 해야 무음/잡음 구간의
    # 임의 위상이 TDOA 회귀에 섞이지 않습니다.
    valid = (
        in_band
        & (coherence >= 0.35)
        & (magnitude >= maximum_magnitude * 0.08)
    )

    # 신호가 한 개의 tone만 남은 경우에도 가장 강한 상관 bin을
    # 사용할 수 있도록 완화된 fallback을 둡니다.
    if np.count_nonzero(valid) < 2:
        valid = (
            in_band
            & (coherence >= 0.15)
            & (magnitude >= maximum_magnitude * 0.02)
        )

    if not np.any(valid):
        raise ValueError("신뢰할 수 있는 교차 스펙트럼 bin이 없습니다.")

    selected_frequencies = frequencies[valid]
    selected_phase = np.unwrap(
        np.angle(cross_sum[valid])
    )

    weights = (
        magnitude[valid]
        * np.clip(coherence[valid], 0.0, 1.0)
    )

    if selected_frequencies.size < 2:
        tau = (
            -selected_phase[0]
            / (2.0 * np.pi * selected_frequencies[0])
        )

    else:
        frequency_mean = np.average(
            selected_frequencies,
            weights=weights,
        )

        phase_mean = np.average(
            selected_phase,
            weights=weights,
        )

        denominator = np.sum(
            weights
            * (selected_frequencies - frequency_mean) ** 2
        )

        if denominator <= 1e-12:
            raise ValueError("TDOA 주파수 분산이 부족합니다.")

        slope = np.sum(
            weights
            * (selected_frequencies - frequency_mean)
            * (selected_phase - phase_mean)
        ) / denominator

        tau = (
            -slope
            / (2.0 * np.pi)
        )

    if max_tau is not None and abs(tau) > max_tau * 1.25:
        raise ValueError("TDOA가 마이크 간 물리 한계를 초과했습니다.")

    return float(tau)


def _select_active_tone_window(
    selected,
    fs,
    minimum_tone_dbfs=DOA_REFERENCE_TONE_MIN_DBFS,
):
    """Find a clean window containing a configured DOA reference tone.

    The Sipeed packet detector has already authenticated the complete XYXY
    packet before the ROS DOA node calls this estimator. Requiring the weaker
    companion tone again in every short ReSpeaker block loses valid distant
    samples. The optional pair check remains available for standalone tests.
    """

    selected = np.asarray(selected, dtype=np.float64)
    nperseg = min(
        max(int(round(DOA_REFERENCE_WINDOW_SEC * fs)), 256),
        selected.shape[0],
    )

    if nperseg < 256:
        raise ValueError("DOA 기준 tone을 고르기 위한 오디오가 짧습니다.")

    nfft = 4096
    window = np.hanning(nperseg)
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / fs)
    masks = {
        target: np.abs(frequencies - target) <= 25.0
        for target in _BEACON_TONES_HZ
    }
    best = None
    reference_tones = {float(tone) for tone in DOA_REFERENCE_TONES_HZ}

    for start in range(
        0,
        selected.shape[0] - nperseg + 1,
        max(nperseg // 2, 1),
    ):
        frame = selected[start:start + nperseg]
        spectrum = np.abs(
            np.fft.rfft(frame * window[:, None], n=nfft, axis=0)
        )
        frame_scores = {}

        for target in _BEACON_TONES_HZ:
            amplitude = (
                2.0
                * np.max(spectrum[masks[target]], axis=0)
                / max(np.sum(window), 1e-12)
            )
            frame_scores[target] = 20.0 * math.log10(
                max(float(np.median(amplitude)), 1e-12)
            )

        allowed_tones = reference_tones

        if DOA_REQUIRE_SYMBOL_PAIR:
            x_score = min(frame_scores[500.0], frame_scores[900.0])
            y_score = min(frame_scores[650.0], frame_scores[1050.0])
            allowed_tones = set()

            if (
                x_score >= PATTERN_MIN_COMPONENT_DBFS
                and x_score - y_score >= PATTERN_PAIR_MARGIN_DB
            ):
                allowed_tones.update((500.0, 900.0))

            if (
                y_score >= PATTERN_MIN_COMPONENT_DBFS
                and y_score - x_score >= PATTERN_PAIR_MARGIN_DB
            ):
                allowed_tones.update((650.0, 1050.0))

        usable_tones = [
            (tone, frame_scores[tone])
            for tone in reference_tones
            if tone in allowed_tones
            and frame_scores[tone] >= float(minimum_tone_dbfs)
        ]

        if not usable_tones:
            continue

        tone, tone_strength = max(
            usable_tones,
            key=lambda item: item[1],
        )
        if best is None or tone_strength > best[0]:
            best = (tone_strength, tone, start, nperseg)

    if best is None:
        raise ValueError(
            "충분히 강한 DOA 기준 tone을 찾지 못했습니다. "
            "현재 slot/guard 또는 먼 거리 신호는 DOA에서 제외합니다."
        )

    strength, tone, start, window_size = best
    return tone, strength, slice(start, start + window_size)


def _select_active_tone(selected, fs):
    """Return the reference tone and strength for compatibility helpers."""

    tone, strength, _window = _select_active_tone_window(selected, fs)
    return tone, strength


def classify_doa_sequence_frame(selected, fs=DOA_SAMPLE_RATE):
    """Classify one short ReSpeaker frame for the DOA-only XYX gate.

    X intentionally means a strong 900 Hz component only.  The 500 Hz
    companion is attenuated too strongly at this microphone to be a useful
    gate. After Sipeed has LOCKED the full XYXY packet, Y is used only as a
    coarse separator before the next X, so one weak Y component is enough.
    """

    selected = np.asarray(selected, dtype=np.float64)
    if selected.ndim != 2 or selected.shape[0] < 64:
        raise ValueError("DOA sequence frame shape이 잘못되었습니다.")

    window = np.hanning(selected.shape[0])
    nfft = 4096
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(selected * window[:, None], n=nfft, axis=0))

    scores = {}
    for tone in _BEACON_TONES_HZ:
        mask = np.abs(frequencies - tone) <= 25.0
        amplitude = (
            2.0 * np.max(spectrum[mask], axis=0)
            / max(np.sum(window), 1e-12)
        )
        scores[tone] = 20.0 * math.log10(
            max(float(np.median(amplitude)), 1e-12)
        )

    x_score = scores[900.0]
    # X는 두 Y 성분이 동시에 강할 때만 방해받아야 합니다. Y 존재 확인은
    # 한 성분만으로 충분하지만, 그 값을 X 판정에 재사용하면 X를 놓칩니다.
    x_opposite_score = min(scores[650.0], scores[1050.0])
    y_presence_score = max(scores[650.0], scores[1050.0])
    if (
        x_score >= DOA_SEQUENCE_X_900_MIN_DBFS
        and x_score - x_opposite_score
        >= float(DOA_SEQUENCE_X_SYMBOL_MARGIN_DB)
    ):
        return "X", scores

    if (
        y_presence_score >= DOA_SEQUENCE_Y_PRESENCE_MIN_DBFS
    ):
        return "Y", scores

    return None, scores


def has_doa_y_presence(scores):
    """Return whether either weak ReSpeaker Y component is present."""

    return max(scores[650.0], scores[1050.0]) >= (
        DOA_SEQUENCE_Y_PRESENCE_MIN_DBFS
    )


def circular_angle_difference_deg(first_deg, second_deg):
    """Return the shortest signed angular difference (first - second)."""

    return (float(first_deg) - float(second_deg) + 180.0) % 360.0 - 180.0


def circular_angle_mean_deg(angles):
    """Return a mean that remains correct across the -180°/180° boundary."""

    radians = np.deg2rad(np.asarray(angles, dtype=np.float64))
    return math.degrees(math.atan2(
        float(np.mean(np.sin(radians))),
        float(np.mean(np.cos(radians))),
    ))


def tone_phase_tdoa(
    signal,
    reference,
    fs,
    frequency,
    max_tau=None,
    nperseg=1024,
    nfft=4096,
):
    """한 개의 비콘 tone 위상으로 fractional-sample TDOA를 계산합니다.

    실제 XYXY 비콘은 250 ms 단위로 tone이 바뀌므로, 450~1150 Hz 전체를
    한 번에 회귀하면 서로 다른 tone/guard가 섞일 수 있습니다. 활성 tone의
    한 주파수에서만 cross-spectrum을 평균하면 이 문제를 피할 수 있습니다.
    """

    signal = np.asarray(signal, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if signal.shape != reference.shape or signal.ndim != 1:
        raise ValueError("tone TDOA 입력 신호가 잘못되었습니다.")

    nperseg = min(int(nperseg), signal.size)
    if nperseg < 256:
        raise ValueError("tone TDOA 계산에 필요한 오디오가 짧습니다.")

    target_bin = int(round(float(frequency) * nfft / fs))
    window = np.hanning(nperseg)
    cross_sum = 0.0j
    magnitude_sum = 0.0
    frame_count = 0
    step = max(nperseg // 2, 1)

    for start in range(0, signal.size - nperseg + 1, step):
        signal_frame = signal[start:start + nperseg]
        reference_frame = reference[start:start + nperseg]
        signal_fft = np.fft.rfft(
            (signal_frame - np.mean(signal_frame)) * window,
            n=nfft,
        )
        reference_fft = np.fft.rfft(
            (reference_frame - np.mean(reference_frame)) * window,
            n=nfft,
        )

        cross = signal_fft[target_bin] * np.conj(reference_fft[target_bin])
        magnitude = abs(cross)

        cross_sum += cross
        magnitude_sum += magnitude
        frame_count += 1

    if frame_count == 0 or magnitude_sum <= 1e-15:
        raise ValueError("활성 tone의 교차 스펙트럼이 없습니다.")

    coherence = abs(cross_sum) / magnitude_sum
    if coherence < 0.45:
        raise ValueError(
            f"활성 tone 위상 일관성이 낮습니다: coherence={coherence:.2f}"
        )

    tau = -np.angle(cross_sum) / (2.0 * np.pi * float(frequency))

    if max_tau is not None and abs(tau) > max_tau * 1.25:
        raise ValueError("활성 tone TDOA가 마이크 간 물리 한계를 초과했습니다.")

    return float(tau)


def solve_direction(
    taus,
    positions,
    sound_speed=SOUND_SPEED,
    sign=DOA_SIGN,
    max_residual=None,
):
    """
    기준 마이크와 나머지 마이크의 TDOA로 방향각을 계산합니다.

    이 함수는 하드웨어 오디오 입력과 분리되어 있어 합성 신호 검증에도
    동일한 방향 계산식을 사용할 수 있습니다.
    """

    positions = np.asarray(
        positions,
        dtype=np.float64,
    )

    taus = np.asarray(
        taus,
        dtype=np.float64,
    )

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("마이크 위치는 N x 2 배열이어야 합니다.")

    if taus.size != positions.shape[0] - 1:
        raise ValueError("TDOA 개수와 마이크 위치 개수가 맞지 않습니다.")

    matrix_a = positions[1:] - positions[0]
    vector_b = sound_speed * taus * sign

    direction, _, _, _ = np.linalg.lstsq(
        matrix_a,
        vector_b,
        rcond=None,
    )

    norm = np.linalg.norm(direction)

    if norm < 1e-9:
        raise RuntimeError("DOA 방향벡터 계산 실패")

    residual = np.linalg.norm(
        matrix_a @ direction - vector_b
    ) / max(np.linalg.norm(vector_b), 1e-9)

    if (
        max_residual is not None
        and residual > max_residual
    ):
        raise ValueError(
            "마이크별 TDOA가 하나의 방향으로 일치하지 않습니다. "
            f"residual={residual:.2f}"
        )

    direction = direction / norm

    angle = math.degrees(
        math.atan2(
            direction[1],
            direction[0],
        )
    )

    return (
        (angle + 180)
        % 360
        - 180
    )


def solve_direction_from_pair_taus(
    pair_taus,
    positions,
    sound_speed=SOUND_SPEED,
    sign=DOA_SIGN,
    inlier_tolerance_m=DOA_PAIR_INLIER_TOLERANCE_M,
    min_inlier_pairs=DOA_MIN_INLIER_PAIRS,
):
    """Estimate a direction from all microphone-pair TDOAs robustly.

    ``pair_taus`` contains ``(reference_index, signal_index, tau_sec)``.
    A direct path reaching every microphone gives six mutually consistent
    pair equations.  If an obstacle or reflection corrupts one microphone,
    its three equations become outliers but the remaining three microphones
    still contribute a consistent triangle.  The old one-reference scheme
    could not recover in that case.
    """

    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("마이크 위치는 N x 2 배열이어야 합니다.")

    baselines = []
    distances = []
    for reference_index, signal_index, tau in pair_taus:
        if not (
            0 <= reference_index < len(positions)
            and 0 <= signal_index < len(positions)
            and reference_index != signal_index
        ):
            raise ValueError("TDOA 마이크 인덱스가 잘못되었습니다.")
        if not math.isfinite(float(tau)):
            continue
        baselines.append(
            positions[signal_index] - positions[reference_index]
        )
        distances.append(float(sound_speed) * float(tau) * float(sign))

    if len(baselines) < min_inlier_pairs:
        raise ValueError("유효한 마이크 쌍 TDOA가 부족합니다.")

    matrix = np.asarray(baselines, dtype=np.float64)
    vector = np.asarray(distances, dtype=np.float64)
    tolerance = float(inlier_tolerance_m)
    if tolerance <= 0.0:
        raise ValueError("DOA pair inlier tolerance은 양수여야 합니다.")

    best_direction = None
    best_score = None
    best_attempt = None

    # 모든 비평행 쌍으로 방향 후보를 만든 뒤 단위 원 위로 정규화합니다.
    # 이전에는 이 벡터의 크기가 0.5~1.5인지 먼저 검사해, 위상 지연의
    # 공통 스케일 오차가 있을 때 모든 후보를 버렸습니다. 크기는 방향의
    # 신뢰도 조건이 아니므로 버리고, 여섯 쌍의 잔차로만 후보를 평가합니다.
    for first, second in combinations(range(len(matrix)), 2):
        candidate_matrix = matrix[[first, second]]
        if np.linalg.matrix_rank(candidate_matrix) < 2:
            continue

        direction, _, _, _ = np.linalg.lstsq(
            candidate_matrix,
            vector[[first, second]],
            rcond=None,
        )
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            continue
        direction = direction / norm

        residuals = np.abs(matrix @ direction - vector)
        inliers = residuals <= tolerance
        inlier_count = int(np.count_nonzero(inliers))
        inlier_median = (
            float(np.median(residuals[inliers]))
            if inlier_count
            else math.inf
        )
        score = (inlier_count, -inlier_median)
        if best_attempt is None or score > best_attempt[0]:
            best_attempt = (score, residuals, inliers)

        if inlier_count < min_inlier_pairs:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_direction = direction

    if best_direction is None:
        if best_attempt is None:
            raise ValueError(
                "마이크 쌍 TDOA가 하나의 방향으로 충분히 일치하지 않습니다."
            )
        _score, best_residuals, best_inliers = best_attempt
        raise ValueError(
            "마이크 쌍 TDOA가 하나의 방향으로 충분히 일치하지 않습니다. "
            f"best={int(np.count_nonzero(best_inliers))}/{len(matrix)}, "
            f"median={float(np.median(best_residuals)) * 1000.0:.1f}mm, "
            f"limit={tolerance * 1000.0:.1f}mm"
        )

    angle = math.degrees(math.atan2(best_direction[1], best_direction[0]))
    return (angle + 180.0) % 360.0 - 180.0


def _stable_cluster_angle(measurements):
    """Return a circular inlier-cluster angle, or ``None`` when unstable."""

    if len(measurements) < DOA_MIN_VALID_MEASUREMENTS:
        return None

    # 현재 운용은 900 Hz DOA를 두 번만 측정합니다. 평균 중심에서 각각
    # 18° 이내라는 기존 조건은 두 값의 차이가 36°여도 통과시킵니다.
    # 두 측정값일 때는 요구사항대로 값끼리 직접 비교해야 합니다.
    if len(measurements) == 2:
        difference = abs(
            circular_angle_difference_deg(measurements[0], measurements[1])
        )
        if difference > float(DOA_PAIR_MAX_ANGLE_DIFF_DEG):
            return None
        return circular_angle_mean_deg(measurements)

    radians = np.deg2rad(measurements)
    center = math.atan2(
        float(np.mean(np.sin(radians))),
        float(np.mean(np.cos(radians))),
    )
    errors = np.abs(
        np.arctan2(
            np.sin(radians - center),
            np.cos(radians - center),
        )
    )
    inliers = errors <= math.radians(DOA_OUTLIER_TOLERANCE_DEG)

    if np.count_nonzero(inliers) < DOA_MIN_VALID_MEASUREMENTS:
        return None

    inlier_radians = radians[inliers]
    angle_radians = math.atan2(
        float(np.mean(np.sin(inlier_radians))),
        float(np.mean(np.cos(inlier_radians))),
    )
    spread = np.max(np.abs(
        np.arctan2(
            np.sin(inlier_radians - angle_radians),
            np.cos(inlier_radians - angle_radians),
        )
    ))

    if spread > math.radians(DOA_MAX_ANGLE_SPREAD_DEG):
        return None

    return math.degrees(angle_radians)


# ============================================================
# DOA Solver
# ============================================================

class DOAEstimator:

    def __init__(self):
        self.audio = DOAAudioReader()
        self.last_diagnostics = {}
        self.channel_phase_offsets_sec, calibration = load_channel_phase_offsets()
        if DEBUG:
            if calibration:
                offsets_us = ", ".join(
                    f"{value * 1e6:+.1f}us"
                    for value in self.channel_phase_offsets_sec
                )
                print(f"[DOA] calibration loaded: {offsets_us}")
            else:
                print("[DOA] calibration not found: raw channel delays in use")


    def _estimate_selected(self, selected, gate=None, active_tone_hz=None):
        """Fuse MUSIC, SRP-PHAT and six-pair TDOA for one active symbol.

        SIPEED's complete XYXY lock remains the only packet authentication.
        ReSpeaker only chooses the currently strong X(900 Hz) or Y(1050 Hz)
        tone, then every spatial method evaluates that *same* narrow-band
        window.  This avoids mixing different packet slots in one DOA result.
        """

        del gate
        self.last_diagnostics = {}
        selected = np.asarray(selected, dtype=np.float64)
        if selected.ndim != 2:
            raise ValueError("DOA 오디오 배열 형식이 잘못되었습니다.")

        selected = bandpass(selected)
        channel_rms = np.sqrt(np.mean(selected ** 2, axis=0))
        signal_dbfs = 20.0 * math.log10(
            max(float(np.max(channel_rms)), 1e-12)
        )
        if DEBUG:
            print(f"[DOA] band_signal={signal_dbfs:.1f}dBFS")
        if signal_dbfs < DOA_MIN_SIGNAL_DBFS:
            raise ValueError(
                "비콘 대역 신호가 너무 약합니다. "
                f"signal={signal_dbfs:.1f}dBFS"
            )

        positions = np.asarray(MIC_POSITIONS_M, dtype=np.float64)
        if len(DOA_CHANNELS) != len(positions):
            raise RuntimeError(
                "DOA_CHANNELS 개수와 MIC_POSITIONS_M 개수가 같아야 합니다."
            )

        (
            symbol,
            detected_tone_hz,
            strength_dbfs,
            symbol_margin_db,
            tone_window,
            tone_levels,
        ) = select_active_doa_tone_window(selected, DOA_SAMPLE_RATE)
        if active_tone_hz is not None:
            forced_tone_hz = float(active_tone_hz)
            if not math.isclose(detected_tone_hz, forced_tone_hz, abs_tol=1e-6):
                raise ValueError(
                    "요청한 DOA symbol과 현재 ReSpeaker symbol이 다릅니다. "
                    f"active={symbol}({detected_tone_hz:.0f}Hz), "
                    f"requested={forced_tone_hz:.0f}Hz"
                )

        active = selected[tone_window]
        tone_snr_db = _tone_snr_db(active, DOA_SAMPLE_RATE, detected_tone_hz)
        if tone_snr_db < float(DOA_ACTIVE_TONE_MIN_SNR_DB):
            raise ValueError(
                "활성 DOA tone SNR이 낮습니다. "
                f"snr={tone_snr_db:.1f}dB, "
                f"minimum={DOA_ACTIVE_TONE_MIN_SNR_DB:.1f}dB"
            )
        if DEBUG:
            print(
                f"[DOA] active_symbol={symbol} tone={detected_tone_hz:.0f}Hz "
                f"strength={strength_dbfs:.1f}dBFS "
                f"margin={symbol_margin_db:.1f}dB SNR={tone_snr_db:.1f}dB "
                f"(900={tone_levels['X']:.1f}, 1050={tone_levels['Y']:.1f})"
            )

        music_angles, music_spectrum = music_tone_spectrum(
            active,
            DOA_SAMPLE_RATE,
            positions,
            detected_tone_hz,
            channel_offsets_sec=self.channel_phase_offsets_sec,
        )
        music_peaks = extract_circular_music_peaks(music_angles, music_spectrum)
        music_background = float(np.median(music_spectrum))
        music_peak_ratio = float(music_peaks[0][1]) / max(music_background, 1e-12)
        if music_peak_ratio < float(DOA_MUSIC_MIN_PEAK_RATIO):
            raise ValueError(
                "MUSIC peak 분리가 낮습니다. "
                f"ratio={music_peak_ratio:.2f}, "
                f"minimum={DOA_MUSIC_MIN_PEAK_RATIO:.2f}"
            )

        music_peak_margin_db = (
            10.0 * math.log10(
                max(float(music_peaks[0][1]), 1e-12)
                / max(float(music_peaks[1][1]), 1e-12)
            )
            if len(music_peaks) >= 2
            else 120.0
        )
        if music_peak_margin_db < float(DOA_MUSIC_MIN_TOP_PEAK_MARGIN_DB):
            raise ValueError(
                "MUSIC 1/2순위 peak 차이가 작습니다. "
                f"margin={music_peak_margin_db:.1f}dB, "
                f"minimum={DOA_MUSIC_MIN_TOP_PEAK_MARGIN_DB:.1f}dB"
            )

        srp_angle, srp_confidence, srp_scores = srp_phat_doa(
            active,
            DOA_SAMPLE_RATE,
            positions,
            tones_hz=(detected_tone_hz,),
            # Compare in physical microphone coordinates. DOA_SIGN is only
            # applied once after MUSIC/SRP/TDOA select the same candidate.
            sign=1.0,
            channel_offsets_sec=self.channel_phase_offsets_sec,
        )
        srp_angles = np.arange(
            -180.0,
            180.0,
            float(DOA_SRP_PHAT_ANGLE_STEP_DEG),
        )
        srp_normalized = _normalized_candidate_scores(srp_scores)
        pair_taus = collect_tone_pair_taus(
            active,
            DOA_SAMPLE_RATE,
            detected_tone_hz,
            positions,
            self.channel_phase_offsets_sec,
        )
        music_normalized = _normalized_candidate_scores(
            [peak[1] for peak in music_peaks]
        )
        candidates = []
        for (candidate_angle, music_value), music_score in zip(
            music_peaks,
            music_normalized,
        ):
            srp_index = int(np.argmin(np.abs(
                [
                    circular_angle_difference_deg(candidate_angle, angle)
                    for angle in srp_angles
                ]
            )))
            srp_score = float(srp_normalized[srp_index])
            srp_candidate_diff_deg = abs(
                circular_angle_difference_deg(candidate_angle, srp_angle)
            )
            if srp_candidate_diff_deg > float(
                DOA_FUSION_MAX_SRP_CANDIDATE_DIFF_DEG
            ):
                continue
            (
                tdoa_support,
                inlier_count,
                median_residual_m,
                _residuals,
            ) = score_tdoa_candidate(
                candidate_angle,
                pair_taus,
                positions,
                # MUSIC candidates live in the physical microphone coordinate.
                # DOA_SIGN is applied only once when publishing the final
                # ReSpeaker-relative angle below.
                sign=1.0,
            )
            if inlier_count < int(DOA_FUSION_MIN_TDOA_INLIERS):
                continue
            fused_score = (
                float(DOA_FUSION_MUSIC_WEIGHT) * float(music_score)
                + float(DOA_FUSION_SRP_WEIGHT) * srp_score
                + float(DOA_FUSION_TDOA_WEIGHT) * tdoa_support
            )
            candidates.append((
                fused_score,
                candidate_angle,
                float(music_value),
                srp_score,
                inlier_count,
                median_residual_m,
                srp_candidate_diff_deg,
            ))

        if not candidates:
            raise ValueError(
                "MUSIC 후보가 6-pair TDOA와 충분히 일치하지 않습니다."
            )
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        (
            fused_score,
            physical_angle,
            _music_value,
            srp_score,
            inlier_count,
            median_residual_m,
            srp_candidate_diff_deg,
        ) = candidates[0]
        if fused_score < float(DOA_FUSION_MIN_SCORE):
            raise ValueError(
                "MUSIC/SRP/TDOA 통합 신뢰도가 낮습니다. "
                f"score={fused_score:.2f}, "
                f"minimum={DOA_FUSION_MIN_SCORE:.2f}"
            )

        angle = (float(physical_angle) * float(DOA_SIGN) + 180.0) % 360.0 - 180.0
        self.last_diagnostics = {
            "active_symbol": str(symbol),
            "active_tone_hz": float(detected_tone_hz),
            "active_tone_strength_dbfs": float(strength_dbfs),
            "active_tone_snr_db": float(tone_snr_db),
            "music_peak_ratio": float(music_peak_ratio),
            "music_top_peak_margin_db": float(music_peak_margin_db),
            "srp_peak_angle_deg": float(srp_angle),
            "srp_confidence": float(srp_confidence),
            "srp_candidate_diff_deg": float(srp_candidate_diff_deg),
            "srp_candidate_score": float(srp_score),
            "tdoa_inlier_pairs": int(inlier_count),
            "tdoa_total_pairs": int(len(pair_taus)),
            "tdoa_median_residual_m": float(median_residual_m),
            "fused_score": float(fused_score),
            "angle_deg": float(angle),
        }
        if DEBUG:
            peak_text = ", ".join(
                f"{candidate_angle:+.0f}°"
                for candidate_angle, _value in music_peaks
            )
            print(
                f"[DOA] MUSIC peaks={peak_text}; "
                f"fused={angle:+.1f}° score={fused_score:.2f} "
                f"SRP_conf={srp_confidence:.2f} SRP_score={srp_score:.2f} "
                f"MUSIC_gap={music_peak_margin_db:.1f}dB "
                f"SNR={tone_snr_db:.1f}dB "
                f"TDOA={inlier_count}/{len(pair_taus)} "
                f"median={median_residual_m * 1000.0:.1f}mm"
            )
        return float(angle)

    def _estimate_once(self):
        """Measure one independent active-symbol fused DOA frame."""

        data = self.audio.read(DOA_BLOCK_SEC)
        return self._estimate_selected(data[:, DOA_CHANNELS])

    def _estimate_xyx_pair(self):
        """Accept two 900 Hz DOAs only after ReSpeaker observes X→Y→X.

        The two X windows must be separated by a verified Y window.  This
        prevents two unrelated 900 Hz fragments from being treated as a
        direction estimate while avoiding the weak 500 Hz component entirely.
        """

        deadline = time.monotonic() + DOA_ESTIMATE_TIMEOUT_SEC
        state = "WAIT_X1"
        x_frames = []
        y_frame_count = 0
        angles = []
        x_rejects = 0
        best_y_scores = None
        y_not_before = 0.0

        while time.monotonic() < deadline:
            data = self.audio.read(float(DOA_SEQUENCE_FRAME_SEC))
            selected = data[:, DOA_CHANNELS]
            symbol, scores = classify_doa_sequence_frame(
                selected, DOA_SAMPLE_RATE
            )

            if state in {"WAIT_X1", "WAIT_X2"}:
                if symbol == "X":
                    x_frames.append(selected)
                    x_frames = x_frames[-int(DOA_SEQUENCE_X_FRAMES_REQUIRED):]
                    if len(x_frames) < int(DOA_SEQUENCE_X_FRAMES_REQUIRED):
                        continue

                    try:
                        angle = self._estimate_selected(
                            np.vstack(x_frames),
                            active_tone_hz=900.0,
                        )
                    except ValueError as exc:
                        x_rejects += 1
                        if DEBUG and (
                            x_rejects == 1 or x_rejects % 4 == 0
                        ):
                            print(
                                "[DOA] XYX X(900Hz) DOA 제외: "
                                f"{exc}"
                            )
                        # Slide one frame at a time inside the same X tone.
                        continue

                    angles.append(angle)
                    if DEBUG:
                        print(
                            f"[DOA] XYX {len(angles)}/2 X(900Hz) = {angle:.1f}°"
                        )

                    if state == "WAIT_X1":
                        state = "WAIT_Y"
                        x_frames.clear()
                        y_frame_count = 0
                        y_not_before = (
                            time.monotonic()
                            + float(DOA_SEQUENCE_Y_MIN_DELAY_SEC)
                        )
                        continue

                    difference = abs(
                        circular_angle_difference_deg(angles[0], angles[1])
                    )
                    if difference <= float(DOA_PAIR_MAX_ANGLE_DIFF_DEG):
                        result = circular_angle_mean_deg(angles)
                        if DEBUG:
                            print(
                                "[DOA] XYX 두 X 각도 일치 = "
                                f"{result:.1f}° (diff={difference:.1f}°)"
                            )
                        return result

                    if DEBUG:
                        print(
                            "[DOA] XYX 두 X 각도 불일치 -> 재시작 "
                            f"(diff={difference:.1f}°)"
                        )
                    state = "WAIT_X1"
                    angles.clear()
                    x_frames.clear()
                    continue

                # Guard/잡음 사이에 끊긴 X frame을 이어 붙이지 않습니다.
                if symbol is None:
                    x_frames.clear()
                continue

            # WAIT_Y: 첫 번째 X DOA 뒤의 Y(650+1050)만 확인합니다.
            if (
                best_y_scores is None
                or max(scores[650.0], scores[1050.0])
                > max(best_y_scores[650.0], best_y_scores[1050.0])
            ):
                best_y_scores = scores

            if (
                time.monotonic() >= y_not_before
                and has_doa_y_presence(scores)
            ):
                y_frame_count += 1
                if y_frame_count >= int(DOA_SEQUENCE_Y_FRAMES_REQUIRED):
                    state = "WAIT_X2"
                    x_frames.clear()
                    if DEBUG:
                        print("[DOA] XYX gate: Y(650+1050Hz) 확인")
            else:
                y_frame_count = 0

        if DEBUG and state == "WAIT_Y" and best_y_scores is not None:
            print(
                "[DOA] XYX Y 존재 미검출: "
                f"650={best_y_scores[650.0]:.1f}dBFS, "
                f"1050={best_y_scores[1050.0]:.1f}dBFS, "
                f"900={best_y_scores[900.0]:.1f}dBFS"
            )
        return None


    def estimate(self):
        """
        여러 번 측정하고 같은 방향 군집만 사용.

        반환:
            상대 방향(degree)

        예:
             0° = 기준 전방
           +30° = 한쪽 방향
           -30° = 반대 방향

        실제 좌우 부호는 MicArray A 배치에 따라
        DOA_SIGN으로 교정합니다.
        """

        if DOA_REQUIRE_XYX_SEQUENCE:
            return self._estimate_xyx_pair()

        measurements = []
        attempts = 0
        deadline = time.monotonic() + DOA_ESTIMATE_TIMEOUT_SEC

        while (
            len(measurements) < DOA_MEASUREMENTS
            and time.monotonic() < deadline
        ):
            attempts += 1

            try:

                angle = (
                    self._estimate_once()
                )

                measurements.append(
                    angle
                )

                if DEBUG:

                    print(
                        f"[DOA] "
                        f"{len(measurements)}/"
                        f"{DOA_MEASUREMENTS}"
                        f" (attempt {attempts})"
                        f" = "
                        f"{angle:.1f}°"
                    )

                stable_angle = _stable_cluster_angle(measurements)
                if stable_angle is not None:
                    self.last_diagnostics = dict(self.last_diagnostics)
                    self.last_diagnostics.update({
                        "cluster_result_angle_deg": float(stable_angle),
                        "cluster_measurements_deg": [
                            float(value) for value in measurements
                        ],
                        "cluster_count": len(measurements),
                        "cluster_max_pair_diff_deg": max(
                            (
                                abs(circular_angle_difference_deg(first, second))
                                for index, first in enumerate(measurements)
                                for second in measurements[index + 1:]
                            ),
                            default=0.0,
                        ),
                    })
                    if DEBUG:
                        print(
                            "[DOA] 조기 안정 군집 확정 = "
                            f"{stable_angle:.1f}°"
                        )
                    return stable_angle


            except Exception as e:

                if DEBUG and (
                    attempts == 1
                    or attempts % 4 == 0
                ):
                    print(
                        "[DOA] 유효 비콘 블록 대기/재시도:",
                        e,
                    )


        if len(measurements) < DOA_MIN_VALID_MEASUREMENTS:
            if DEBUG:
                print(
                    "[DOA] 유효 측정 부족: "
                    f"{len(measurements)}/"
                    f"{DOA_MIN_VALID_MEASUREMENTS}"
                )
            return None

        angle = _stable_cluster_angle(measurements)

        if angle is None:
            if DEBUG:
                print(
                    "[DOA] 방향 측정이 서로 일치하지 않습니다: "
                    f"{[round(value, 1) for value in measurements]}"
                )
            return None

        self.last_diagnostics = dict(self.last_diagnostics)
        self.last_diagnostics.update({
            "cluster_result_angle_deg": float(angle),
            "cluster_measurements_deg": [float(value) for value in measurements],
            "cluster_count": len(measurements),
            "cluster_max_pair_diff_deg": max(
                (
                    abs(circular_angle_difference_deg(first, second))
                    for index, first in enumerate(measurements)
                    for second in measurements[index + 1:]
                ),
                default=0.0,
            ),
        })

        if DEBUG:

            print(
                f"[DOA] 최종 방향 = "
                f"{angle:.1f}°"
            )


        return angle


    def close(self):

        self.audio.stop()
