"""Pure decision helpers for UWB recovery during map-based XYXY search."""

from __future__ import annotations

import math
from statistics import fmean, median, pstdev
from typing import Sequence


def assess_continuous_fresh_recovery(
    samples: Sequence[tuple[float, float]],
    *,
    fresh_started_at: float | None,
    now: float,
    required_duration_sec: float,
    min_samples: int,
    max_sigma_m: float,
) -> dict[str, object]:
    """Accept recovery after a continuous, low-variance FRESH interval."""

    duration_sec = (
        max(0.0, float(now) - float(fresh_started_at))
        if fresh_started_at is not None
        else 0.0
    )
    values = [
        float(value)
        for stamp, value in samples
        if fresh_started_at is not None
        and float(stamp) >= float(fresh_started_at)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    ]
    center = float(median(values)) if values else None
    sigma = float(pstdev(values)) if len(values) >= 2 else None
    reasons: list[str] = []
    if fresh_started_at is None:
        reasons.append("fresh_not_continuous")
    if duration_sec < float(required_duration_sec):
        reasons.append("fresh_duration_too_short")
    if len(values) < max(1, int(min_samples)):
        reasons.append("insufficient_fresh_samples")
    if sigma is None or sigma > float(max_sigma_m):
        reasons.append("range_variance_too_high")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "duration_sec": round(duration_sec, 4),
        "fresh_samples": len(values),
        "median_m": None if center is None else round(center, 4),
        "sigma_m": None if sigma is None else round(sigma, 4),
    }


def assess_stationary_uwb_recovery(
    samples: Sequence[tuple[float, str, float | None]],
    *,
    robot_displacement_m: float | None,
    min_fresh_samples: int,
    min_fresh_ratio: float,
    max_sigma_m: float,
    max_stationary_motion_m: float,
    range_motion_margin_m: float,
) -> dict[str, object]:
    """Assess whether UWB genuinely recovered at one stationary map point.

    A changing NLOS value can be labelled ``FRESH`` by the low-level reader.
    Recovery therefore needs more than a continuous status duration: enough
    FRESH samples across the complete stationary window, low variance, and a
    range change that is plausible for the measured robot displacement.
    """

    usable = [
        (float(stamp), str(status).upper(), value)
        for stamp, status, value in samples
        if math.isfinite(float(stamp))
    ]
    fresh = [
        (stamp, float(value))
        for stamp, status, value in usable
        if status == "FRESH"
        and value is not None
        and math.isfinite(float(value))
        and float(value) >= 0.0
    ]
    total_samples = len(usable)
    fresh_values = [value for _stamp, value in fresh]
    fresh_count = len(fresh_values)
    fresh_ratio = fresh_count / total_samples if total_samples else 0.0
    median_m = float(median(fresh_values)) if fresh_values else None
    sigma_m = float(pstdev(fresh_values)) if len(fresh_values) >= 2 else None
    mad_sigma_m = None
    early_median_m = None
    late_median_m = None
    range_shift_m = None
    if fresh_values:
        center = float(median(fresh_values))
        mad_sigma_m = 1.4826 * float(median([abs(value - center) for value in fresh_values]))
        edge_count = max(1, fresh_count // 3)
        early_median_m = float(median(fresh_values[:edge_count]))
        late_median_m = float(median(fresh_values[-edge_count:]))
        range_shift_m = abs(late_median_m - early_median_m)

    reasons: list[str] = []
    if total_samples <= 0:
        reasons.append("no_status_samples")
    elif usable[-1][1] != "FRESH":
        reasons.append("latest_status_not_fresh")
    if fresh_count < max(1, int(min_fresh_samples)):
        reasons.append("insufficient_fresh_samples")
    if fresh_ratio < float(min_fresh_ratio):
        reasons.append("fresh_ratio_too_low")
    if sigma_m is None or sigma_m > float(max_sigma_m):
        reasons.append("range_variance_too_high")
    if robot_displacement_m is None:
        reasons.append("robot_pose_unavailable")
    elif robot_displacement_m > float(max_stationary_motion_m):
        reasons.append("robot_not_stationary")
    if range_shift_m is None:
        reasons.append("range_shift_unavailable")
    elif robot_displacement_m is not None and range_shift_m > (
        max(0.0, float(robot_displacement_m)) + float(range_motion_margin_m)
    ):
        reasons.append("range_change_inconsistent_with_motion")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "total_samples": total_samples,
        "fresh_samples": fresh_count,
        "fresh_ratio": round(fresh_ratio, 4),
        "median_m": None if median_m is None else round(median_m, 4),
        "sigma_m": None if sigma_m is None else round(sigma_m, 4),
        "mad_sigma_m": None if mad_sigma_m is None else round(mad_sigma_m, 4),
        "early_median_m": None if early_median_m is None else round(early_median_m, 4),
        "late_median_m": None if late_median_m is None else round(late_median_m, 4),
        "range_shift_m": None if range_shift_m is None else round(range_shift_m, 4),
        "robot_displacement_m": (
            None if robot_displacement_m is None else round(float(robot_displacement_m), 4)
        ),
    }


def assess_repeated_xyxy_peak(
    samples: Sequence[dict[str, object]],
    *,
    current_cycle: int,
    min_packets: int,
    min_repeats: int,
    repeat_radius_m: float,
    max_repeat_spread_db: float,
    required_margin_db: float,
) -> dict[str, object]:
    """Accept a loud XYXY location only when it repeats in this cycle."""

    usable: list[tuple[int, float, float, float]] = []
    for index, sample in enumerate(samples):
        if int(sample.get("cycle", -1)) != int(current_cycle):
            continue
        if int(sample.get("packet_valid_packets", 0)) < int(min_packets):
            continue
        point = sample.get("map_point")
        level = sample.get("direction_level_dbfs")
        if not isinstance(point, dict) or level is None:
            continue
        try:
            x_m = float(point["x_m"])
            y_m = float(point["y_m"])
            level_dbfs = float(level)
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in (x_m, y_m, level_dbfs)):
            usable.append((index, x_m, y_m, level_dbfs))

    result: dict[str, object] = {
        "accepted": False,
        "reason": "insufficient_current_cycle_samples",
        "cycle": int(current_cycle),
        "usable_samples": len(usable),
        "repeat_count": 0,
    }
    if len(usable) < max(2, int(min_repeats) + 1):
        return result

    candidates: list[dict[str, object]] = []
    for _seed_index, seed_x, seed_y, _seed_level in usable:
        members = [
            item for item in usable
            if math.hypot(item[1] - seed_x, item[2] - seed_y)
            <= float(repeat_radius_m)
        ]
        if len(members) < int(min_repeats):
            continue
        levels = [item[3] for item in members]
        spread_db = max(levels) - min(levels)
        if spread_db > float(max_repeat_spread_db):
            continue
        candidates.append({
            "member_indices": [item[0] for item in members],
            "repeat_count": len(members),
            "x_m": fmean(item[1] for item in members),
            "y_m": fmean(item[2] for item in members),
            "level_dbfs": float(median(levels)),
            "spread_db": spread_db,
        })

    if not candidates:
        result["reason"] = "no_repeated_peak_cluster"
        return result
    best = max(candidates, key=lambda item: float(item["level_dbfs"]))
    member_indices = set(int(value) for value in best["member_indices"])
    runner_levels = [
        level for index, _x_m, _y_m, level in usable
        if index not in member_indices
    ]
    result.update({
        "repeat_count": int(best["repeat_count"]),
        "best_x_m": round(float(best["x_m"]), 4),
        "best_y_m": round(float(best["y_m"]), 4),
        "best_level_dbfs": round(float(best["level_dbfs"]), 4),
        "repeat_spread_db": round(float(best["spread_db"]), 4),
    })
    if not runner_levels:
        result["reason"] = "no_independent_runner_up"
        return result
    runner_up_dbfs = max(runner_levels)
    advantage_db = float(best["level_dbfs"]) - runner_up_dbfs
    result.update({
        "runner_up_dbfs": round(runner_up_dbfs, 4),
        "advantage_db": round(advantage_db, 4),
    })
    if advantage_db < float(required_margin_db):
        result["reason"] = "repeated_peak_margin_too_small"
        return result
    result["accepted"] = True
    result["reason"] = "accepted"
    return result
