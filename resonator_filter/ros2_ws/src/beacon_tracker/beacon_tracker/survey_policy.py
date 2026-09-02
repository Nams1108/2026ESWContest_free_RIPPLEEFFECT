#!/usr/bin/env python3
"""Pure decision gates for the map-first UWB/XYXY survey phase."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median, pstdev
from typing import Sequence


@dataclass(frozen=True)
class SurveyGateAssessment:
    ready: bool
    reason: str | None
    required_joint_waypoints: int
    required_observed_zones: int


@dataclass(frozen=True)
class SurveyDepartureAssessment:
    """Robust UWB departure state for the map-first collection phase."""

    ready: bool
    best_median_m: float | None
    window_median_m: float | None
    window_sigma_m: float | None
    increase_m: float | None
    sample_count: int
    span_sec: float
    reason: str


def assess_survey_departure(
    samples: Sequence[tuple[float, float]],
    *,
    previous_best_median_m: float | None,
    window_sec: float,
    minimum_span_sec: float,
    minimum_samples: int,
    required_increase_m: float,
    max_sigma_m: float,
) -> SurveyDepartureAssessment:
    """Detect sustained movement away from the best recent UWB location.

    A single NLOS spike must not end map evidence collection.  Only the most
    recent time window is used, it must span the requested duration, contain
    enough FRESH samples and remain below the configured dispersion limit.
    The running reference is the smallest *stable window median*, not the
    smallest individual range sample.
    """

    if window_sec <= 0.0 or minimum_span_sec < 0.0:
        raise ValueError("survey departure window durations are invalid")
    if minimum_samples < 1 or required_increase_m <= 0.0 or max_sigma_m <= 0.0:
        raise ValueError("survey departure thresholds must be positive")

    usable = sorted(
        (
            (float(stamp), float(value))
            for stamp, value in samples
            if math.isfinite(float(stamp))
            and math.isfinite(float(value))
            and float(value) >= 0.0
        ),
        key=lambda item: item[0],
    )
    if not usable:
        return SurveyDepartureAssessment(
            False, previous_best_median_m, None, None, None, 0, 0.0,
            "no_fresh_samples",
        )

    latest = usable[-1][0]
    window = [item for item in usable if latest - item[0] <= window_sec]
    span_sec = window[-1][0] - window[0][0] if len(window) >= 2 else 0.0
    if len(window) < minimum_samples:
        return SurveyDepartureAssessment(
            False, previous_best_median_m, None, None, None, len(window),
            span_sec, "insufficient_samples",
        )
    if span_sec < minimum_span_sec:
        return SurveyDepartureAssessment(
            False, previous_best_median_m, None, None, None, len(window),
            span_sec, "window_too_short",
        )

    values = [value for _stamp, value in window]
    center = float(median(values))
    sigma = float(pstdev(values)) if len(values) >= 2 else 0.0
    if sigma > max_sigma_m:
        return SurveyDepartureAssessment(
            False, previous_best_median_m, center, sigma, None, len(window),
            span_sec, "range_variance_too_high",
        )

    best = center if previous_best_median_m is None else min(
        float(previous_best_median_m), center
    )
    increase = center - best
    ready = increase >= required_increase_m
    return SurveyDepartureAssessment(
        ready,
        best,
        center,
        sigma,
        increase,
        len(window),
        span_sec,
        "sustained_range_increase" if ready else "collecting",
    )


def assess_survey_gate(
    *,
    first_fresh_seen: bool,
    range_limit_seen: bool,
    coverage_complete: bool,
    select_on_coverage_complete: bool,
    joint_waypoints: int,
    configured_min_joint_waypoints: int,
    evidence_min_waypoints: int,
    observed_zones: int,
    configured_min_observed_zones: int,
    total_zones: int,
) -> SurveyGateAssessment:
    """Decide whether timestamped map evidence may select a room.

    Reaching the UWB boundary is only a collection-completion signal.  It
    cannot bypass the independent spatial and joint UWB/XYXY evidence gates.
    A completed full-map pass is a bounded fallback when the deployment map
    never permits a FRESH reading at exactly the configured boundary.
    """

    required_joint = max(
        1,
        int(configured_min_joint_waypoints),
        int(evidence_min_waypoints),
    )
    required_zones = min(
        max(1, int(total_zones)),
        max(1, int(configured_min_observed_zones)),
    )
    if not first_fresh_seen:
        return SurveyGateAssessment(False, None, required_joint, required_zones)
    if int(joint_waypoints) < required_joint:
        return SurveyGateAssessment(False, None, required_joint, required_zones)
    if int(observed_zones) < required_zones:
        return SurveyGateAssessment(False, None, required_joint, required_zones)
    if range_limit_seen:
        return SurveyGateAssessment(
            True,
            "UWB_RANGE_LIMIT",
            required_joint,
            required_zones,
        )
    if coverage_complete and select_on_coverage_complete:
        return SurveyGateAssessment(
            True,
            "COVERAGE_COMPLETE",
            required_joint,
            required_zones,
        )
    return SurveyGateAssessment(False, None, required_joint, required_zones)
