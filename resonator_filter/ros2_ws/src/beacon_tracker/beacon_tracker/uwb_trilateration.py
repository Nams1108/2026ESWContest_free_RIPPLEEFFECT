#!/usr/bin/env python3
"""Robust helpers for locating one stationary UWB anchor in 2D."""

from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RangeAggregate:
    """Filtered range statistics collected while the robot is stationary."""

    range_m: float
    sigma_m: float
    used_count: int
    total_count: int


@dataclass(frozen=True)
class RangeObservation:
    """One stationary robot position and its filtered UWB range."""

    x_m: float
    y_m: float
    range_m: float
    sigma_m: float = 0.1


@dataclass(frozen=True)
class TrilaterationResult:
    """Estimated UWB anchor position and fit diagnostics."""

    x_m: float
    y_m: float
    rms_residual_m: float
    max_residual_m: float
    confidence_radius_m: float
    observation_count: int
    iterations: int


def aggregate_ranges(
    values: Iterable[float],
    *,
    outlier_scale: float = 3.5,
    minimum_sigma_m: float = 0.03,
) -> RangeAggregate:
    """Return a median/MAD estimate after rejecting large range outliers."""

    samples = [
        float(value)
        for value in values
        if math.isfinite(float(value)) and float(value) >= 0.0
    ]
    if not samples:
        raise ValueError("at least one finite non-negative range is required")
    if outlier_scale <= 0.0 or minimum_sigma_m <= 0.0:
        raise ValueError("outlier_scale and minimum_sigma_m must be positive")

    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    sigma = max(1.4826 * mad, minimum_sigma_m)
    threshold = outlier_scale * sigma
    filtered = [value for value in samples if abs(value - median) <= threshold]
    if not filtered:
        filtered = samples

    filtered_median = statistics.median(filtered)
    filtered_mad = statistics.median(
        abs(value - filtered_median) for value in filtered
    )
    filtered_sigma = max(1.4826 * filtered_mad, minimum_sigma_m)
    return RangeAggregate(
        range_m=float(filtered_median),
        sigma_m=float(filtered_sigma),
        used_count=len(filtered),
        total_count=len(samples),
    )


def estimate_anchor_position(
    observations: Sequence[RangeObservation],
    *,
    minimum_triangle_area_m2: float = 0.04,
    huber_delta_m: float = 0.2,
    max_iterations: int = 25,
) -> TrilaterationResult:
    """Estimate a 2D anchor with weighted robust nonlinear least squares."""

    valid = [
        observation
        for observation in observations
        if all(
            math.isfinite(value)
            for value in (
                observation.x_m,
                observation.y_m,
                observation.range_m,
                observation.sigma_m,
            )
        )
        and observation.range_m >= 0.0
        and observation.sigma_m > 0.0
    ]
    if len(valid) < 3:
        raise ValueError("at least three valid observations are required")
    if minimum_triangle_area_m2 <= 0.0:
        raise ValueError("minimum_triangle_area_m2 must be positive")
    if huber_delta_m <= 0.0 or max_iterations <= 0:
        raise ValueError("solver parameters must be positive")

    area = _maximum_triangle_area(valid)
    if area < minimum_triangle_area_m2:
        raise ValueError(
            f"observation geometry is nearly collinear: area={area:.4f}m^2"
        )

    x_m, y_m = _linear_initial_estimate(valid)
    iterations = 0
    final_normal = (0.0, 0.0, 0.0)

    for iterations in range(1, max_iterations + 1):
        h_xx = 0.0
        h_xy = 0.0
        h_yy = 0.0
        g_x = 0.0
        g_y = 0.0

        for observation in valid:
            dx = x_m - observation.x_m
            dy = y_m - observation.y_m
            distance = max(math.hypot(dx, dy), 1e-9)
            residual = distance - observation.range_m
            j_x = dx / distance
            j_y = dy / distance
            base_weight = 1.0 / max(observation.sigma_m**2, 1e-6)
            robust_weight = min(1.0, huber_delta_m / max(abs(residual), 1e-12))
            weight = base_weight * robust_weight

            h_xx += weight * j_x * j_x
            h_xy += weight * j_x * j_y
            h_yy += weight * j_y * j_y
            g_x += weight * j_x * residual
            g_y += weight * j_y * residual

        step_x, step_y = _solve_symmetric_2x2(
            h_xx,
            h_xy,
            h_yy,
            -g_x,
            -g_y,
        )
        x_m += step_x
        y_m += step_y
        final_normal = (h_xx, h_xy, h_yy)
        if math.hypot(step_x, step_y) < 1e-5:
            break

    residuals = [
        math.hypot(x_m - observation.x_m, y_m - observation.y_m)
        - observation.range_m
        for observation in valid
    ]
    rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    max_residual = max(abs(value) for value in residuals)
    confidence_radius = _confidence_radius(final_normal, rms)
    if not all(
        math.isfinite(value)
        for value in (x_m, y_m, rms, max_residual, confidence_radius)
    ):
        raise ValueError("trilateration produced a non-finite result")

    return TrilaterationResult(
        x_m=x_m,
        y_m=y_m,
        rms_residual_m=rms,
        max_residual_m=max_residual,
        confidence_radius_m=confidence_radius,
        observation_count=len(valid),
        iterations=iterations,
    )


def _maximum_triangle_area(observations: Sequence[RangeObservation]) -> float:
    maximum = 0.0
    for first, second, third in itertools.combinations(observations, 3):
        twice_area = abs(
            (second.x_m - first.x_m) * (third.y_m - first.y_m)
            - (second.y_m - first.y_m) * (third.x_m - first.x_m)
        )
        maximum = max(maximum, 0.5 * twice_area)
    return maximum


def _linear_initial_estimate(
    observations: Sequence[RangeObservation],
) -> tuple[float, float]:
    reference = observations[0]
    h_xx = 0.0
    h_xy = 0.0
    h_yy = 0.0
    g_x = 0.0
    g_y = 0.0

    for observation in observations[1:]:
        a_x = 2.0 * (observation.x_m - reference.x_m)
        a_y = 2.0 * (observation.y_m - reference.y_m)
        b = (
            reference.range_m**2
            - observation.range_m**2
            + observation.x_m**2
            + observation.y_m**2
            - reference.x_m**2
            - reference.y_m**2
        )
        variance = reference.sigma_m**2 + observation.sigma_m**2
        weight = 1.0 / max(variance, 1e-6)
        h_xx += weight * a_x * a_x
        h_xy += weight * a_x * a_y
        h_yy += weight * a_y * a_y
        g_x += weight * a_x * b
        g_y += weight * a_y * b

    return _solve_symmetric_2x2(h_xx, h_xy, h_yy, g_x, g_y)


def _solve_symmetric_2x2(
    a: float,
    b: float,
    d: float,
    rhs_x: float,
    rhs_y: float,
) -> tuple[float, float]:
    determinant = a * d - b * b
    if abs(determinant) < 1e-12:
        raise ValueError("trilateration geometry is singular")
    return (
        (d * rhs_x - b * rhs_y) / determinant,
        (-b * rhs_x + a * rhs_y) / determinant,
    )


def _confidence_radius(
    normal: tuple[float, float, float],
    rms_residual_m: float,
) -> float:
    del rms_residual_m
    h_xx, h_xy, h_yy = normal
    determinant = h_xx * h_yy - h_xy * h_xy
    if determinant <= 1e-12:
        return float("inf")
    covariance_trace = (h_xx + h_yy) / determinant
    return math.sqrt(max(covariance_trace, 0.0))
