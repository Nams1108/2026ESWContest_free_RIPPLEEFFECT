#!/usr/bin/env python3
"""Map-independent evidence planning for indoor beacon search.

This module intentionally contains no ROS imports.  It turns an occupancy
grid into safe coverage waypoints, coarse open-space zones and narrow portal
candidates, then ranks zones from stationary UWB and validated XYXY evidence.
The topology is a heuristic: it does not assume a particular building, room
number, corridor direction, or map origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class MapPoint:
    x_m: float
    y_m: float
    yaw_rad: float = 0.0


@dataclass(frozen=True)
class EvidenceWaypoint:
    """Robust stationary evidence collected at one Nav2 waypoint."""

    waypoint_id: int
    pose: MapPoint
    range_m: float
    range_sigma_m: float
    fresh_samples: int
    packet_valid_packets: int
    packet_locked: bool
    # Full four-tone weak-link metric: packet identity/diagnostic value.
    packet_level_dbfs: float | None
    packet_quality_db: float | None
    # Direction metric from the already validated XYXY packet. It is the
    # conservative weaker value of 900 Hz (X) and 1050 Hz (Y), which matches
    # the SIPEED resonator direction experiment. Old CSV/metrics omit it;
    # those records safely fall back to ``packet_level_dbfs``.
    packet_direction_level_dbfs: float | None = None
    packet_direction_quality_db: float | None = None
    # XYXY is measured by the SIPEED microphone, which may not be located at
    # base_link. UWB fit intentionally continues to use ``pose`` (the robot
    # center / UWB reference position).
    packet_pose: MapPoint | None = None
    # Retain the actual stationary observation time so room decisions can be
    # audited against Nav2/UWB/packet logs without relying on waypoint order.
    observed_wall_time_sec: float | None = None
    observed_monotonic_sec: float | None = None


@dataclass(frozen=True)
class DoaObservation:
    """One packet-validated, map-frame ReSpeaker bearing observation.

    ``bearing_rad`` is already converted from the microphone's relative DOA
    angle into the map frame.  It is deliberately only a soft likelihood for
    map zones: this project never turns one reflected acoustic bearing into a
    direct Nav2 point.
    """

    microphone_pose: MapPoint
    bearing_rad: float
    # Scale the configured DOA contribution by acoustic-space quality.
    # 1.375 maps the default 0.20 weight to 0.275 (general open space), while
    # 0.625 maps it to 0.125 (one-sided 120-degree open fan).
    confidence_weight: float = 1.0


@dataclass(frozen=True)
class DoaSpaceAssessment:
    """Static-map acoustic suitability around the ReSpeaker position."""

    allowed: bool
    mode: str
    reason: str
    nearest_wall_m: float
    open_sector_count: int
    fan_center_rad: float | None
    fan_width_rad: float
    fan_center_depth_m: float
    fan_open_ratio: float
    ray_angles_rad: tuple[float, ...]
    ray_depths_m: tuple[float, ...]
    confidence_weight: float


@dataclass(frozen=True)
class Zone:
    zone_id: int
    interior: MapPoint
    area_m2: float
    max_clearance_m: float


@dataclass(frozen=True)
class Portal:
    portal_id: int
    zone_a: int
    zone_b: int
    point: MapPoint
    clearance_m: float


@dataclass(frozen=True)
class EvidenceConfig:
    """Geometry and conservative decision gates for generic maps."""

    robot_clearance_m: float = 0.30
    coverage_spacing_m: float = 1.8
    coverage_max_waypoints: int = 60
    zone_clearance_m: float = 0.60
    portal_min_clearance_m: float = 0.25
    portal_max_clearance_m: float = 0.65
    minimum_zone_area_m2: float = 1.2
    candidate_spacing_m: float = 0.50
    max_range_sigma_m: float = 0.45
    min_evidence_waypoints: int = 4
    min_packet_valid_packets: int = 2
    portal_evidence_radius_m: float = 1.5
    # A missing SLAM return can otherwise look like a narrow door. Coverage
    # candidates and portal seeds must remain this far from unknown map cells.
    # This is intentionally separate from body clearance: it rejects dubious
    # goals without making every physical corridor too narrow to traverse.
    coverage_unknown_clearance_m: float = 0.40
    portal_unknown_clearance_m: float = 0.35
    # A static indoor map must not turn the unobserved exterior around the
    # building into a coverage region. This also contains a corrupted /map
    # sample where unknown (-1) cells arrive as ordinary free cells.
    exclude_border_connected_free_space: bool = True
    huber_delta_sigma: float = 2.0
    range_weight: float = 0.70
    audio_weight: float = 0.30
    # DOA is a reflected-sound-prone tie breaker.  It is intentionally lower
    # than the UWB and full-XYXY evidence weights and only participates after
    # multiple independent stable observations have been collected.
    doa_weight: float = 0.20
    doa_sigma_rad: float = 0.55
    doa_min_observations: int = 2
    doa_require_line_of_sight: bool = True
    min_decision_margin: float = 0.12
    min_audio_margin_db: float = 2.0
    entry_staging_distance_m: float = 0.45
    entry_interior_distance_m: float = 0.90

    def __post_init__(self) -> None:
        if self.robot_clearance_m <= 0.0 or self.coverage_spacing_m <= 0.0:
            raise ValueError("clearance and coverage spacing must be positive")
        if self.coverage_max_waypoints <= 0 or self.candidate_spacing_m <= 0.0:
            raise ValueError("coverage limit and candidate spacing must be positive")
        if not (self.robot_clearance_m <= self.zone_clearance_m):
            raise ValueError("zone clearance must be at least robot clearance")
        if not (0.0 < self.portal_min_clearance_m <= self.portal_max_clearance_m):
            raise ValueError("portal clearance range is invalid")
        if self.minimum_zone_area_m2 <= 0.0 or self.max_range_sigma_m <= 0.0:
            raise ValueError("zone area and range sigma must be positive")
        if self.min_evidence_waypoints < 3 or self.min_packet_valid_packets < 1:
            raise ValueError("evidence gates are too small")
        if self.portal_evidence_radius_m <= 0.0 or self.huber_delta_sigma <= 0.0:
            raise ValueError("portal radius and Huber delta must be positive")
        if (
            self.coverage_unknown_clearance_m < 0.0
            or self.portal_unknown_clearance_m < 0.0
        ):
            raise ValueError("unknown-space clearances must be non-negative")
        if (
            self.range_weight < 0.0
            or self.audio_weight < 0.0
            or self.doa_weight < 0.0
        ):
            raise ValueError("weights must be non-negative")
        if self.range_weight + self.audio_weight <= 0.0:
            raise ValueError("at least one score weight must be positive")
        if self.doa_sigma_rad <= 0.0 or self.doa_min_observations < 1:
            raise ValueError("DOA uncertainty and observation gate are invalid")


@dataclass
class MapTopology:
    """Free-space topology derived from a static OccupancyGrid."""

    resolution_m: float
    origin_x_m: float
    origin_y_m: float
    origin_yaw_rad: float
    free: np.ndarray
    clearance_m: np.ndarray
    # Distance to cells which have never been observed by mapping (-1). This
    # is not a collision clearance; it is a map-confidence clearance used to
    # reject phantom doors created by missing scans.
    unknown_clearance_m: np.ndarray
    zone_labels: np.ndarray
    zones: tuple[Zone, ...]
    portals: tuple[Portal, ...]
    config: EvidenceConfig

    @property
    def shape(self) -> tuple[int, int]:
        return self.free.shape

    def cell_to_point(self, row: int, column: int, yaw_rad: float = 0.0) -> MapPoint:
        local_x = (float(column) + 0.5) * self.resolution_m
        local_y = (float(row) + 0.5) * self.resolution_m
        cos_yaw = math.cos(self.origin_yaw_rad)
        sin_yaw = math.sin(self.origin_yaw_rad)
        return MapPoint(
            x_m=self.origin_x_m + cos_yaw * local_x - sin_yaw * local_y,
            y_m=self.origin_y_m + sin_yaw * local_x + cos_yaw * local_y,
            yaw_rad=yaw_rad,
        )

    def point_to_cell(self, point: MapPoint) -> tuple[int, int] | None:
        dx = point.x_m - self.origin_x_m
        dy = point.y_m - self.origin_y_m
        cos_yaw = math.cos(self.origin_yaw_rad)
        sin_yaw = math.sin(self.origin_yaw_rad)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        column = math.floor(local_x / self.resolution_m)
        row = math.floor(local_y / self.resolution_m)
        height, width = self.shape
        if not (0 <= row < height and 0 <= column < width):
            return None
        return row, column

    def safe(self, point: MapPoint) -> bool:
        cell = self.point_to_cell(point)
        if cell is None:
            return False
        row, column = cell
        return bool(
            self.free[row, column]
            and self.clearance_m[row, column] >= self.config.robot_clearance_m
            and self.unknown_clearance_m[row, column]
            >= self.config.coverage_unknown_clearance_m
        )

    def zone_for_point(self, point: MapPoint) -> int | None:
        cell = self.point_to_cell(point)
        if cell is None:
            return None
        zone_id = int(self.zone_labels[cell])
        return zone_id if zone_id > 0 else None


def assess_doa_open_space(
    topology: MapTopology,
    sensor_pose: MapPoint,
    *,
    ray_step_deg: float = 10.0,
    ray_max_range_m: float = 3.0,
    minimum_wall_clearance_m: float = 0.60,
    general_wall_clearance_m: float = 1.20,
    open_depth_m: float = 1.50,
    general_min_open_sectors: int = 4,
    fan_width_deg: float = 120.0,
    fan_center_depth_m: float = 2.0,
    fan_min_open_ratio: float = 0.70,
    general_confidence_weight: float = 1.375,
    fan_confidence_weight: float = 0.625,
) -> DoaSpaceAssessment:
    """Classify a pose as general-open, one-sided-open, or reflection-prone.

    Rays stop at occupied, unknown, or out-of-map cells.  The six-sector test
    prevents a long but narrow corridor from looking open, while the fan test
    admits a doorway/corner pose only when at least 120 degrees opens into a
    genuinely deep area.
    """

    if not (0.0 < ray_step_deg <= 60.0):
        raise ValueError("DOA ray step must be in (0, 60] degrees")
    if ray_max_range_m <= 0.0 or open_depth_m <= 0.0:
        raise ValueError("DOA ray distances must be positive")
    if not (0.0 <= fan_min_open_ratio <= 1.0):
        raise ValueError("DOA fan open ratio must be between zero and one")
    if not (0.0 < fan_width_deg <= 360.0):
        raise ValueError("DOA fan width must be in (0, 360] degrees")

    sensor_cell = topology.point_to_cell(sensor_pose)
    if sensor_cell is None:
        return DoaSpaceAssessment(
            False, "BLOCKED", "sensor_outside_map", 0.0, 0,
            None, 0.0, 0.0, 0.0, (), (), 0.0,
        )
    row, column = sensor_cell
    if not bool(topology.free[row, column]):
        return DoaSpaceAssessment(
            False, "BLOCKED", "sensor_not_in_observed_free_space", 0.0, 0,
            None, 0.0, 0.0, 0.0, (), (), 0.0,
        )

    nearest_wall_m = float(topology.clearance_m[row, column])
    ray_count = max(6, int(round(360.0 / ray_step_deg)))
    actual_step_rad = 2.0 * math.pi / float(ray_count)
    angles = tuple(
        _wrap_angle_rad(sensor_pose.yaw_rad + index * actual_step_rad)
        for index in range(ray_count)
    )
    depths = tuple(
        _raycast_observed_free_depth(
            topology,
            sensor_pose,
            angle,
            max_range_m=ray_max_range_m,
        )
        for angle in angles
    )

    sector_depths: list[float] = []
    for sector in range(6):
        sector_angle = _wrap_angle_rad(
            sensor_pose.yaw_rad + sector * math.pi / 3.0
        )
        index = min(
            range(ray_count),
            key=lambda item: abs(_wrap_angle_rad(angles[item] - sector_angle)),
        )
        sector_depths.append(depths[index])
    open_sector_count = sum(depth >= open_depth_m for depth in sector_depths)

    fan_samples = min(
        ray_count,
        max(1, int(math.ceil(math.radians(fan_width_deg) / actual_step_rad))),
    )
    best_fan: tuple[float, float, float, float] | None = None
    # tuple ordering: ratio, center depth, mean depth, center angle
    doubled_depths = depths + depths
    for start in range(ray_count):
        window = doubled_depths[start:start + fan_samples]
        ratio = sum(depth >= open_depth_m for depth in window) / float(fan_samples)
        center_index = (start + fan_samples // 2) % ray_count
        center_depth = depths[center_index]
        mean_depth = float(np.mean(window))
        candidate = (ratio, center_depth, mean_depth, angles[center_index])
        if best_fan is None or candidate[:3] > best_fan[:3]:
            best_fan = candidate

    fan_ratio, center_depth, _mean_depth, center_angle = (
        best_fan if best_fan is not None else (0.0, 0.0, 0.0, 0.0)
    )
    fan_width_rad = fan_samples * actual_step_rad

    if (
        nearest_wall_m >= general_wall_clearance_m
        and open_sector_count >= int(general_min_open_sectors)
    ):
        return DoaSpaceAssessment(
            True,
            "GENERAL_OPEN",
            "general_open_space",
            nearest_wall_m,
            open_sector_count,
            center_angle,
            fan_width_rad,
            center_depth,
            fan_ratio,
            angles,
            depths,
            general_confidence_weight,
        )

    if (
        nearest_wall_m >= minimum_wall_clearance_m
        and fan_ratio >= fan_min_open_ratio
        and center_depth >= fan_center_depth_m
    ):
        return DoaSpaceAssessment(
            True,
            "OPEN_FAN",
            "one_sided_open_fan",
            nearest_wall_m,
            open_sector_count,
            center_angle,
            fan_width_rad,
            center_depth,
            fan_ratio,
            angles,
            depths,
            fan_confidence_weight,
        )

    reason = (
        "wall_too_close"
        if nearest_wall_m < minimum_wall_clearance_m
        else "no_qualified_open_fan"
    )
    return DoaSpaceAssessment(
        False,
        "BLOCKED",
        reason,
        nearest_wall_m,
        open_sector_count,
        center_angle,
        fan_width_rad,
        center_depth,
        fan_ratio,
        angles,
        depths,
        0.0,
    )


def doa_bearing_is_open(
    assessment: DoaSpaceAssessment,
    bearing_rad: float,
    *,
    minimum_depth_m: float = 1.50,
) -> bool:
    """Return whether a final world-frame DOA points into mapped free space."""

    if not assessment.allowed or not assessment.ray_angles_rad:
        return False
    index = min(
        range(len(assessment.ray_angles_rad)),
        key=lambda item: abs(
            _wrap_angle_rad(assessment.ray_angles_rad[item] - bearing_rad)
        ),
    )
    if assessment.ray_depths_m[index] < minimum_depth_m:
        return False
    if assessment.mode != "OPEN_FAN" or assessment.fan_center_rad is None:
        return True
    return abs(_wrap_angle_rad(bearing_rad - assessment.fan_center_rad)) <= (
        assessment.fan_width_rad * 0.5
    )


def _raycast_observed_free_depth(
    topology: MapTopology,
    start: MapPoint,
    bearing_rad: float,
    *,
    max_range_m: float,
) -> float:
    """Measure map-observed free depth without stepping over thin walls."""

    step_m = max(topology.resolution_m * 0.5, 0.01)
    distance_m = 0.0
    while distance_m + step_m <= max_range_m + 1e-9:
        distance_m += step_m
        point = MapPoint(
            start.x_m + math.cos(bearing_rad) * distance_m,
            start.y_m + math.sin(bearing_rad) * distance_m,
        )
        cell = topology.point_to_cell(point)
        if cell is None or not bool(topology.free[cell]):
            return max(0.0, distance_m - step_m)
    return float(max_range_m)


@dataclass(frozen=True)
class ZoneScore:
    zone: Zone
    fit_cost: float
    proximity_cost_m: float
    audio_level_dbfs: float | None
    portal: Portal | None
    range_score: float
    audio_score: float
    total_score: float
    doa_fit_cost: float | None = None
    doa_score: float | None = None
    doa_observation_count: int = 0


@dataclass(frozen=True)
class ZoneProbability:
    """Live, relative beacon probability for one map-derived section.

    ``probability`` is normalized across all current sections.  It is an
    evidence-based ranking, not a calibrated physical probability: UWB can be
    biased by NLOS and acoustic levels change around doors and reflections.
    The accompanying ``confidence`` in :class:`ZoneProbabilitySnapshot` makes
    that distinction explicit to the operator.
    """

    zone: Zone
    probability: float
    fit_cost: float | None
    proximity_cost_m: float | None
    audio_level_dbfs: float | None
    range_score: float | None
    audio_score: float | None
    total_score: float | None
    doa_fit_cost: float | None = None
    doa_score: float | None = None
    doa_observation_count: int = 0


@dataclass(frozen=True)
class ZoneProbabilitySnapshot:
    """One map-wide probability update suitable for ROS publication."""

    sections: tuple[ZoneProbability, ...]
    valid_evidence_count: int
    packet_evidence_count: int
    observed_zone_count: int
    confidence: float
    doa_observation_count: int = 0


@dataclass(frozen=True)
class RoomDecision:
    zone: Zone
    portal: Portal | None
    staging_pose: MapPoint | None
    interior_pose: MapPoint
    total_score: float
    runner_up_score: float | None
    reason: str


def build_topology(
    occupancy: Sequence[int] | np.ndarray,
    *,
    width: int,
    height: int,
    resolution_m: float,
    origin_x_m: float,
    origin_y_m: float,
    origin_yaw_rad: float,
    occupancy_threshold: int,
    unknown_is_blocked: bool,
    config: EvidenceConfig,
) -> MapTopology:
    """Create topology from any map-server OccupancyGrid geometry."""

    if width <= 0 or height <= 0 or resolution_m <= 0.0:
        raise ValueError("map dimensions and resolution must be positive")
    values = np.asarray(occupancy, dtype=np.int16)
    if values.size != width * height:
        raise ValueError("occupancy data size does not match map dimensions")
    grid = values.reshape((height, width))
    unknown = grid < 0
    free = grid < occupancy_threshold
    if unknown_is_blocked:
        free &= grid >= 0

    if config.exclude_border_connected_free_space and np.any(free):
        free_labels, _ = ndimage.label(free, structure=_cross_structure())
        border_labels = np.unique(np.concatenate((
            free_labels[0, :],
            free_labels[-1, :],
            free_labels[:, 0],
            free_labels[:, -1],
        )))
        border_labels = border_labels[border_labels != 0]
        if border_labels.size:
            free &= ~np.isin(free_labels, border_labels)

    clearance_m = ndimage.distance_transform_edt(free) * resolution_m
    unknown_clearance_m = (
        ndimage.distance_transform_edt(~unknown) * resolution_m
    )
    seed = free & (clearance_m >= config.zone_clearance_m)
    seed_labels, _ = ndimage.label(seed, structure=_cross_structure())
    seed_labels = _remove_small_seed_regions(
        seed_labels,
        resolution_m=resolution_m,
        min_area_m2=config.minimum_zone_area_m2,
    )
    if not np.any(seed_labels):
        seed_labels, _ = ndimage.label(
            free & (clearance_m >= config.robot_clearance_m),
            structure=_cross_structure(),
        )

    zone_labels = _propagate_zone_labels(free, seed_labels)
    zones = _build_zones(
        zone_labels,
        clearance_m,
        resolution_m,
        origin_x_m,
        origin_y_m,
        origin_yaw_rad,
    )
    topology = MapTopology(
        resolution_m=resolution_m,
        origin_x_m=origin_x_m,
        origin_y_m=origin_y_m,
        origin_yaw_rad=origin_yaw_rad,
        free=free,
        clearance_m=clearance_m,
        unknown_clearance_m=unknown_clearance_m,
        zone_labels=zone_labels,
        zones=zones,
        portals=(),
        config=config,
    )
    topology.portals = _build_portals(topology)
    return topology


def coverage_waypoints(
    topology: MapTopology,
    start: MapPoint,
) -> list[MapPoint]:
    """Generate an ordered, spacing-limited coverage route over free space."""

    traversable = topology.free & (
        topology.clearance_m >= topology.config.robot_clearance_m
    )
    safe = traversable & (
        topology.unknown_clearance_m
        >= topology.config.coverage_unknown_clearance_m
    )

    # Only inspect the traversable component that actually contains the
    # robot. Disconnected scan islands and rooms behind an unobserved wall
    # must never become planner preflight targets merely because their center
    # pixel is white in the static image.
    start_cell = topology.point_to_cell(start)
    if start_cell is None:
        return []
    # Connectivity follows observed free cells, while destinations still
    # require full robot clearance. A doorway may be narrower in the static
    # distance field than this conservative destination clearance but remain
    # traversable according to Nav2's exact footprint and live costmap.
    labels, _ = ndimage.label(topology.free, structure=_cross_structure())
    start_label = int(labels[start_cell])
    if start_label == 0:
        rows, columns = np.nonzero(traversable)
        if not len(rows):
            return []
        nearest_index = min(
            range(len(rows)),
            key=lambda index: _distance(
                start,
                topology.cell_to_point(int(rows[index]), int(columns[index])),
            ),
        )
        nearest = topology.cell_to_point(
            int(rows[nearest_index]), int(columns[nearest_index])
        )
        if _distance(start, nearest) > max(
            0.75, 2.0 * topology.config.robot_clearance_m
        ):
            return []
        start_label = int(
            labels[int(rows[nearest_index]), int(columns[nearest_index])]
        )
    safe &= labels == start_label
    stride = max(1, int(round(topology.config.coverage_spacing_m / topology.resolution_m)))
    candidates: list[tuple[int, int]] = []
    height, width = topology.shape
    for row0 in range(0, height, stride):
        for column0 in range(0, width, stride):
            block = safe[row0:min(row0 + stride, height), column0:min(column0 + stride, width)]
            if not np.any(block):
                continue
            local_row, local_column = np.unravel_index(
                int(np.argmax(topology.clearance_m[row0:min(row0 + stride, height), column0:min(column0 + stride, width)] * block)),
                block.shape,
            )
            candidates.append((row0 + int(local_row), column0 + int(local_column)))

    if not candidates:
        return []
    points = [topology.cell_to_point(row, column) for row, column in candidates]
    points = _spatially_reduce(
        points,
        start,
        topology.config.coverage_max_waypoints,
    )
    return _nearest_neighbor_order(points, start)


def closed_space_first_coverage_waypoints(
    topology: MapTopology,
    start: MapPoint,
) -> list[MapPoint]:
    """Order safe coverage points by map-derived room/portal structure.

    The base coverage generator remains the authority for connected, observed
    and robot-clear free space.  This function changes only ordering: zones
    with zero or one portal are treated as enclosed/terminal spaces and are
    visited before multi-portal corridor or hub zones.  No room number, map
    orientation, image coordinate, UWB value or acoustic bearing is used.

    Within each zone nearest-neighbour ordering keeps Nav2 legs short.  Any
    candidate which could not be assigned to a zone is retained at the end so
    coverage is never silently lost because of a coarse topology boundary.
    """

    base_route = coverage_waypoints(topology, start)
    if not base_route or not topology.zones:
        return base_route

    portal_degree = {zone.zone_id: 0 for zone in topology.zones}
    for portal in topology.portals:
        if portal.zone_a in portal_degree:
            portal_degree[portal.zone_a] += 1
        if portal.zone_b in portal_degree:
            portal_degree[portal.zone_b] += 1

    points_by_zone: dict[int, list[MapPoint]] = {
        zone.zone_id: [] for zone in topology.zones
    }
    unassigned: list[MapPoint] = []
    for point in base_route:
        zone_id = topology.zone_for_point(point)
        if zone_id is None or zone_id not in points_by_zone:
            unassigned.append(point)
        else:
            points_by_zone[zone_id].append(point)

    zones_by_id = {zone.zone_id: zone for zone in topology.zones}
    active_zone_ids = [
        zone_id for zone_id, points in points_by_zone.items() if points
    ]
    active_zone_ids.sort(
        key=lambda zone_id: (
            # Terminal/enclosed sections first, then corridor/hub sections.
            0 if portal_degree.get(zone_id, 0) <= 1 else 1,
            portal_degree.get(zone_id, 0),
            _distance(start, zones_by_id[zone_id].interior),
            zones_by_id[zone_id].area_m2,
            zone_id,
        )
    )

    ordered: list[MapPoint] = []
    current = start
    for zone_id in active_zone_ids:
        local = _nearest_neighbor_order(points_by_zone[zone_id], current)
        ordered.extend(local)
        if local:
            current = local[-1]
    ordered.extend(_nearest_neighbor_order(unassigned, current))
    return ordered


def zone_local_waypoints(
    topology: MapTopology,
    zone_id: int,
    anchor: MapPoint,
    *,
    radius_m: float,
    max_waypoints: int,
) -> list[MapPoint]:
    """Return a small safe sampling set that stays inside one section.

    A strong close UWB/XYXY observation must be verified locally before the
    search crosses a portal and resumes global coverage.  The candidates are
    generated from the current map, so this does not depend on a hand-made
    room list or a particular map orientation.
    """

    if radius_m <= 0.0 or max_waypoints <= 0:
        return []
    safe = (
        (topology.zone_labels == zone_id)
        & topology.free
        & (topology.clearance_m >= topology.config.robot_clearance_m)
        & (
            topology.unknown_clearance_m
            >= topology.config.coverage_unknown_clearance_m
        )
    )
    rows, columns = np.nonzero(safe)
    stride = max(
        1,
        int(round(topology.config.candidate_spacing_m / topology.resolution_m)),
    )
    candidates = [
        topology.cell_to_point(int(row), int(column))
        for row, column in zip(rows[::stride], columns[::stride])
    ]
    # Avoid re-sampling the exact anchor point, but retain a useful baseline
    # to determine whether the nearby UWB observation is repeatable.
    candidates = [
        point
        for point in candidates
        if 0.30 <= _distance(point, anchor) <= radius_m
    ]
    if not candidates:
        return []
    selected = _spatially_reduce(candidates, anchor, max_waypoints)
    return _nearest_neighbor_order(selected, anchor)


def rank_zones(
    topology: MapTopology,
    evidence: Sequence[EvidenceWaypoint],
    *,
    doa_observations: Sequence[DoaObservation] = (),
) -> list[ZoneScore]:
    """Rank map zones without assuming one sensor is a perfect locator.

    A DOA observation is only a weak map-section likelihood.  In particular,
    this function does not turn an acoustic ray into a coordinate or a Nav2
    target.  With fewer than ``doa_min_observations`` inputs, its behaviour is
    identical to the UWB + validated-XYXY ranker used previously.
    """

    valid = _valid_range_evidence(topology, evidence)
    if len(valid) < topology.config.min_evidence_waypoints:
        return []

    return _score_zones(topology, valid, doa_observations=doa_observations)


def estimate_zone_probabilities(
    topology: MapTopology,
    evidence: Sequence[EvidenceWaypoint],
    *,
    doa_observations: Sequence[DoaObservation] = (),
    temperature: float = 0.18,
) -> ZoneProbabilitySnapshot:
    """Estimate live relative beacon probability for every map section.

    The same robust UWB plus validated-XYXY scoring used for room entry is
    evaluated even before the conservative entry gate has enough evidence.
    With no valid UWB sample the output is deliberately uniform and reports
    zero confidence, instead of inventing a preferred room.
    """

    if temperature <= 0.0:
        raise ValueError("probability temperature must be positive")
    if not topology.zones:
        return ZoneProbabilitySnapshot((), 0, 0, 0, 0.0, 0)

    valid = _valid_range_evidence(topology, evidence)
    packet_count = sum(
        1
        for item in valid
        if (
            packet_audio_level_dbfs(item) is not None
            and item.packet_valid_packets >= topology.config.min_packet_valid_packets
        )
    )
    observed_zone_ids = {
        zone_id
        for item in valid
        for zone_id in (topology.zone_for_point(item.pose),)
        if zone_id is not None
    }
    confidence = _probability_confidence(
        topology,
        valid_count=len(valid),
        packet_count=packet_count,
        observed_zone_count=len(observed_zone_ids),
    )

    if not valid:
        uniform = 1.0 / len(topology.zones)
        return ZoneProbabilitySnapshot(
            sections=tuple(
                ZoneProbability(
                    zone=zone,
                    probability=uniform,
                    fit_cost=None,
                    proximity_cost_m=None,
                    audio_level_dbfs=None,
                    range_score=None,
                    audio_score=None,
                    total_score=None,
                )
                for zone in topology.zones
            ),
            valid_evidence_count=0,
            packet_evidence_count=0,
            observed_zone_count=0,
            confidence=0.0,
            doa_observation_count=0,
        )

    scores = _score_zones(
        topology,
        valid,
        doa_observations=doa_observations,
    )
    if not scores:
        uniform = 1.0 / len(topology.zones)
        return ZoneProbabilitySnapshot(
            sections=tuple(
                ZoneProbability(zone, uniform, None, None, None, None, None, None)
                for zone in topology.zones
            ),
            valid_evidence_count=len(valid),
            packet_evidence_count=packet_count,
            observed_zone_count=len(observed_zone_ids),
            confidence=confidence,
            doa_observation_count=0,
        )

    raw_scores = np.array([score.total_score for score in scores], dtype=float)
    logits = (raw_scores - float(np.max(raw_scores))) / temperature
    weights = np.exp(logits)
    probabilities = weights / float(np.sum(weights))
    sections = tuple(
        ZoneProbability(
            zone=score.zone,
            probability=float(probability),
            fit_cost=score.fit_cost,
            proximity_cost_m=score.proximity_cost_m,
            audio_level_dbfs=score.audio_level_dbfs,
            range_score=score.range_score,
            audio_score=score.audio_score,
            total_score=score.total_score,
            doa_fit_cost=score.doa_fit_cost,
            doa_score=score.doa_score,
            doa_observation_count=score.doa_observation_count,
        )
        for score, probability in zip(scores, probabilities)
    )
    return ZoneProbabilitySnapshot(
        sections=sections,
        valid_evidence_count=len(valid),
        packet_evidence_count=packet_count,
        observed_zone_count=len(observed_zone_ids),
        confidence=confidence,
        doa_observation_count=max(
            (score.doa_observation_count for score in scores),
            default=0,
        ),
    )


def _valid_range_evidence(
    topology: MapTopology,
    evidence: Sequence[EvidenceWaypoint],
) -> list[EvidenceWaypoint]:
    return [
        item
        for item in evidence
        if (
            math.isfinite(item.range_m)
            and item.range_m >= 0.0
            and math.isfinite(item.range_sigma_m)
            and item.range_sigma_m <= topology.config.max_range_sigma_m
        )
    ]


def packet_audio_level_dbfs(item: EvidenceWaypoint) -> float | None:
    """Return the XYXY level used for room/entrance ranking.

    Packet validation is always full XYXY before an observation enters this
    pipeline. Once that identity check is complete, direction/entrance
    ranking should use the SIPEED channels tuned for 900/1050 Hz. Older logs
    and decoder processes have no direction field, so retain the original
    four-tone weak-link metric as a compatibility fallback.
    """

    direction = item.packet_direction_level_dbfs
    if direction is not None and math.isfinite(direction):
        return float(direction)
    legacy = item.packet_level_dbfs
    if legacy is not None and math.isfinite(legacy):
        return float(legacy)
    return None


def _probability_confidence(
    topology: MapTopology,
    *,
    valid_count: int,
    packet_count: int,
    observed_zone_count: int,
) -> float:
    """Report data support, not certainty that a section contains a beacon."""

    evidence_factor = min(
        1.0,
        valid_count / float(max(1, topology.config.min_evidence_waypoints)),
    )
    spatial_factor = min(
        1.0,
        observed_zone_count / float(max(1, len(topology.zones))),
    )
    packet_target = max(1, topology.config.min_evidence_waypoints // 2)
    packet_factor = min(1.0, packet_count / float(packet_target))
    # A strong range fit from one observed section cannot represent confidence
    # in the *whole map*.  Make spatial coverage a multiplicative gate so an
    # operator does not mistake extrapolation for a completed room comparison.
    return evidence_factor * (0.45 + 0.55 * spatial_factor) * (
        0.70 + 0.30 * packet_factor
    )


def _score_zones(
    topology: MapTopology,
    valid: Sequence[EvidenceWaypoint],
    *,
    doa_observations: Sequence[DoaObservation] = (),
) -> list[ZoneScore]:
    """Score sections from pre-filtered evidence without an evidence-count gate."""

    fit_costs: dict[int, float] = {}
    proximity_costs: dict[int, float] = {}
    audio_by_zone: dict[int, tuple[float | None, Portal | None]] = {}
    doa_by_zone: dict[int, tuple[float | None, int]] = {}
    for zone in topology.zones:
        candidates = _zone_candidate_points(topology, zone.zone_id)
        if not candidates:
            continue
        fit_costs[zone.zone_id] = _robust_range_fit_cost(
            candidates,
            valid,
            topology.config.huber_delta_sigma,
        )
        proximity_costs[zone.zone_id] = _short_range_proximity_cost(
            candidates,
            valid,
        )
        audio_by_zone[zone.zone_id] = _zone_audio_evidence(
            topology,
            zone,
            valid,
        )
        doa_by_zone[zone.zone_id] = _zone_doa_fit_cost(
            topology,
            candidates,
            doa_observations,
        )

    fit_score = _inverse_normalized(fit_costs)
    proximity_score = _inverse_normalized(proximity_costs)
    finite_audio = {
        zone_id: value
        for zone_id, (value, _) in audio_by_zone.items()
        if value is not None and math.isfinite(value)
    }
    audio_score = _normalized(finite_audio)
    finite_doa = {
        zone_id: value
        for zone_id, (value, observation_count) in doa_by_zone.items()
        if (
            value is not None
            and observation_count >= topology.config.doa_min_observations
            and math.isfinite(value)
        )
    }
    doa_score = _inverse_normalized(finite_doa)
    # Do not re-weight UWB/XYXY when DOA could not compare at least two
    # candidate sections.  This preserves old behaviour under weak, blocked,
    # or reflected sound and avoids a one-zone DOA artefact becoming evidence.
    active_doa_weight = (
        topology.config.doa_weight
        * float(np.mean([
            observation.confidence_weight
            for observation in doa_observations
            if (
                math.isfinite(observation.confidence_weight)
                and observation.confidence_weight > 0.0
            )
        ]))
        if (
            len(finite_doa) >= 2
            and any(
                math.isfinite(observation.confidence_weight)
                and observation.confidence_weight > 0.0
                for observation in doa_observations
            )
        )
        else 0.0
    )
    weight_total = (
        topology.config.range_weight
        + topology.config.audio_weight
        + active_doa_weight
    )
    result: list[ZoneScore] = []
    for zone in topology.zones:
        if zone.zone_id not in fit_costs:
            continue
        audio_level, portal = audio_by_zone[zone.zone_id]
        range_component = 0.65 * fit_score[zone.zone_id] + 0.35 * proximity_score[zone.zone_id]
        audio_component = audio_score.get(zone.zone_id, 0.0)
        doa_cost, doa_count = doa_by_zone[zone.zone_id]
        doa_component = doa_score.get(zone.zone_id)
        total = (
            topology.config.range_weight * range_component
            + topology.config.audio_weight * audio_component
            + active_doa_weight * (doa_component or 0.0)
        ) / weight_total
        result.append(ZoneScore(
            zone=zone,
            fit_cost=fit_costs[zone.zone_id],
            proximity_cost_m=proximity_costs[zone.zone_id],
            audio_level_dbfs=audio_level,
            portal=portal,
            range_score=range_component,
            audio_score=audio_component,
            total_score=total,
            doa_fit_cost=doa_cost,
            doa_score=doa_component,
            doa_observation_count=doa_count,
        ))
    return sorted(result, key=lambda item: item.total_score, reverse=True)


def decide_room_entry(
    topology: MapTopology,
    evidence: Sequence[EvidenceWaypoint],
    *,
    doa_observations: Sequence[DoaObservation] = (),
) -> RoomDecision | None:
    """Return a conservative room/portal decision only when both agree."""

    ranked = rank_zones(
        topology,
        evidence,
        doa_observations=doa_observations,
    )
    if not ranked:
        return None
    best = ranked[0]
    if best.audio_level_dbfs is None:
        return None
    runner = ranked[1] if len(ranked) > 1 else None
    if runner is not None and best.total_score - runner.total_score < topology.config.min_decision_margin:
        return None
    # A physical doorway belongs to both adjacent zones.  Therefore compare
    # its locked-packet strength with the route baseline, not merely with the
    # same portal viewed from the corridor side.
    route_levels = [
        float(packet_audio_level_dbfs(item))
        for item in evidence
        if (
            packet_audio_level_dbfs(item) is not None
            and item.packet_valid_packets >= topology.config.min_packet_valid_packets
        )
    ]
    if not route_levels:
        return None
    if best.audio_level_dbfs - float(np.median(route_levels)) < topology.config.min_audio_margin_db:
        return None

    staging, interior = _entry_poses(topology, best.zone, best.portal)
    if interior is None:
        return None
    return RoomDecision(
        zone=best.zone,
        portal=best.portal,
        staging_pose=staging,
        interior_pose=interior,
        total_score=best.total_score,
        runner_up_score=runner.total_score if runner is not None else None,
        reason=(
            "robust UWB zone fit, locked XYXY doorway evidence, and "
            f"{best.doa_observation_count} stable map-frame DOA bearings agree"
            if best.doa_score is not None
            and best.doa_observation_count >= topology.config.doa_min_observations
            else "robust UWB zone fit and locked XYXY doorway evidence agree"
        ),
    )


def _cross_structure() -> np.ndarray:
    return np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8)


def _remove_small_seed_regions(labels: np.ndarray, *, resolution_m: float, min_area_m2: float) -> np.ndarray:
    labels = labels.copy()
    for label in range(1, int(labels.max()) + 1):
        if np.count_nonzero(labels == label) * resolution_m**2 < min_area_m2:
            labels[labels == label] = 0
    relabelled, _ = ndimage.label(labels > 0, structure=_cross_structure())
    return relabelled


def _propagate_zone_labels(free: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    if not np.any(seeds):
        return np.zeros_like(seeds, dtype=np.int32)
    _, indices = ndimage.distance_transform_edt(~(seeds > 0), return_indices=True)
    propagated = seeds[tuple(indices)]
    return np.where(free, propagated, 0).astype(np.int32)


def _build_zones(labels: np.ndarray, clearance_m: np.ndarray, resolution_m: float, origin_x_m: float, origin_y_m: float, origin_yaw_rad: float) -> tuple[Zone, ...]:
    zones: list[Zone] = []
    for zone_id in range(1, int(labels.max()) + 1):
        mask = labels == zone_id
        if not np.any(mask):
            continue
        row, column = np.unravel_index(int(np.argmax(clearance_m * mask)), mask.shape)
        local_x = (float(column) + 0.5) * resolution_m
        local_y = (float(row) + 0.5) * resolution_m
        cos_yaw = math.cos(origin_yaw_rad)
        sin_yaw = math.sin(origin_yaw_rad)
        zones.append(Zone(
            zone_id=zone_id,
            interior=MapPoint(
                origin_x_m + cos_yaw * local_x - sin_yaw * local_y,
                origin_y_m + sin_yaw * local_x + cos_yaw * local_y,
            ),
            area_m2=float(np.count_nonzero(mask) * resolution_m**2),
            max_clearance_m=float(clearance_m[row, column]),
        ))
    return tuple(zones)


def _build_portals(topology: MapTopology) -> tuple[Portal, ...]:
    labels = topology.zone_labels
    # A narrow band is not sufficient proof of a doorway: a missed wall scan
    # also produces one. Reject bands adjacent to unobserved map space, then
    # let the usual two-zone boundary test find internal, observed doors.
    free_band = (
        topology.free
        & (
            topology.clearance_m
            >= topology.config.portal_min_clearance_m
        )
        & (
            topology.clearance_m
            <= topology.config.portal_max_clearance_m
        )
        & (
            topology.unknown_clearance_m
            >= topology.config.portal_unknown_clearance_m
        )
    )
    pairs: dict[tuple[int, int], np.ndarray] = {}
    for dr, dc in ((1, 0), (0, 1)):
        source = labels[max(0, -dr):labels.shape[0] - max(0, dr), max(0, -dc):labels.shape[1] - max(0, dc)]
        target = labels[max(0, dr):labels.shape[0] - max(0, -dr), max(0, dc):labels.shape[1] - max(0, -dc)]
        source_band = free_band[max(0, -dr):free_band.shape[0] - max(0, dr), max(0, -dc):free_band.shape[1] - max(0, dc)]
        target_band = free_band[max(0, dr):free_band.shape[0] - max(0, -dr), max(0, dc):free_band.shape[1] - max(0, -dc)]
        different = (source > 0) & (target > 0) & (source != target) & (source_band | target_band)
        for left, right in zip(source[different], target[different]):
            pair = tuple(sorted((int(left), int(right))))
            mask = pairs.setdefault(pair, np.zeros_like(labels, dtype=bool))
            # The boundary cell is adequate as a generic portal seed.
            positions = np.argwhere((labels == left) | (labels == right))
            # Keep only the actual narrow boundary neighborhood below.
            del positions
            mask |= free_band & (((labels == pair[0]) | (labels == pair[1])))

    portals: list[Portal] = []
    portal_id = 1
    for pair, broad_mask in pairs.items():
        # Keep cells near the closest contact of the two assigned zones.
        a = labels == pair[0]
        b = labels == pair[1]
        distance_to_a = ndimage.distance_transform_edt(~a)
        distance_to_b = ndimage.distance_transform_edt(~b)
        boundary = broad_mask & (np.abs(distance_to_a - distance_to_b) <= 1.5)
        components, count = ndimage.label(boundary, structure=_cross_structure())
        for component in range(1, count + 1):
            mask = components == component
            if np.count_nonzero(mask) == 0:
                continue
            row, column = np.unravel_index(int(np.argmax(topology.clearance_m * mask)), mask.shape)
            portals.append(Portal(
                portal_id=portal_id,
                zone_a=pair[0],
                zone_b=pair[1],
                point=topology.cell_to_point(int(row), int(column)),
                clearance_m=float(topology.clearance_m[row, column]),
            ))
            portal_id += 1
    return tuple(portals)


def _spatially_reduce(points: list[MapPoint], start: MapPoint, limit: int) -> list[MapPoint]:
    if len(points) <= limit:
        return points
    selected = [min(points, key=lambda point: _distance(point, start))]
    remaining = [point for point in points if point != selected[0]]
    while remaining and len(selected) < limit:
        next_point = max(
            remaining,
            key=lambda point: min(_distance(point, chosen) for chosen in selected),
        )
        selected.append(next_point)
        remaining.remove(next_point)
    return selected


def _nearest_neighbor_order(points: list[MapPoint], start: MapPoint) -> list[MapPoint]:
    ordered: list[MapPoint] = []
    remaining = list(points)
    current = start
    while remaining:
        next_point = min(remaining, key=lambda point: _distance(point, current))
        ordered.append(next_point)
        remaining.remove(next_point)
        current = next_point
    return ordered


def _zone_candidate_points(topology: MapTopology, zone_id: int) -> list[MapPoint]:
    mask = (
        (topology.zone_labels == zone_id)
        & (topology.clearance_m >= topology.config.robot_clearance_m)
        & (
            topology.unknown_clearance_m
            >= topology.config.coverage_unknown_clearance_m
        )
    )
    rows, columns = np.nonzero(mask)
    stride = max(1, int(round(topology.config.candidate_spacing_m / topology.resolution_m)))
    selected = list(zip(rows[::stride], columns[::stride]))
    return [topology.cell_to_point(int(row), int(column)) for row, column in selected]


def _robust_range_fit_cost(candidates: Sequence[MapPoint], evidence: Sequence[EvidenceWaypoint], delta_sigma: float) -> float:
    candidate_xy = np.array([(point.x_m, point.y_m) for point in candidates])
    evidence_xy = np.array([(item.pose.x_m, item.pose.y_m) for item in evidence])
    ranges = np.array([item.range_m for item in evidence])
    sigma = np.maximum(np.array([item.range_sigma_m for item in evidence]), 0.12)
    predicted = np.hypot(candidate_xy[:, None, 0] - evidence_xy[None, :, 0], candidate_xy[:, None, 1] - evidence_xy[None, :, 1])
    scaled = (predicted - ranges[None, :]) / sigma[None, :]
    absolute = np.abs(scaled)
    loss = np.where(absolute <= delta_sigma, 0.5 * scaled**2, delta_sigma * (absolute - 0.5 * delta_sigma))
    return float(np.min(np.mean(loss, axis=1)))


def _short_range_proximity_cost(candidates: Sequence[MapPoint], evidence: Sequence[EvidenceWaypoint]) -> float:
    best = sorted(evidence, key=lambda item: item.range_m)[:min(3, len(evidence))]
    candidate_xy = np.array([(point.x_m, point.y_m) for point in candidates])
    best_xy = np.array([(item.pose.x_m, item.pose.y_m) for item in best])
    distance = np.hypot(candidate_xy[:, None, 0] - best_xy[None, :, 0], candidate_xy[:, None, 1] - best_xy[None, :, 1])
    return float(np.min(np.mean(distance, axis=1)))


def _zone_audio_evidence(topology: MapTopology, zone: Zone, evidence: Sequence[EvidenceWaypoint]) -> tuple[float | None, Portal | None]:
    portals = [portal for portal in topology.portals if zone.zone_id in (portal.zone_a, portal.zone_b)]
    best_level: float | None = None
    best_portal: Portal | None = None
    for portal in portals:
        levels = [
            float(packet_audio_level_dbfs(item))
            for item in evidence
            if (
                packet_audio_level_dbfs(item) is not None
                and item.packet_valid_packets >= topology.config.min_packet_valid_packets
                and _distance(item.packet_pose or item.pose, portal.point)
                <= topology.config.portal_evidence_radius_m
            )
        ]
        if levels:
            level = float(np.median(levels))
            if best_level is None or level > best_level:
                best_level, best_portal = level, portal
    if best_level is not None:
        return best_level, best_portal
    # Open plans may not have a narrow map portal.  Keep the same conservative
    # packet gate and use the strongest sample assigned to this zone instead.
    levels = [
        float(packet_audio_level_dbfs(item))
        for item in evidence
        if (
            packet_audio_level_dbfs(item) is not None
            and item.packet_valid_packets >= topology.config.min_packet_valid_packets
            and topology.zone_for_point(item.packet_pose or item.pose) == zone.zone_id
        )
    ]
    return (float(np.median(levels)), None) if levels else (None, None)


def _zone_doa_fit_cost(
    topology: MapTopology,
    candidates: Sequence[MapPoint],
    observations: Sequence[DoaObservation],
) -> tuple[float | None, int]:
    """Return a zone's best multi-pose acoustic-bearing fit.

    This deliberately asks one *map candidate* to satisfy every observation.
    A reflected bearing that happens to intersect a single room is therefore
    not enough; it must agree with a second measurement from a different
    robot pose.  If direct line of sight is required, candidates separated
    from a microphone by wall/unknown cells are discarded before the angular
    likelihood is calculated.
    """

    required = topology.config.doa_min_observations
    if len(observations) < required or not candidates:
        return None, 0

    usable: list[DoaObservation] = [
        observation
        for observation in observations
        if (
            math.isfinite(observation.microphone_pose.x_m)
            and math.isfinite(observation.microphone_pose.y_m)
            and math.isfinite(observation.bearing_rad)
            and math.isfinite(observation.confidence_weight)
            and observation.confidence_weight > 0.0
        )
    ]
    if len(usable) < required:
        return None, 0

    best_cost: float | None = None
    best_count = 0
    sigma = topology.config.doa_sigma_rad
    for candidate in candidates:
        errors: list[float] = []
        for observation in usable:
            if (
                topology.config.doa_require_line_of_sight
                and not _line_of_sight_free(
                    topology,
                    observation.microphone_pose,
                    candidate,
                )
            ):
                # A direct-bearing likelihood must not cross a mapped wall
                # or unobserved gap.  The observation remains in diagnostics,
                # but provides no direct support to this candidate.
                errors = []
                break
            predicted = math.atan2(
                candidate.y_m - observation.microphone_pose.y_m,
                candidate.x_m - observation.microphone_pose.x_m,
            )
            errors.append(_wrap_angle_rad(predicted - observation.bearing_rad))

        if len(errors) < required:
            continue
        scaled = np.asarray(errors, dtype=float) / sigma
        absolute = np.abs(scaled)
        delta = topology.config.huber_delta_sigma
        loss = np.where(
            absolute <= delta,
            0.5 * scaled**2,
            delta * (absolute - 0.5 * delta),
        )
        cost = float(np.average(
            loss,
            weights=np.asarray(
                [observation.confidence_weight for observation in usable],
                dtype=float,
            ),
        ))
        if best_cost is None or cost < best_cost:
            best_cost, best_count = cost, len(errors)
    return best_cost, best_count


def _line_of_sight_free(
    topology: MapTopology,
    start: MapPoint,
    end: MapPoint,
) -> bool:
    """Check observed free cells along a proposed direct acoustic path."""

    distance_m = _distance(start, end)
    # Include both endpoints.  Use half-cell sampling so a one-cell wall
    # cannot be stepped over on a diagonal.
    steps = max(1, int(math.ceil(distance_m / (topology.resolution_m * 0.5))))
    for index in range(steps + 1):
        fraction = index / steps
        point = MapPoint(
            start.x_m + (end.x_m - start.x_m) * fraction,
            start.y_m + (end.y_m - start.y_m) * fraction,
        )
        cell = topology.point_to_cell(point)
        if cell is None:
            return False
        row, column = cell
        if not topology.free[row, column]:
            return False
    return True


def _wrap_angle_rad(angle: float) -> float:
    """Normalize an angle to [-pi, +pi) without importing ROS helpers."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _inverse_normalized(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if math.isclose(low, high, abs_tol=1e-12):
        return {key: 1.0 for key in values}
    return {key: (high - value) / (high - low) for key, value in values.items()}


def _normalized(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if math.isclose(low, high, abs_tol=1e-12):
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def _entry_poses(topology: MapTopology, zone: Zone, portal: Portal | None) -> tuple[MapPoint | None, MapPoint | None]:
    if portal is None:
        return None, zone.interior if topology.safe(zone.interior) else None
    dx = zone.interior.x_m - portal.point.x_m
    dy = zone.interior.y_m - portal.point.y_m
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return None, zone.interior if topology.safe(zone.interior) else None
    ux, uy = dx / norm, dy / norm
    staging = MapPoint(portal.point.x_m - ux * topology.config.entry_staging_distance_m, portal.point.y_m - uy * topology.config.entry_staging_distance_m, math.atan2(uy, ux))
    interior = MapPoint(portal.point.x_m + ux * topology.config.entry_interior_distance_m, portal.point.y_m + uy * topology.config.entry_interior_distance_m, math.atan2(uy, ux))
    interior_cell = topology.point_to_cell(interior)
    interior_is_open = (
        interior_cell is not None
        and topology.safe(interior)
        and topology.clearance_m[interior_cell]
        >= topology.config.zone_clearance_m
    )
    # A fixed metric step through a doorway can still stop in a narrow entry
    # corridor. Continue to the map-derived maximum-clearance zone interior
    # so the original DOA/YOLO stage begins in genuinely open mapped space.
    if not interior_is_open:
        interior = zone.interior
    if not topology.safe(interior):
        return None, None
    return (staging if topology.safe(staging) else None), interior


def _distance(first: MapPoint, second: MapPoint) -> float:
    return math.hypot(first.x_m - second.x_m, first.y_m - second.y_m)
