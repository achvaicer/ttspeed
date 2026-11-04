"""Split a ball trajectory into shots and estimate the speed of each one.

A shot runs from one racket contact to the next. A contact reverses the ball's motion
along the table, while a bounce only reverses its vertical motion, so a bounce stays
inside the shot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

MPS_TO_KMH = 3.6


@dataclass
class TrackPoint:
    frame_index: int
    time_s: float  # Video time.
    x_px: float
    y_px: float
    x_m: float
    z_m: float
    speed_mps: float = 0.0
    shot_id: int = 0
    y_m: float = 0.0  # Sideways position, known only when a flight model is used.


@dataclass
class Shot:
    shot_id: int
    rally_id: int
    start_time_s: float
    end_time_s: float
    start_frame: int
    end_frame: int
    peak_time_s: float
    peak_speed_mps: float
    mean_speed_mps: float
    initial_speed_mps: float
    direction: str
    bounces: int
    points: List[TrackPoint] = field(default_factory=list)
    confidence: str = "low"


@dataclass
class ShotConfig:
    fps: float = 30.0  # Frames per real second, so slow motion is already included.
    time_scale: float = 1.0
    min_points: int = 3
    min_speed_mps: float = 1.5
    min_path_m: float = 0.15
    max_speed_mps: float = 40.0  # Faster steps join unrelated detections.
    min_table_overlap_m: float = 0.3
    max_gap_s: float = 0.2  # Longer tracking gaps (in real time) split a shot.
    rally_gap_s: float = 1.5  # Longer pauses (in real time) start a new rally.
    table_half_length_m: Optional[float] = None  # Enables the crosses-the-table check.
    table_height_tolerance_m: float = 0.25
    min_coverage: float = 0.6  # Share of the shot's frames where the ball was found.
    flight_model: Optional[object] = None  # A flight.FlightModel, if the table is known.


def trim_stray_ends(times_s: np.ndarray, pixels: np.ndarray) -> Tuple[int, int]:
    """Return the ``[start, end)`` range left after dropping stray points at both ends.

    A trajectory can begin or end on an unrelated blob, and the step reaching that blob
    has a very different speed from its neighbour, or points against it. The frames
    touching a racket contact are dropped in the same way, as their steps mix two flights.

    This works in pixels and compares neighbouring steps only: perspective makes the
    ball's speed in the image vary several times over between the two ends of the table.
    """
    start, end = 0, len(times_s)
    while end - start >= 3:
        steps = np.diff(pixels[start:end], axis=0)
        speeds = np.linalg.norm(steps, axis=1) / np.maximum(np.diff(times_s[start:end]), 1e-6)
        if _is_stray_step(speeds[0], speeds[1], steps[0], steps[1]):
            start += 1
        elif _is_stray_step(speeds[-1], speeds[-2], steps[-1], steps[-2]):
            end -= 1
        else:
            break
    return start, end


def _is_stray_step(speed, neighbor_speed, step, neighbor_step) -> bool:
    return (
        not 0.4 * neighbor_speed <= speed <= 2.2 * neighbor_speed
        or float(np.dot(step, neighbor_step)) < 0.0
    )


def segment_shots(points: Sequence[TrackPoint], config: ShotConfig) -> List[Shot]:
    """Build shots from time-sorted points, assigning their speeds and ids."""
    shots: List[Shot] = []
    last_end_time: Optional[float] = None
    rally_id = 0
    for chunk in _split_on_gaps(points, config):
        for piece in _split_on_reversals(chunk, config):
            shot = _build_shot(piece, config)
            if shot is None:
                continue
            gap = None
            if last_end_time is not None:
                gap = (shot.start_time_s - last_end_time) / config.time_scale
            if gap is None or gap > config.rally_gap_s:
                rally_id += 1
            shot.shot_id = len(shots) + 1
            shot.rally_id = rally_id
            for point in shot.points:
                point.shot_id = shot.shot_id
            shots.append(shot)
            last_end_time = shot.end_time_s
    return shots


def _split_on_gaps(points: Sequence[TrackPoint], config: ShotConfig) -> List[List[TrackPoint]]:
    chunks: List[List[TrackPoint]] = []
    for point in points:
        if chunks:
            previous = chunks[-1][-1]
            gap = (point.time_s - previous.time_s) / config.time_scale
            step_m = float(np.hypot(point.x_m - previous.x_m, point.z_m - previous.z_m))
            speed = step_m / max(gap, 1e-6)
            # Generous, since this assumes the centre line that the flight model replaces.
            if gap <= config.max_gap_s and speed <= 2.0 * config.max_speed_mps:
                # The ball must be where its flight predicts. Otherwise this is a racket
                # contact, which the split handles, or an unrelated detection.
                if _continues_flight(chunks[-1], point, config):
                    chunks[-1].append(point)
                    continue
        chunks.append([point])
    return chunks


def _continues_flight(chunk: List[TrackPoint], point: TrackPoint, config: ShotConfig) -> bool:
    """Check, in pixels, that a point lies where the flight so far predicts."""
    if len(chunk) < 2:
        return True
    last, before = chunk[-1], chunk[-2]
    step_time = max(last.time_s - before.time_s, 1e-6)
    elapsed = point.time_s - last.time_s
    velocity = np.array([last.x_px - before.x_px, last.y_px - before.y_px]) / step_time
    predicted = np.array([last.x_px, last.y_px]) + velocity * elapsed
    error = float(np.hypot(point.x_px - predicted[0], point.y_px - predicted[1]))
    travel = float(np.linalg.norm(velocity)) * elapsed
    if error <= 0.4 * travel + 15.0:
        return True
    # A bounce turns a fall in the image into a rise, at a broadly similar pace.
    step = np.array([point.x_px - last.x_px, point.y_px - last.y_px])
    pace = float(np.linalg.norm(step)) / max(travel, 1e-6)
    return velocity[1] > 0 and step[1] < 0 and 0.3 <= pace <= 2.5


def reversal_indices(x_m: np.ndarray, times_s: np.ndarray) -> List[int]:
    """Return the indices where motion along the table changes sign beyond noise."""
    if len(x_m) < 3:
        return []
    velocity = np.diff(x_m) / np.maximum(np.diff(times_s), 1e-6)

    # Near-zero steps inherit the previous direction instead of creating a reversal.
    signs = np.zeros(len(velocity), dtype=int)
    current = 0
    for index, value in enumerate(velocity):
        if abs(value) >= 0.5:
            current = 1 if value > 0 else -1
        signs[index] = current
    for index in range(len(signs) - 1, 0, -1):
        if signs[index - 1] == 0:
            signs[index - 1] = signs[index]

    # A one-step flip with a tiny displacement is centroid jitter.
    steps = np.diff(x_m)
    for index in range(1, len(signs) - 1):
        if (
            signs[index] != signs[index - 1]
            and signs[index + 1] == signs[index - 1]
            and abs(steps[index]) < 0.08
        ):
            signs[index] = signs[index - 1]

    # Point ``index`` is the frame closest to the contact and opens the new flight.
    return [
        index
        for index in range(1, len(signs))
        if signs[index] != signs[index - 1] and signs[index] != 0 and signs[index - 1] != 0
    ]


def _split_on_reversals(
    points: List[TrackPoint], config: ShotConfig
) -> List[List[TrackPoint]]:
    x = np.array([p.x_m for p in points])
    t = np.array([p.time_s for p in points]) / config.time_scale
    pieces: List[List[TrackPoint]] = []
    start = 0
    for index in reversal_indices(x, t):
        pieces.append(points[start:index])
        start = index
    pieces.append(points[start:])
    return pieces


def _find_bounces(points: Sequence[TrackPoint], config: ShotConfig) -> List[int]:
    """Return indices where the ball's height stops falling and starts rising."""
    if len(points) < 4:
        return []
    z = np.array([p.z_m for p in points])
    t = np.array([p.time_s for p in points]) / config.time_scale
    bounces = []
    window = max(1, int(round(0.02 * config.fps)))
    for index in range(1, len(points) - 1):
        left = max(0, index - window)
        right = min(len(points) - 1, index + window)
        rate_in = (z[index] - z[left]) / max(t[index] - t[left], 1e-6)
        rate_out = (z[right] - z[index]) / max(t[right] - t[index], 1e-6)
        if rate_in >= -0.5 or rate_out <= 0.5:
            continue
        if z[index] != z[left : right + 1].min():
            continue
        if (
            config.table_half_length_m is not None
            and abs(z[index]) > config.table_height_tolerance_m
        ):
            continue
        bounces.append(index)
    return bounces


def estimate_speeds(
    times_s: np.ndarray, positions: np.ndarray, half_window: int
) -> np.ndarray:
    """Estimate speed at each point with a sliding least-squares polynomial fit.

    Timestamps may be irregular. Windows with five or more points use a quadratic,
    which follows the ball's slowing down; smaller ones use a straight line.
    """
    count = len(times_s)
    speeds = np.zeros(count)
    if count < 2:
        return speeds
    for index in range(count):
        low = max(0, index - half_window)
        high = min(count, index + half_window + 1)
        if high - low < 2:
            low, high = max(0, index - 1), min(count, index + 2)
        window_t = times_s[low:high] - times_s[index]
        degree = 2 if high - low >= 5 else 1
        design = np.vander(window_t, degree + 1, increasing=True)
        coefficients, *_ = np.linalg.lstsq(design, positions[low:high], rcond=None)
        speeds[index] = float(np.linalg.norm(coefficients[1]))
    return speeds


def _build_shot(points: List[TrackPoint], config: ShotConfig) -> Optional[Shot]:
    # The first frame is the contact itself: the racket and the differencing of two
    # flights displace the blob's centre, which would inflate the peak speed.
    if len(points) > config.min_points + 1:
        points = points[1:]
    if len(points) < config.min_points:
        return None
    times = np.array([p.time_s for p in points]) / config.time_scale
    world = np.array([[p.x_m, p.z_m] for p in points])
    start, end = trim_stray_ends(times, np.array([[p.x_px, p.y_px] for p in points]))
    points, times, world = points[start:end], times[start:end], world[start:end]
    if len(points) < config.min_points:
        return None
    path = float(np.linalg.norm(np.diff(world, axis=0), axis=1).sum())
    if path < config.min_path_m:
        return None
    if config.table_half_length_m is not None:
        # A real shot passes over the table. Players' hands and balls being collected
        # next to the table do not.
        overlap = min(world[:, 0].max(), config.table_half_length_m) - max(
            world[:, 0].min(), -config.table_half_length_m
        )
        if overlap < config.min_table_overlap_m:
            return None

    frames_spanned = points[-1].frame_index - points[0].frame_index + 1
    if len(points) / frames_spanned < config.min_coverage:
        return None  # A few detections joined by long gaps are not a tracked flight.

    confidence = "low"
    if config.flight_model is not None:
        pixels = np.array([[p.x_px, p.y_px] for p in points])
        fit = config.flight_model.fit(pixels, times)
        bounces = fit.bounces
        speeds = fit.speeds
        confidence = fit.confidence
        # Without a bounce, only crossing the net can show that this was a real shot.
        crosses_net = fit.positions[:, 0].min() < 0.0 < fit.positions[:, 0].max()
        if not bounces and not (crosses_net and len(points) >= 6):
            return None
        for point, position in zip(points, fit.positions):
            point.x_m, point.y_m, point.z_m = (float(v) for v in position)
        world = fit.positions
    else:
        # Fit each flight separately, since a bounce is a discontinuity in velocity.
        bounces = _find_bounces(points, config)
        half_window = max(1, int(round(0.02 * config.fps)))
        speeds = np.zeros(len(points))
        boundaries = [0] + bounces + [len(points) - 1]
        for first, last in zip(boundaries[:-1], boundaries[1:]):
            section = estimate_speeds(times[first : last + 1], world[first : last + 1], half_window)
            speeds[first : last + 1] = section

    for point, speed in zip(points, speeds):
        point.speed_mps = float(speed)
    peak_index = int(np.argmax(speeds))
    duration = float(times[-1] - times[0])
    # The mean of the fitted speeds, as a path through noisy points would read too long.
    area = float((0.5 * (speeds[1:] + speeds[:-1]) * np.diff(times)).sum())
    mean_speed = area / duration if duration > 0 else 0.0
    if speeds[peak_index] < config.min_speed_mps or speeds[peak_index] > config.max_speed_mps:
        return None
    return Shot(
        shot_id=0,
        rally_id=0,
        start_time_s=points[0].time_s,
        end_time_s=points[-1].time_s,
        start_frame=points[0].frame_index,
        end_frame=points[-1].frame_index,
        peak_time_s=points[peak_index].time_s,
        peak_speed_mps=float(speeds[peak_index]),
        mean_speed_mps=mean_speed,
        initial_speed_mps=float(speeds[: min(3, len(speeds))].mean()),
        direction="+x" if world[-1, 0] >= world[0, 0] else "-x",
        bounces=len(bounces),
        points=points,
        confidence=confidence,
    )
