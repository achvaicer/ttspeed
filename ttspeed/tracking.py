"""Link per-frame candidates into ball trajectories.

The whole video is available, so tracking runs offline: every plausible sequence of
candidates grows into a tracklet, implausible tracklets are discarded, and dynamic
programming selects the best set that does not overlap in time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from .calibration import BALL_DIAMETER_M
from .detection import Candidate
from .shots import trim_stray_ends


@dataclass
class TrackerConfig:
    max_speed_mps: float = 40.0
    min_speed_mps: float = 1.5
    max_gap_frames: int = 3
    min_points: int = 3
    min_path_m: float = 0.25
    min_straightness: float = 0.6
    min_consistent_steps: float = 0.7
    min_coverage: float = 0.7
    max_turn_cos: float = 0.3  # Consecutive steps may turn by about 70 degrees at most.
    max_active_tracklets: int = 400
    time_scale: float = 1.0  # Slow-motion factor: video seconds per real second.


@dataclass
class Tracklet:
    points: List[Candidate] = field(default_factory=list)

    @property
    def start_frame(self) -> int:
        return self.points[0].frame_index

    @property
    def end_frame(self) -> int:
        return self.points[-1].frame_index

    def last_pair(self) -> Tuple[int, int]:
        return id(self.points[-2]), id(self.points[-1])


def build_tracklets(
    frames: Sequence[Sequence[Candidate]],
    px_per_meter: float,
    config: TrackerConfig,
) -> List[Tracklet]:
    """Grow tracklets through ``frames``, which must be sorted in time."""
    ball_px = max(3.0, BALL_DIAMETER_M * px_per_meter)
    active: List[Tracklet] = []
    finished: List[Tracklet] = []
    recent: List[Sequence[Candidate]] = []

    for candidates in frames:
        if not candidates:
            recent.append(candidates)
            recent = recent[-(config.max_gap_frames + 1) :]
            continue
        current_frame = candidates[0].frame_index
        survivors: List[Tracklet] = []
        linked_pairs: Set[Tuple[int, int]] = set()

        for tracklet in active:
            if current_frame - tracklet.end_frame > config.max_gap_frames + 1:
                finished.append(tracklet)
                continue
            match = _best_continuation(tracklet, candidates, ball_px, px_per_meter, config)
            if match is not None:
                tracklet.points.append(match)
            survivors.append(tracklet)
            linked_pairs.add(tracklet.last_pair())

        # Any recent candidate can begin a new trajectory. This includes candidates
        # already used by a tracklet: the ball's position at a racket hit ends one
        # trajectory and starts the next.
        for previous_candidates in recent:
            for previous in previous_candidates:
                for candidate in candidates:
                    if (id(previous), id(candidate)) in linked_pairs:
                        continue
                    if _is_valid_seed(previous, candidate, px_per_meter, config):
                        survivors.append(Tracklet([previous, candidate]))

        active = _deduplicate(survivors, config.max_active_tracklets, finished)
        recent.append(candidates)
        recent = recent[-(config.max_gap_frames + 1) :]

    finished.extend(active)
    return [tracklet for tracklet in finished if len(tracklet.points) >= config.min_points]


def _best_continuation(
    tracklet: Tracklet,
    candidates: Sequence[Candidate],
    ball_px: float,
    px_per_meter: float,
    config: TrackerConfig,
):
    last, before = tracklet.points[-1], tracklet.points[-2]
    step_time = max(last.time_s - before.time_s, 1e-6)
    velocity_x = (last.x - before.x) / step_time
    velocity_y = (last.y - before.y) / step_time
    best, best_cost = None, float("inf")
    for candidate in candidates:
        elapsed = candidate.time_s - last.time_s
        if elapsed <= 0:
            continue
        predicted_x = last.x + velocity_x * elapsed
        predicted_y = last.y + velocity_y * elapsed
        distance = float(np.hypot(candidate.x - predicted_x, candidate.y - predicted_y))
        travel = float(np.hypot(velocity_x, velocity_y)) * elapsed
        # Drag, gravity and bounces change the velocity, more so over longer steps.
        gate = 2.5 * ball_px + 0.4 * travel
        if distance > gate:
            continue
        step_px = float(np.hypot(candidate.x - last.x, candidate.y - last.y))
        real_elapsed = elapsed / config.time_scale
        if step_px / px_per_meter / real_elapsed > config.max_speed_mps:
            continue
        cost = distance / gate - 0.25 * candidate.score
        if cost < best_cost:
            best, best_cost = candidate, cost
    return best


def _is_valid_seed(
    previous: Candidate, candidate: Candidate, px_per_meter: float, config: TrackerConfig
) -> bool:
    real_elapsed = (candidate.time_s - previous.time_s) / config.time_scale
    if real_elapsed <= 0:
        return False
    distance_m = float(np.hypot(candidate.x - previous.x, candidate.y - previous.y)) / px_per_meter
    speed = distance_m / real_elapsed
    # The scale is an average, and the ball can move mostly along the line of sight, so
    # a slow step in the image does not rule out a real shot.
    return 0.2 * config.min_speed_mps <= speed <= config.max_speed_mps


def _deduplicate(
    tracklets: List[Tracklet], limit: int, finished: List[Tracklet]
) -> List[Tracklet]:
    """Keep the longest tracklet per trailing pair, and at most ``limit`` overall."""
    best: Dict[Tuple[int, int], Tracklet] = {}
    for tracklet in tracklets:
        key = tracklet.last_pair()
        kept = best.get(key)
        if kept is None or len(tracklet.points) > len(kept.points):
            best[key] = tracklet
    unique = sorted(best.values(), key=lambda t: len(t.points), reverse=True)
    # Tracklets long enough to matter are saved instead of being silently dropped.
    for tracklet in unique[limit:]:
        if len(tracklet.points) >= 3:
            finished.append(tracklet)
    return unique[:limit]


def trim_tracklet(tracklet: Tracklet, world: np.ndarray, time_scale: float) -> np.ndarray:
    """Remove stray points from both ends of a tracklet, returning the kept world points."""
    times = np.array([p.time_s for p in tracklet.points]) / time_scale
    pixels = np.array([[p.x, p.y] for p in tracklet.points])
    start, end = trim_stray_ends(times, pixels)
    tracklet.points = tracklet.points[start:end]
    return world[start:end]


def tracklet_quality(
    tracklet: Tracklet,
    world: np.ndarray,
    expected_ball_px: float,
    config: TrackerConfig,
) -> float:
    """Return a positive score for a plausible ball trajectory, otherwise 0."""
    points = tracklet.points
    if len(points) < config.min_points or not np.all(np.isfinite(world)):
        return 0.0
    # World distances assume the centre line, which an oblique camera can get badly wrong,
    # so they are used only for generous limits. Path shape is judged in pixels.
    path = float(np.linalg.norm(np.diff(world, axis=0), axis=1).sum())
    duration = (points[-1].time_s - points[0].time_s) / config.time_scale
    if duration <= 0 or path / duration > 2.0 * config.max_speed_mps:
        return 0.0
    frames_spanned = points[-1].frame_index - points[0].frame_index + 1
    if len(points) / frames_spanned < config.min_coverage:
        return 0.0  # Sparse points can be joined into a path by chance.
    pixels = np.array([[p.x, p.y] for p in points])
    pixel_steps = np.linalg.norm(np.diff(pixels, axis=0), axis=1)
    straightness = float(np.linalg.norm(pixels[-1] - pixels[0])) / max(float(pixel_steps.sum()), 1e-6)
    if straightness < config.min_straightness:
        return 0.0
    if expected_ball_px > 0 and pixel_steps.sum() < 3.0 * expected_ball_px:
        return 0.0  # Barely moved: not a flight.
    if expected_ball_px > 0:
        minor = float(np.median([p.minor_axis for p in points]))
        # Blur and compression widen the blob, differencing can narrow it slightly.
        if not 0.35 * expected_ball_px - 1.0 <= minor <= 3.0 * expected_ball_px + 2.0:
            return 0.0
    # A flying ball changes speed gradually: erratic steps link unrelated blobs.
    times = np.array([p.time_s for p in points]) / config.time_scale
    step_speeds = pixel_steps / np.maximum(np.diff(times), 1e-6)
    ratios = step_speeds[1:] / np.maximum(step_speeds[:-1], 1e-6)
    if len(ratios) and ((ratios >= 0.4) & (ratios <= 2.5)).mean() < config.min_consistent_steps:
        return 0.0
    # Steps of a flight point in a similar direction, except briefly at a bounce.
    vectors = np.diff(pixels, axis=0)
    norms = np.maximum(pixel_steps, 1e-6)
    turns = (vectors[1:] * vectors[:-1]).sum(axis=1) / (norms[1:] * norms[:-1])
    if len(turns) and (turns >= config.max_turn_cos).mean() < 0.6:
        return 0.0
    if expected_ball_px > 0:
        # A blob can only be as long as the ball plus the ball's travel during the
        # exposure, which is at most one frame step. Limbs are longer than that.
        step_px = float(np.median(pixel_steps))
        major = float(np.median([p.major_axis for p in points]))
        if major > step_px + 2.5 * expected_ball_px + 2.0:
            return 0.0
    color = float(np.mean([p.score for p in points]))
    return len(points) * (0.35 + color) * straightness


def select_tracklets(
    scored: List[Tuple[Tracklet, float]], allowed_overlap_frames: int = 3
) -> List[Tracklet]:
    """Weighted interval scheduling: best-scoring tracklets not overlapping in time.

    Tracklets may overlap by a few frames, because the frames around a racket contact
    can fit both the incoming and the outgoing flight.
    """
    if not scored:
        return []
    items = sorted(scored, key=lambda item: item[0].end_frame)
    ends = [item[0].end_frame for item in items]
    best_score = [0.0] * (len(items) + 1)
    took = [False] * len(items)
    previous_index = [0] * len(items)
    for index, (tracklet, score) in enumerate(items):
        latest_allowed_end = tracklet.start_frame + allowed_overlap_frames
        previous_index[index] = int(np.searchsorted(ends, latest_allowed_end, side="right"))
        # Only earlier items are valid predecessors.
        previous_index[index] = min(previous_index[index], index)
        with_item = best_score[previous_index[index]] + score
        took[index] = with_item > best_score[index]
        best_score[index + 1] = max(with_item, best_score[index])

    selected: List[Tracklet] = []
    index = len(items) - 1
    while index >= 0:
        if took[index]:
            selected.append(items[index][0])
            index = previous_index[index] - 1
        else:
            index -= 1
    selected.reverse()
    return selected
