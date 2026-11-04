"""Reconstruct each shot's 3D flight from a single calibrated camera.

One camera cannot measure depth, so an assumption must fill the gap. Assuming that every
shot flies over the table's centre line fails from an oblique camera, where the pixel rays
cross that plane at a shallow angle and a small sideways offset becomes a large speed error.

Instead, each shot gets its own vertical flight plane, chosen by physics. For each candidate
plane the ball's path is fitted with gravity, air drag and a spin-induced acceleration, then
projected back onto the image. The plane with the smallest error in pixels wins, because a
wrong plane distorts the path into a motion that a real ball cannot follow.

A bounce adds the strongest constraint. The ball is on the table at that moment, so its
3D position is known and the plane must pass through it. Two bounces, as in a serve, leave
only one possible plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .calibration import TABLE_LENGTH_M, TABLE_WIDTH_M, TableCalibration

GRAVITY = 9.81
# Deceleration = DRAG_PER_M * speed^2, from a 40 mm, 2.7 g ball in air (Cd about 0.5).
DRAG_PER_M = 0.14
MAX_YAW_DEG = 32.0  # A diagonal from corner to corner of the table is about 29 degrees.
TABLE_MARGIN_M = 0.12
MAX_OFFSET_M = 0.9
# Extra downward acceleration: topspin is positive, and backspin is negative.
SPIN_ACCELERATIONS = (-2.0, 0.0, 3.0, 6.0)
MAX_GOOD_RESIDUAL_PX = 3.0


@dataclass
class FlightFit:
    positions: np.ndarray  # Rows of (x, y, z) in metres.
    speeds: np.ndarray  # Metres per second, for each point.
    bounces: List[int]
    confidence: str  # "high" when a bounce anchors the plane and the fit is good.
    yaw_deg: float
    residual_px: float


@dataclass
class _Anchor:
    """A bounce: the moment and table position where the ball touched down."""

    index: int
    time_s: float
    xy: np.ndarray


class FlightModel:
    """Fit flight planes and speeds using a table calibration."""

    def __init__(self, calibration: TableCalibration, fps: float) -> None:
        self.calibration = calibration
        self.fps = fps
        self.center = calibration.camera_center
        self.rotation = calibration.rotation
        self.translation = calibration.translation.ravel()
        self.camera_matrix = calibration.camera_matrix
        self._to_world = self.rotation.T @ np.linalg.inv(self.camera_matrix)

    def _rays(self, pixels: np.ndarray) -> np.ndarray:
        homogeneous = np.column_stack([pixels, np.ones(len(pixels))])
        return (self._to_world @ homogeneous.T).T

    def _project(self, positions: np.ndarray) -> np.ndarray:
        camera = self.rotation @ positions.T + self.translation[:, None]
        image = self.camera_matrix @ camera
        return (image[:2] / image[2]).T

    def _on_table(self, pixel: np.ndarray) -> Optional[np.ndarray]:
        """Return the (x, y) table point seen at a pixel, if it is on the table."""
        ray = self._rays(pixel.reshape(1, 2))[0]
        if abs(ray[2]) < 1e-9:
            return None
        scale = -self.center[2] / ray[2]
        if scale <= 0:
            return None
        point = self.center + scale * ray
        inside = (
            abs(point[0]) <= TABLE_LENGTH_M / 2 + TABLE_MARGIN_M
            and abs(point[1]) <= TABLE_WIDTH_M / 2 + TABLE_MARGIN_M
        )
        return point[:2] if inside else None

    def find_bounces(self, pixels: np.ndarray, times: np.ndarray) -> List[int]:
        """Find frames where the ball's vertical motion in the image abruptly changes.

        A bounce takes away downward speed at once, while free flight only changes it
        gradually. The ball need not rise in the image afterwards: one flying towards the
        camera keeps moving down the image after bouncing. Using pixels, the check does
        not depend on a flight plane.
        """
        count = len(pixels)
        window = max(2, int(round(0.05 * self.fps)))
        threshold = 1.5 * self.calibration.mean_px_per_meter()  # About 1.5 m/s.
        y = pixels[:, 1]
        jumps = np.zeros(count)
        for index in range(2, count - 2):
            before = np.arange(max(0, index - window), index)
            after = np.arange(index + 1, min(count, index + window + 1))
            rate_before = np.polyfit(times[before], y[before], 1)[0]
            rate_after = np.polyfit(times[after], y[after], 1)[0]
            jumps[index] = rate_after - rate_before

        bounces: List[int] = []
        for index in range(2, count - 2):
            low, high = max(0, index - window), min(count, index + window + 1)
            if jumps[index] > -threshold or jumps[index] != jumps[low:high].min():
                continue
            if self._on_table(pixels[index]) is None and self._anchor(index, pixels, times) is None:
                continue
            bounces.append(index)
        return bounces

    def _anchor(self, index: int, pixels: np.ndarray, times: np.ndarray) -> Optional[_Anchor]:
        """Locate a bounce where the arcs before and after it cross.

        The bounce happens between frames, and the nearest frame shows the ball slightly
        above the table. From a low camera that small height moves the apparent table
        position a long way, so the arcs are extrapolated to where they meet.
        """
        before = np.arange(max(0, index - 3), index)
        after = np.arange(index + 1, min(len(pixels), index + 4))
        touchdown, pixel = times[index], pixels[index]
        if len(before) >= 2 and len(after) >= 2:
            degree_before, degree_after = min(2, len(before) - 1), min(2, len(after) - 1)
            grid = np.linspace(times[before[-1]], times[after[0]], 41)
            y_before = np.polyval(np.polyfit(times[before], pixels[before, 1], degree_before), grid)
            y_after = np.polyval(np.polyfit(times[after], pixels[after, 1], degree_after), grid)
            best = int(np.argmin(np.abs(y_before - y_after)))
            x_before = np.polyval(np.polyfit(times[before], pixels[before, 0], degree_before), grid[best])
            x_after = np.polyval(np.polyfit(times[after], pixels[after, 0], degree_after), grid[best])
            touchdown = float(grid[best])
            pixel = np.array([(x_before + x_after) / 2.0, (y_before[best] + y_after[best]) / 2.0])
        xy = self._on_table(pixel)
        if xy is None:
            xy = self._on_table(pixels[index])
        return None if xy is None else _Anchor(index, float(touchdown), xy)

    def _intersect(self, rays: np.ndarray, yaw: float, offset: float) -> np.ndarray:
        """Intersect rays with the vertical plane y = offset + x * tan(yaw)."""
        normal = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        anchor = np.array([0.0, offset, 0.0])
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = ((anchor - self.center) @ normal) / (rays @ normal)
        positions = self.center + rays * scale[:, None]
        positions[(scale <= 0) | ~np.isfinite(scale)] = np.nan
        return positions

    def _sections(self, count: int, anchors: Sequence[_Anchor]) -> List[Tuple[np.ndarray, Optional[_Anchor], Optional[_Anchor]]]:
        """Split point indices into flights, leaving out each ambiguous bounce frame."""
        sections = []
        start = 0
        previous: Optional[_Anchor] = None
        for anchor in anchors:
            sections.append((np.arange(start, anchor.index), previous, anchor))
            start = anchor.index + 1
            previous = anchor
        sections.append((np.arange(start, count), previous, None))
        return [section for section in sections if len(section[0]) >= 1]

    def _model(self, positions, times, sections, yaw, spin) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Fit the physical flight in a plane. Returns the modelled points and speeds."""
        direction = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        offset_origin = positions[np.isfinite(positions[:, 0])]
        if len(offset_origin) == 0:
            return None
        lateral = np.array([-np.sin(yaw), np.cos(yaw), 0.0]) @ offset_origin[0]
        along = positions @ direction
        modelled = np.full_like(positions, np.nan)
        speeds = np.full(len(positions), np.nan)
        gravity = GRAVITY + spin

        for indices, start_anchor, end_anchor in sections:
            t_all = list(times[indices])
            s_all = list(along[indices])
            z_all = list(positions[indices, 2])
            weight = list(self._weights(positions, indices))
            # The bounce points are extra observations, as their height is known.
            for anchor in (start_anchor, end_anchor):
                if anchor is not None:
                    t_all.append(anchor.time_s)
                    s_all.append(float(anchor.xy @ direction[:2]))
                    z_all.append(0.02)
                    weight.append(max(weight) if weight else 1.0)
            if len(t_all) < 2:
                continue
            origin = t_all[0] if start_anchor is None else start_anchor.time_s
            t = np.array(t_all) - origin
            s = np.array(s_all)
            z = np.array(z_all)
            w = np.array(weight)

            # Quadratic drag gives v(t) = v0 / (1 + k v0 t), so the distance grows with
            # log(1 + k v0 t) / k. It depends on v0 itself, hence the repeated refinement.
            _, rate = _weighted_line(t, s, w)
            sign = 1.0 if rate >= 0 else -1.0
            start_speed = max(abs(rate), 0.01)
            base = 0.0
            for _ in range(3):
                distance = np.log1p(DRAG_PER_M * start_speed * t) / DRAG_PER_M
                base, ratio = _weighted_line(distance, sign * s, w)
                start_speed = max(start_speed * ratio, 0.01)
            height_base, height_rate = _weighted_line(t, z + 0.5 * gravity * t * t, w)

            point_t = times[indices] - origin
            drag_term = DRAG_PER_M * start_speed * point_t
            model_s = sign * (base + np.log1p(drag_term) / DRAG_PER_M)
            model_z = height_base + height_rate * point_t - 0.5 * gravity * point_t ** 2
            modelled[indices] = np.column_stack(
                [
                    model_s * direction[0] - lateral * np.sin(yaw),
                    model_s * direction[1] + lateral * np.cos(yaw),
                    model_z,
                ]
            )
            along_rate = start_speed / (1.0 + drag_term)
            speeds[indices] = np.hypot(along_rate, height_rate - gravity * point_t)
        return modelled, speeds

    def _weights(self, positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Trust points by how many pixels a metre covers there, along the flight."""
        if len(indices) < 2:
            return np.ones(len(indices))
        image = self._project(positions[indices])
        pixel_steps = np.linalg.norm(np.diff(image, axis=0), axis=1)
        metre_steps = np.linalg.norm(np.diff(positions[indices], axis=0), axis=1)
        scale = pixel_steps / np.maximum(metre_steps, 1e-3)
        scale = np.append(scale, scale[-1])
        return np.clip(scale, 5.0, 2000.0)

    def _cost(self, rays, pixels, times, sections, yaw, offset, spin) -> float:
        positions = self._intersect(rays, yaw, offset)
        if not np.all(np.isfinite(positions)):
            return float("inf")
        result = self._model(positions, times, sections, yaw, spin)
        if result is None:
            return float("inf")
        modelled, _ = result
        valid = np.all(np.isfinite(modelled), axis=1)
        if valid.sum() < 2:
            return float("inf")
        camera_z = self.rotation[2] @ modelled[valid].T + self.translation[2]
        if np.any(camera_z <= 0.05):
            return float("inf")
        errors = np.linalg.norm(self._project(modelled[valid]) - pixels[valid], axis=1)
        cost = float(np.sqrt((errors * errors).mean()))

        # Physical limits, in pixels per metre of violation.
        below = np.maximum(-positions[:, 2] - 0.05, 0.0).sum()
        over_table = np.abs(positions[:, 0]) <= TABLE_LENGTH_M / 2
        wide = 0.0
        if over_table.any():
            wide = np.maximum(np.abs(positions[over_table, 1]) - (TABLE_WIDTH_M / 2 + 0.5), 0.0).sum()
        # Preferences settle near-equal fits: straight, central shots with little spin.
        # From the side a change of plane barely moves the pixels, so these decide; from
        # an oblique camera a wrong plane costs far more pixels, so the data overrules them.
        return cost + 10.0 * float(below + wide) + 3.0 * abs(yaw) + 2.0 * abs(offset) + 0.02 * abs(spin)

    def fit(self, pixels: np.ndarray, times: np.ndarray) -> FlightFit:
        rays = self._rays(pixels)
        bounces = self.find_bounces(pixels, times)
        anchors = [a for a in (self._anchor(i, pixels, times) for i in bounces) if a is not None]
        sections = self._sections(len(pixels), anchors)
        yaws = np.radians(np.arange(-MAX_YAW_DEG, MAX_YAW_DEG + 0.1, 2.0))

        candidates: List[Tuple[float, float]] = []
        if len(anchors) >= 2:
            first, last = anchors[0].xy, anchors[-1].xy
            yaw = float(np.arctan((last[1] - first[1]) / (last[0] - first[0] or 1e-6)))
            if abs(yaw) <= np.radians(MAX_YAW_DEG):
                candidates = [(yaw, float(first[1] - first[0] * np.tan(yaw)))]
        if not candidates and anchors:
            xy = anchors[0].xy
            candidates = [(float(yaw), float(xy[1] - xy[0] * np.tan(yaw))) for yaw in yaws]
        if not candidates:
            candidates = [
                (float(yaw), float(offset))
                for yaw in yaws
                for offset in np.linspace(-MAX_OFFSET_M, MAX_OFFSET_M, 13)
            ]

        best, best_cost = (0.0, 0.0, 0.0), float("inf")
        for yaw, offset in candidates:
            for spin in SPIN_ACCELERATIONS:
                cost = self._cost(rays, pixels, times, sections, yaw, offset, spin)
                if cost < best_cost:
                    best, best_cost = (yaw, offset, spin), cost
        return self._finish(rays, pixels, times, sections, bounces, anchors, best)

    def _finish(self, rays, pixels, times, sections, bounces, anchors, best) -> FlightFit:
        yaw, offset, spin = best
        positions = self._intersect(rays, yaw, offset)
        result = self._model(positions, times, sections, yaw, spin)
        count = len(pixels)
        if result is None:
            return FlightFit(positions, np.zeros(count), bounces, "low", np.degrees(yaw), float("inf"))
        modelled, speeds = result
        valid = np.all(np.isfinite(modelled), axis=1)
        residual = float("inf")
        if valid.sum() >= 2:
            errors = np.linalg.norm(self._project(modelled[valid]) - pixels[valid], axis=1)
            residual = float(np.sqrt((errors * errors).mean()))
        # Bounce frames were left out of the fits, so they take their neighbour's speed.
        known = np.isfinite(speeds)
        if known.any():
            speeds = np.interp(np.arange(count), np.flatnonzero(known), speeds[known])
            modelled[~valid] = positions[~valid]
        else:
            speeds = np.zeros(count)
        confidence = "high" if anchors and residual <= MAX_GOOD_RESIDUAL_PX and count >= 8 else "low"
        return FlightFit(modelled, speeds, list(bounces), confidence, float(np.degrees(yaw)), residual)


def _weighted_line(t: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """Return the intercept and slope of a weighted least-squares line."""
    w2 = w * w
    total = w2.sum()
    mean_t = (w2 * t).sum() / total
    mean_y = (w2 * y).sum() / total
    spread = (w2 * (t - mean_t) ** 2).sum()
    if spread <= 1e-12:
        return float(mean_y), 0.0
    slope = (w2 * (t - mean_t) * (y - mean_y)).sum() / spread
    return float(mean_y - slope * mean_t), float(slope)
