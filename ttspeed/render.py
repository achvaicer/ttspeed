"""Draw the speed overlay on video frames."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .shots import MPS_TO_KMH, Shot, TrackPoint

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GRAY = (190, 190, 190)
PANEL = (28, 28, 28)
TRAIL_FRAMES = 12


def speed_color(speed_kmh: float) -> Tuple[int, int, int]:
    """Map a speed to a BGR colour: green when slow, through yellow, red when fast."""
    ratio = float(np.clip(speed_kmh / 80.0, 0.0, 1.0))
    if ratio < 0.5:
        return (60, 220, int(60 + 390 * ratio))
    return (60, int(220 - 340 * (ratio - 0.5)), 255)


class OverlayRenderer:
    """Compose the ball trail and the speed panel for each frame."""

    def __init__(self, frame_size: Tuple[int, int], shots: Sequence[Shot]) -> None:
        self.width, self.height = frame_size
        self.unit = max(0.5, min(self.width, self.height) / 720.0)
        self.shots = list(shots)
        self.points_by_frame: Dict[int, TrackPoint] = {}
        for shot in self.shots:
            for point in shot.points:
                self.points_by_frame[point.frame_index] = point
        self.shot_by_id = {shot.shot_id: shot for shot in self.shots}
        # Estimates can be far off, so they only count when nothing better exists.
        trusted = [s for s in self.shots if s.confidence == "high"] or self.shots
        self.best_kmh = max((s.peak_speed_mps * MPS_TO_KMH for s in trusted), default=0.0)

    def draw(self, frame: np.ndarray, frame_index: int, time_s: float) -> np.ndarray:
        self._draw_trail(frame, frame_index)
        current = self.points_by_frame.get(frame_index)
        active = self._shot_in_progress(frame_index)
        last = self._last_finished_shot(frame_index)
        self._draw_panel(frame, time_s, current, active, last)
        return frame

    def _shot_in_progress(self, frame_index: int):
        for shot in self.shots:
            if shot.start_frame <= frame_index <= shot.end_frame:
                return shot
        return None

    def _last_finished_shot(self, frame_index: int):
        finished = None
        for shot in self.shots:
            if shot.end_frame < frame_index:
                finished = shot
            else:
                break
        return finished

    def _draw_trail(self, frame: np.ndarray, frame_index: int) -> None:
        trail: List[TrackPoint] = []
        for index in range(frame_index - TRAIL_FRAMES, frame_index + 1):
            point = self.points_by_frame.get(index)
            if point is not None:
                trail.append(point)
        # Never join points across a racket contact.
        if trail:
            trail = [p for p in trail if p.shot_id == trail[-1].shot_id]
        thickness_max = max(2, int(round(5 * self.unit)))
        for position, (start, end) in enumerate(zip(trail[:-1], trail[1:])):
            age = (position + 1) / max(1, len(trail) - 1)
            cv2.line(
                frame,
                (int(start.x_px), int(start.y_px)),
                (int(end.x_px), int(end.y_px)),
                speed_color(end.speed_mps * MPS_TO_KMH),
                max(1, int(round(thickness_max * age))),
                cv2.LINE_AA,
            )
        point = self.points_by_frame.get(frame_index)
        if point is None:
            return
        center = (int(point.x_px), int(point.y_px))
        color = speed_color(point.speed_mps * MPS_TO_KMH)
        cv2.circle(frame, center, int(round(11 * self.unit)), color, max(1, int(2 * self.unit)), cv2.LINE_AA)
        label = f"{point.speed_mps * MPS_TO_KMH:.0f} km/h"
        scale = 0.6 * self.unit
        (text_w, text_h), _ = cv2.getTextSize(label, FONT, scale, 2)
        x = int(np.clip(center[0] - text_w / 2, 4, self.width - text_w - 4))
        y = center[1] - int(18 * self.unit)
        if y - text_h < 4:
            y = center[1] + int(18 * self.unit) + text_h
        pad = int(4 * self.unit)
        cv2.rectangle(frame, (x - pad, y - text_h - pad), (x + text_w + pad, y + pad), PANEL, -1)
        _text(frame, label, (x, y), scale, WHITE, 2)

    def _draw_panel(self, frame, time_s, current, active, last) -> None:
        u = self.unit
        panel_w, panel_h = int(300 * u), int(158 * u)
        margin = int(16 * u)
        x0, y0 = margin, margin
        region = frame[y0 : y0 + panel_h, x0 : x0 + panel_w]
        region[:] = cv2.addWeighted(region, 0.3, np.full_like(region, PANEL), 0.7, 0)
        pad = int(14 * u)
        left = x0 + pad

        if active is not None:
            speed = (current.speed_mps if current else active.peak_speed_mps) * MPS_TO_KMH
            title = f"SHOT {active.shot_id}  LIVE{_confidence_note(active)}"
            footer = f"Peak so far: {self._peak_so_far(active, current):.0f} km/h"
        elif last is not None:
            speed = last.peak_speed_mps * MPS_TO_KMH
            title = f"SHOT {last.shot_id}  PEAK{_confidence_note(last)}"
            footer = f"Mean: {last.mean_speed_mps * MPS_TO_KMH:.0f} km/h"
        else:
            speed = None
            title = "WAITING FOR BALL"
            footer = ""

        _text(frame, title, (left, y0 + int(26 * u)), 0.5 * u, GRAY, 1)
        if speed is None:
            _text(frame, "--", (left, y0 + int(82 * u)), 1.6 * u, WHITE, 3)
        else:
            value = f"{speed:.0f}"
            (value_w, _), _ = cv2.getTextSize(value, FONT, 1.6 * u, 3)
            _text(frame, value, (left, y0 + int(82 * u)), 1.6 * u, speed_color(speed), 3)
            _text(frame, "km/h", (left + value_w + int(10 * u), y0 + int(82 * u)), 0.7 * u, WHITE, 2)
        if footer:
            _text(frame, footer, (left, y0 + int(110 * u)), 0.5 * u, WHITE, 1)
        summary = f"Best: {self.best_kmh:.0f} km/h   t={_format_time(time_s)}"
        _text(frame, summary, (left, y0 + int(138 * u)), 0.5 * u, GRAY, 1)

    @staticmethod
    def _peak_so_far(shot: Shot, current) -> float:
        limit = current.frame_index if current else shot.end_frame
        speeds = [p.speed_mps for p in shot.points if p.frame_index <= limit]
        return max(speeds, default=0.0) * MPS_TO_KMH


def _confidence_note(shot: Shot) -> str:
    return "  (ESTIMATE)" if shot.confidence == "low" else ""


def _text(frame, text, origin, scale, color, thickness) -> None:
    cv2.putText(frame, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def _format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:05.2f}"
