"""Per-frame detection of ball candidates.

A fixed camera is assumed. Each frame is compared with a rolling median background,
which keeps the ball's whole disc even when it moves less than its own diameter between
frames, as it does when flying towards or away from the camera. A motion test against
neighbouring frames then discards recent changes to the scene that are no longer moving.
Connected components are filtered by size and ranked by colour and size.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np


MIN_BLOB_WIDTH_PX = 3.5


@dataclass
class Candidate:
    frame_index: int
    time_s: float
    x: float  # Centre in full-resolution pixels.
    y: float
    minor_axis: float  # Blob width across the motion blur, full-resolution pixels.
    major_axis: float
    score: float  # 0 to 1: how much the blob's colour looks like a ball.


@dataclass
class DetectorConfig:
    ball_color: str = "auto"  # "auto", "white" or "orange".
    processing_width: int = 1280
    min_diameter_px: float = 2.0
    max_diameter_px: float = 60.0
    expected_diameter_px: Optional[float] = None  # Ranks candidates when known.
    max_candidates_per_frame: int = 20
    min_threshold: int = 10
    # Frames compared on each side of the current one. At high frame rates neighbouring
    # frames show the ball almost in the same place, and their difference vanishes.
    frame_stride: int = 1
    background_interval_frames: int = 15
    background_samples: int = 9


class BallDetector:
    """Detect moving, ball-sized blobs from a stream of frames."""

    def __init__(
        self,
        frame_size: Tuple[int, int],
        config: Optional[DetectorConfig] = None,
        initial_background: Optional[np.ndarray] = None,
    ):
        self.config = config or DetectorConfig()
        width, height = frame_size
        self.scale = min(1.0, self.config.processing_width / float(width))
        self.processing_size = (int(round(width * self.scale)), int(round(height * self.scale)))
        self._frames: List[Tuple[int, float, np.ndarray, np.ndarray]] = []
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._motion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self._samples: Deque[np.ndarray] = deque(maxlen=self.config.background_samples)
        self._background: Optional[np.ndarray] = None
        self._seen = 0
        if initial_background is not None:
            self._background = self._prepare_gray(initial_background)
        self._search_mask: Optional[np.ndarray] = None

    def _prepare_gray(self, frame: np.ndarray) -> np.ndarray:
        small = frame
        if self.scale < 1.0:
            small = cv2.resize(frame, self.processing_size, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)

    def _update_background(self, gray: np.ndarray) -> None:
        if self._seen % self.config.background_interval_frames == 0:
            self._samples.append(gray)
            # Until the window fills, a seed from the whole video is the better estimate.
            if len(self._samples) == self.config.background_samples or self._background is None:
                self._background = np.median(np.stack(self._samples), axis=0).astype(np.uint8)
        self._seen += 1

    def set_search_polygon(self, polygon: Optional[np.ndarray]) -> None:
        """Restrict detection to a polygon given in full-resolution pixels."""
        if polygon is None:
            self._search_mask = None
            return
        mask = np.zeros(self.processing_size[::-1], dtype=np.uint8)
        scaled = np.round(np.asarray(polygon) * self.scale).astype(np.int32)
        cv2.fillPoly(mask, [scaled], 255)
        self._search_mask = mask

    def process(self, index: int, time_s: float, frame: np.ndarray) -> List[Candidate]:
        """Add a frame; returns the candidates of the frame ``frame_stride`` steps earlier."""
        small = frame
        if self.scale < 1.0:
            small = cv2.resize(frame, self.processing_size, interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        self._update_background(gray)
        self._frames.append((index, time_s, small, gray))
        window = 2 * self.config.frame_stride + 1
        if len(self._frames) < window:
            return []
        if len(self._frames) > window:
            self._frames.pop(0)
        return self._detect_middle()

    def _detect_middle(self) -> List[Candidate]:
        stride = self.config.frame_stride
        before = self._frames[0][3]
        index, time_s, color, middle = self._frames[stride]
        after = self._frames[2 * stride][3]
        foreground = cv2.absdiff(middle, self._background)

        # Scale the threshold to the sensor noise, measured as the median difference.
        noise = float(np.median(foreground))
        threshold = max(self.config.min_threshold, int(4 * noise + 4))
        moving = cv2.threshold(foreground, threshold, 255, cv2.THRESH_BINARY)[1]

        changed = cv2.bitwise_or(
            cv2.threshold(cv2.absdiff(middle, before), threshold, 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(cv2.absdiff(after, middle), threshold, 255, cv2.THRESH_BINARY)[1],
        )
        # Widened because a slow ball only changes along its rim, not at its centre.
        moving = cv2.bitwise_and(moving, cv2.dilate(changed, self._motion_kernel))
        if self._search_mask is not None:
            moving = cv2.bitwise_and(moving, self._search_mask)
        moving = cv2.morphologyEx(moving, cv2.MORPH_OPEN, self._kernel)
        moving = cv2.morphologyEx(moving, cv2.MORPH_CLOSE, self._kernel)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(moving, connectivity=8)
        candidates = []
        hsv = None
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            box_w = stats[label, cv2.CC_STAT_WIDTH]
            box_h = stats[label, cv2.CC_STAT_HEIGHT]
            if area < 4 or box_w * box_h < 6:
                continue
            # Cheap rejection of body-sized blobs before the costlier ellipse fit.
            longest_allowed = self.config.max_diameter_px * self.scale * 8
            if max(box_w, box_h) > longest_allowed:
                continue
            x0, y0 = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP]
            region = labels[y0 : y0 + box_h, x0 : x0 + box_w] == label
            ys, xs = np.nonzero(region)
            minor, major = _principal_axes(xs, ys)
            minor_full = minor / self.scale
            major_full = major / self.scale
            if not self.config.min_diameter_px <= minor_full <= self.config.max_diameter_px:
                continue
            if minor < MIN_BLOB_WIDTH_PX:
                continue  # One-pixel-wide outlines of moving bodies and compression noise.
            fill = area / max(np.pi / 4.0 * minor * major, 1.0)
            if fill < 0.4:  # Thin, irregular outlines of limbs or clothing.
                continue
            if hsv is None:
                hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
            pixels = hsv[y0 + ys, x0 + xs]
            candidates.append(
                Candidate(
                    frame_index=index,
                    time_s=time_s,
                    x=(x0 + float(xs.mean()) + 0.5) / self.scale - 0.5,
                    y=(y0 + float(ys.mean()) + 0.5) / self.scale - 0.5,
                    minor_axis=float(minor_full),
                    major_axis=float(major_full),
                    score=_color_score(pixels, self.config.ball_color),
                )
            )
        candidates.sort(key=self._rank, reverse=True)
        return candidates[: self.config.max_candidates_per_frame]

    def _rank(self, candidate: Candidate) -> float:
        expected = self.config.expected_diameter_px
        if not expected:
            return candidate.score
        mismatch = (candidate.minor_axis - expected) / expected
        return candidate.score * float(np.exp(-mismatch * mismatch))


def _principal_axes(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float]:
    """Return the blob's full (minor, major) extent from its second moments."""
    if len(xs) < 3:
        return 1.0, 1.0
    covariance = np.cov(np.stack([xs, ys]).astype(np.float64))
    eigenvalues = np.linalg.eigvalsh(covariance)
    # For a uniform ellipse the variance along an axis equals (axis / 4) ** 2.
    # The pixel-grid term keeps one-pixel-wide blobs from collapsing to zero.
    minor = 4.0 * np.sqrt(max(eigenvalues[0], 0.0) + 1.0 / 12.0)
    major = 4.0 * np.sqrt(max(eigenvalues[1], 0.0) + 1.0 / 12.0)
    return float(minor), float(major)


def _color_score(hsv_pixels: np.ndarray, ball_color: str) -> float:
    """Score how ball-like a set of HSV pixels is, between 0 and 1.

    Motion blur mixes the ball with the background, so this is a soft preference used
    to rank tracks and not a hard filter.
    """
    hue = hsv_pixels[:, 0].astype(np.float64)
    saturation = hsv_pixels[:, 1].astype(np.float64) / 255.0
    value = hsv_pixels[:, 2].astype(np.float64) / 255.0
    white = float(np.mean(value * (1.0 - saturation)))
    orange_hue = (hue >= 3) & (hue <= 28)
    orange = float(np.mean(orange_hue * saturation * value))
    if ball_color == "white":
        return min(1.0, white * 1.3)
    if ball_color == "orange":
        return min(1.0, orange * 1.5)
    return min(1.0, max(white * 1.3, orange * 1.5))
