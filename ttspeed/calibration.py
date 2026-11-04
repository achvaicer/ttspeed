"""Mapping from image pixels to real-world metres.

Two models are available:

* ``TableCalibration`` recovers the camera pose from the four table corners and
  intersects each pixel ray with the vertical plane through the table's long
  centre line. This corrects perspective and is the recommended model.
* ``ScaleCalibration`` applies one pixels-per-metre factor. It is only valid when
  the camera looks perpendicular to the flight plane.

Both return world coordinates as ``(x, z)``: ``x`` runs along the ball's main
direction of travel and ``z`` is the height.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

TABLE_LENGTH_M = 2.74
TABLE_WIDTH_M = 1.525
BALL_DIAMETER_M = 0.040


class CalibrationError(ValueError):
    """Raised when the supplied geometry cannot produce a usable calibration."""


@dataclass
class ScaleCalibration:
    """Uniform scale calibration."""

    px_per_meter: float
    method: str = "scale"

    def to_world(self, points_px: np.ndarray) -> np.ndarray:
        points = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
        world = points / self.px_per_meter
        world[:, 1] = -world[:, 1]  # Image y grows downwards, height grows upwards.
        return world

    def px_per_meter_at(self, point_px: Sequence[float]) -> float:
        return self.px_per_meter

    def mean_px_per_meter(self) -> float:
        return self.px_per_meter

    def warnings(self) -> List[str]:
        return []

    def describe(self) -> dict:
        return {"method": self.method, "px_per_meter": round(self.px_per_meter, 2)}


class TableCalibration:
    """Camera-pose calibration from the four corners of the table.

    ``corners`` must follow the table perimeter, with the first two points forming
    a long (2.74 m) edge.
    """

    def __init__(
        self,
        corners: Sequence[Sequence[float]],
        frame_size: Tuple[int, int],
        focal_px: Optional[float] = None,
        method: str = "table-corners",
    ) -> None:
        self.corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        self.frame_size = frame_size
        self.method = method
        width, height = frame_size
        self.principal_point = np.array([width / 2.0, height / 2.0])

        self.focal_px = focal_px or estimate_focal_from_corners(self.corners, frame_size)
        self.focal_was_estimated = focal_px is None
        self.camera_matrix = np.array(
            [
                [self.focal_px, 0.0, self.principal_point[0]],
                [0.0, self.focal_px, self.principal_point[1]],
                [0.0, 0.0, 1.0],
            ]
        )
        self.rotation, self.translation, self.reprojection_error = _solve_pose(
            self.corners, self.camera_matrix
        )
        self.camera_center = (-self.rotation.T @ self.translation).ravel()
        self._inverse_camera = np.linalg.inv(self.camera_matrix)

    def to_world(self, points_px: np.ndarray) -> np.ndarray:
        points = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
        homogeneous = np.column_stack([points, np.ones(len(points))])
        directions = (self.rotation.T @ self._inverse_camera @ homogeneous.T).T
        # Intersect each pixel ray with the plane y = 0 (the flight plane).
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = -self.camera_center[1] / directions[:, 1]
        world = self.camera_center + directions * scale[:, None]
        result = world[:, [0, 2]]
        result[(scale <= 0) | ~np.isfinite(scale)] = np.nan
        return result

    def px_per_meter_at(self, point_px: Sequence[float]) -> float:
        """Return the local image scale at a pixel on the flight plane."""
        world = self.to_world(np.array([point_px], dtype=np.float64))[0]
        if not np.all(np.isfinite(world)):
            return self.mean_px_per_meter()
        offset = 0.05
        base = np.array([[world[0], 0.0, world[1]], [world[0] + offset, 0.0, world[1]]])
        rvec, _ = cv2.Rodrigues(self.rotation)
        projected, _ = cv2.projectPoints(base, rvec, self.translation, self.camera_matrix, None)
        projected = projected.reshape(-1, 2)
        return float(np.linalg.norm(projected[1] - projected[0]) / offset)

    def mean_px_per_meter(self) -> float:
        center = self.project_table_point(0.0, 0.0, 0.0)
        return self.px_per_meter_at(center)

    def project_table_point(self, x: float, y: float, z: float) -> Tuple[float, float]:
        rvec, _ = cv2.Rodrigues(self.rotation)
        projected, _ = cv2.projectPoints(
            np.array([[x, y, z]], dtype=np.float64),
            rvec,
            self.translation,
            self.camera_matrix,
            None,
        )
        u, v = projected.reshape(2)
        return float(u), float(v)

    def warnings(self) -> List[str]:
        messages = []
        if self.reprojection_error > 0.01 * max(self.frame_size):
            messages.append(
                f"Table corners fit poorly (reprojection error "
                f"{self.reprojection_error:.1f} px). Check the corner coordinates."
            )
        lateral = abs(self.camera_center[1])
        along = abs(self.camera_center[0])
        if lateral < 1.0 or along > 2.5 * lateral:
            messages.append(
                "The camera is close to the table's end line. Speeds are unreliable "
                "from this angle; a view from the side of the table works best."
            )
        return messages

    def describe(self) -> dict:
        return {
            "method": self.method,
            "focal_px": round(float(self.focal_px), 1),
            "focal_estimated": self.focal_was_estimated,
            "reprojection_error_px": round(float(self.reprojection_error), 2),
            "camera_position_m": [round(float(v), 2) for v in self.camera_center],
            "corners_px": [[round(float(v), 1) for v in c] for c in self.corners],
        }


def _table_object_points(sign: float = 1.0) -> np.ndarray:
    half_length = TABLE_LENGTH_M / 2.0
    half_width = sign * TABLE_WIDTH_M / 2.0
    return np.array(
        [
            [-half_length, -half_width, 0.0],
            [half_length, -half_width, 0.0],
            [half_length, half_width, 0.0],
            [-half_length, half_width, 0.0],
        ],
        dtype=np.float64,
    )


def _solve_pose(corners: np.ndarray, camera_matrix: np.ndarray):
    """Return the pose with the camera above the table and the lowest error."""
    best = None
    for sign in (1.0, -1.0):
        object_points = _table_object_points(sign)
        count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points, corners, camera_matrix, None, flags=cv2.SOLVEPNP_IPPE
        )
        for index in range(count):
            rotation, _ = cv2.Rodrigues(rvecs[index])
            center = -rotation.T @ tvecs[index]
            if center[2, 0] <= 0:
                continue
            projected, _ = cv2.projectPoints(
                object_points, rvecs[index], tvecs[index], camera_matrix, None
            )
            error = float(np.linalg.norm(projected.reshape(-1, 2) - corners, axis=1).mean())
            if best is None or error < best[2]:
                best = (rotation, tvecs[index], error)
    if best is None:
        raise CalibrationError("No camera pose places the camera above the table.")
    return best


def estimate_focal_from_corners(
    corners: np.ndarray, frame_size: Tuple[int, int]
) -> float:
    """Estimate the focal length from the two vanishing points of the table.

    When an edge pair is almost parallel in the image the vanishing point is badly
    conditioned, so a typical phone-camera focal length is used instead.
    """
    width, height = frame_size
    fallback = 0.8 * max(width, height)
    center = np.array([width / 2.0, height / 2.0])
    points = [np.array([c[0], c[1], 1.0]) for c in corners]

    def vanishing(a, b, c, d):
        point = np.cross(np.cross(a, b), np.cross(c, d))
        if abs(point[2]) < 1e-9:
            return None
        return point[:2] / point[2]

    first = vanishing(points[0], points[1], points[3], points[2])
    second = vanishing(points[1], points[2], points[0], points[3])
    if first is None or second is None:
        return fallback
    limit = 6.0 * max(width, height)
    if np.linalg.norm(first - center) > limit or np.linalg.norm(second - center) > limit:
        return fallback
    squared = -float(np.dot(first - center, second - center))
    if squared <= 0:
        return fallback
    focal = float(np.sqrt(squared))
    if not 0.3 * max(width, height) <= focal <= 4.0 * max(width, height):
        return fallback
    return focal


def orient_corners(corners: np.ndarray, frame_size: Tuple[int, int]) -> np.ndarray:
    """Rotate perimeter-ordered corners so that the first edge is a long edge.

    Used for automatically detected corners, where the long edge is unknown. Both
    orderings are fitted and the one that best matches the table's aspect ratio wins.
    """
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    candidates = [corners, np.roll(corners, -1, axis=0)]
    scores = []
    for candidate in candidates:
        try:
            calibration = TableCalibration(candidate, frame_size)
            scores.append(calibration.reprojection_error)
        except (CalibrationError, cv2.error):
            scores.append(float("inf"))
    if abs(scores[0] - scores[1]) > 1.0:
        return candidates[int(np.argmin(scores))]
    # Fit quality does not separate them: the longer edge in pixels is the long edge.
    first_edge = np.linalg.norm(corners[0] - corners[1]) + np.linalg.norm(corners[2] - corners[3])
    second_edge = np.linalg.norm(corners[1] - corners[2]) + np.linalg.norm(corners[3] - corners[0])
    return candidates[0] if first_edge >= second_edge else candidates[1]


def detect_table_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    """Find the table as the largest blue or green quadrilateral in a frame."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    # Hue ranges cover the usual green and blue table tops (OpenCV hue is 0-179). The
    # saturation limit is low because glare can wash out part of the surface.
    mask = cv2.inRange(hsv, (35, 18, 35), (130, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= 0.01 * width * height]
    contours.sort(key=cv2.contourArea, reverse=True)
    best = None
    best_area = 0.0
    for seed in contours:
        # The net splits the table top in two, so join large regions of the same colour.
        # Hue separates the table from others, such as a green pool table beside it.
        group = [
            c
            for c in contours
            if cv2.contourArea(c) >= 0.1 * cv2.contourArea(seed)
            and abs(_median_hue(hsv, c) - _median_hue(hsv, seed)) <= 12
        ]
        points = np.vstack(group)
        area = float(sum(cv2.contourArea(c) for c in group))
        if area <= best_area:
            continue
        quad = _hull_to_quad(cv2.convexHull(points))
        if quad is None:
            continue
        quad_area = cv2.contourArea(quad.astype(np.float32))
        # A table fills its quadrilateral; large coloured walls or floors rarely do
        # and they tend to touch the frame border.
        solidity = area / max(quad_area, 1.0)
        touches_border = np.any(
            (quad[:, 0] < 3)
            | (quad[:, 0] > width - 4)
            | (quad[:, 1] < 3)
            | (quad[:, 1] > height - 4)
        )
        if solidity < 0.7 or touches_border:
            continue
        best, best_area = quad, area
    return best


def _median_hue(hsv: np.ndarray, contour: np.ndarray) -> float:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return float(np.median(hsv[:, :, 0][mask > 0]))


def _hull_to_quad(hull: np.ndarray) -> Optional[np.ndarray]:
    perimeter = cv2.arcLength(hull, True)
    for factor in np.linspace(0.005, 0.1, 40):
        approx = cv2.approxPolyDP(hull, factor * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float64)
        if len(approx) < 4:
            return None
    return None


def parse_points(text: str, expected: int) -> np.ndarray:
    """Parse ``"x1,y1;x2,y2;..."`` into an array of points."""
    try:
        points = [[float(v) for v in chunk.split(",")] for chunk in text.split(";")]
    except ValueError as error:
        raise CalibrationError(f"Could not parse points from '{text}'.") from error
    if len(points) != expected or any(len(point) != 2 for point in points):
        raise CalibrationError(
            f"Expected {expected} points formatted as 'x,y;x,y', got '{text}'."
        )
    return np.array(points, dtype=np.float64)
