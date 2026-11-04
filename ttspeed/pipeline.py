"""End-to-end analysis: calibrate, detect, track, segment, export and render."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import export
from .calibration import (
    BALL_DIAMETER_M,
    TABLE_LENGTH_M,
    CalibrationError,
    ScaleCalibration,
    TableCalibration,
    detect_table_corners,
    orient_corners,
    parse_points,
)
from .detection import BallDetector, Candidate, DetectorConfig
from .flight import FlightModel
from .render import OverlayRenderer
from .shots import Shot, ShotConfig, TrackPoint, reversal_indices, segment_shots
from .tracking import (
    Tracklet,
    TrackerConfig,
    build_tracklets,
    select_tracklets,
    tracklet_quality,
    trim_tracklet,
)
from .video import FrameWriter, VideoInfo, find_ffmpeg, iter_frames, open_video


BLOB_HALO_PX = 3.0


@dataclass
class Options:
    input_path: str
    output_dir: str = "output"
    table_corners: Optional[str] = None
    calib_line: Optional[str] = None
    calib_length_m: float = TABLE_LENGTH_M
    px_per_meter: Optional[float] = None
    focal_px: Optional[float] = None
    ball_color: str = "auto"
    slowmo: float = 1.0
    max_speed_kmh: float = 140.0
    min_speed_kmh: float = 10.0
    render_video: bool = True


@dataclass
class Result:
    shots: List[Shot]
    calibration: object
    warnings: List[str] = field(default_factory=list)
    output_paths: dict = field(default_factory=dict)


Progress = Callable[[str], None]


def reference_frame(path: str, samples: int = 15) -> np.ndarray:
    """Return the median of frames sampled across the video, which removes players."""
    capture, info = open_video(path)
    frames = []
    total = max(info.frame_count, 1)
    for position in np.linspace(0, max(total - 1, 0), samples).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"Could not read any frames from {path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def save_coordinate_grid(path: str, output_path: str) -> str:
    """Save a reference frame with a pixel grid, for reading calibration coordinates."""
    frame = reference_frame(path)
    height, width = frame.shape[:2]
    step = 100 if max(width, height) <= 2000 else 200
    scale = 0.4 if max(width, height) <= 2000 else 0.8
    for x in range(0, width, step):
        cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 1)
        cv2.putText(frame, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), 1, cv2.LINE_AA)
    for y in range(0, height, step):
        cv2.line(frame, (0, y), (width, y), (255, 255, 255), 1)
        cv2.putText(frame, str(y), (2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, frame)
    return output_path


def _manual_or_auto_calibration(options: Options, info: VideoInfo, warnings: List[str], log: Progress):
    """Pick a calibration in order of reliability. None means use the ball's size."""
    if options.table_corners:
        corners = parse_points(options.table_corners, 4)
        return TableCalibration(corners, info.size, options.focal_px)
    if options.calib_line:
        ends = parse_points(options.calib_line, 2)
        length_px = float(np.linalg.norm(ends[1] - ends[0]))
        if length_px <= 0 or options.calib_length_m <= 0:
            raise CalibrationError("The calibration line must have a positive length.")
        warnings.append(
            "Line calibration ignores perspective: speeds are only accurate when the "
            "camera looks perpendicular to the table's long side."
        )
        return ScaleCalibration(length_px / options.calib_length_m, method="line")
    if options.px_per_meter:
        warnings.append(
            "Fixed-scale calibration ignores perspective: speeds are only accurate when "
            "the camera looks perpendicular to the table's long side."
        )
        return ScaleCalibration(options.px_per_meter, method="px-per-meter")

    log("Looking for the table...")
    corners = detect_table_corners(reference_frame(options.input_path))
    if corners is not None:
        try:
            calibration = TableCalibration(
                orient_corners(corners, info.size),
                info.size,
                options.focal_px,
                method="table-auto",
            )
            # A wrong outline, such as half the table, cannot fit the table's proportions.
            if calibration.reprojection_error > 0.01 * max(info.size):
                raise CalibrationError(
                    f"the outline found does not fit a table "
                    f"(error {calibration.reprojection_error:.0f} px)"
                )
            warnings.append(
                "The table was detected automatically: check the outline in the "
                "calibration image. If it is wrong, pass --table-corners."
            )
            return calibration
        except (CalibrationError, cv2.error) as error:
            log(f"Automatic table calibration failed: {error}")
    warnings.append(
        "The table was not found, so the scale comes from the ball's apparent size. "
        "This can be far off: use --table-corners for trustworthy speeds."
    )
    return None


def _search_polygon(calibration) -> Optional[np.ndarray]:
    """Project the volume where the ball can be in play onto the image."""
    if not isinstance(calibration, TableCalibration):
        return None
    corners = np.array(
        [[x, y, z] for x in (-4.5, 4.5) for y in (-1.5, 1.5) for z in (-0.3, 3.0)],
        dtype=np.float64,
    )
    in_camera = (calibration.rotation @ corners.T + calibration.translation).T
    if np.any(in_camera[:, 2] <= 0.1):
        return None  # The volume reaches behind the camera, so its outline is invalid.
    rvec, _ = cv2.Rodrigues(calibration.rotation)
    projected, _ = cv2.projectPoints(
        corners, rvec, calibration.translation, calibration.camera_matrix, None
    )
    hull = cv2.convexHull(projected.astype(np.float32))
    return hull.reshape(-1, 2)


def detect_candidates(
    options: Options, info: VideoInfo, calibration, log: Progress
) -> Tuple[List[List[Candidate]], List[Tuple[int, float]]]:
    config = DetectorConfig(
        ball_color=options.ball_color,
        frame_stride=max(1, int(round(info.fps * options.slowmo / 60.0))),
    )
    if calibration is not None:
        expected = BALL_DIAMETER_M * calibration.mean_px_per_meter()
        config.min_diameter_px = max(1.5, 0.4 * expected)
        # Generous: the ball looks much larger than average when close to the camera.
        config.max_diameter_px = max(8.0, 4.0 * expected)
        config.expected_diameter_px = expected
    config.background_interval_frames = max(1, int(round(info.fps / 2.0)))
    detector = BallDetector(info.size, config, reference_frame(options.input_path))
    detector.set_search_polygon(_search_polygon(calibration))

    capture, _ = open_video(options.input_path)
    frames: List[List[Candidate]] = []
    timeline: List[Tuple[int, float]] = []
    for index, time_s, frame in iter_frames(capture, info.fps):
        timeline.append((index, time_s))
        candidates = detector.process(index, time_s, frame)
        if candidates:
            frames.append(candidates)
        if index and index % 300 == 0:
            log(f"Detection: frame {index}/{info.frame_count}")
    capture.release()
    return frames, timeline


def _min_points(options: Options, info: VideoInfo) -> int:
    """Return the point count equal to about 66 ms of flight, and never fewer than 4.

    Three points always look like a smooth path, so they cannot confirm a real flight.
    """
    return max(4, int(round(0.066 * info.fps * options.slowmo)))


def _tracker_config(options: Options, info: VideoInfo) -> TrackerConfig:
    return TrackerConfig(
        min_points=_min_points(options, info),
        max_speed_mps=options.max_speed_kmh / 3.6,
        min_speed_mps=options.min_speed_kmh / 3.6,
        time_scale=options.slowmo,
        # The tolerated gap stays constant in real time, whatever the frame rate.
        max_gap_frames=max(2, int(round(0.015 * info.fps * options.slowmo))),
    )


def track_ball(
    frames: Sequence[Sequence[Candidate]],
    calibration,
    config: TrackerConfig,
) -> List[Tracklet]:
    px_per_meter = calibration.mean_px_per_meter()
    is_table = isinstance(calibration, TableCalibration)
    expected_ball_px = BALL_DIAMETER_M * px_per_meter
    scored = []
    for whole in build_tracklets(frames, px_per_meter, config):
        pixels = np.array([[p.x, p.y] for p in whole.points])
        world = calibration.to_world(pixels)
        if not np.all(np.isfinite(world)):
            continue
        # At high frame rates the per-frame gate is wide enough for a tracklet to follow
        # the ball through a racket contact, so each flight is judged on its own.
        times = np.array([p.time_s for p in whole.points]) / config.time_scale
        bounds = [0] + reversal_indices(world[:, 0], times) + [len(world)]
        for start, end in zip(bounds[:-1], bounds[1:]):
            tracklet = Tracklet(whole.points[start:end])
            if len(tracklet.points) < config.min_points:
                continue
            piece = trim_tracklet(tracklet, world[start:end], config.time_scale)
            if is_table and np.any(
                (np.abs(piece[:, 0]) > 4.5) | (piece[:, 1] < -0.3) | (piece[:, 1] > 3.5)
            ):
                continue
            score = tracklet_quality(tracklet, piece, expected_ball_px, config)
            if score > 0:
                scored.append((tracklet, score))
    return select_tracklets(scored)


def _calibration_from_ball_size(
    frames: Sequence[Sequence[Candidate]],
    info: VideoInfo,
    config: TrackerConfig,
    warnings: List[str],
) -> ScaleCalibration:
    """Estimate the scale from the ball's diameter across its motion blur."""
    # Starting guess: the table spanning about 60% of the frame width.
    guess = ScaleCalibration(0.6 * info.width / TABLE_LENGTH_M, method="ball-size")
    tracklets = track_ball(frames, guess, config)
    # The longest tracks are the most certainly the ball.
    longest = max((len(t.points) for t in tracklets), default=0)
    minors = [
        point.minor_axis
        for tracklet in tracklets
        if len(tracklet.points) >= 0.5 * longest
        for point in tracklet.points
    ]
    if longest < 8:
        warnings.append(
            "Too few ball detections to measure its size: the scale is a rough guess "
            "that assumes the table spans 60% of the frame width."
        )
        return guess
    # Thresholding the blurred difference widens a blob by a few processing pixels.
    processing_scale = min(1.0, DetectorConfig().processing_width / float(info.width))
    diameter_px = max(2.0, float(np.median(minors)) - BLOB_HALO_PX / processing_scale)
    return ScaleCalibration(diameter_px / BALL_DIAMETER_M, method="ball-size")


def _to_points(tracklets: Sequence[Tracklet], calibration) -> List[TrackPoint]:
    points: List[TrackPoint] = []
    seen = set()
    for tracklet in tracklets:
        pixels = np.array([[p.x, p.y] for p in tracklet.points])
        world = calibration.to_world(pixels)
        for candidate, (x_m, z_m) in zip(tracklet.points, world):
            # Consecutive tracklets can share the frame of a racket contact.
            if candidate.frame_index in seen:
                continue
            seen.add(candidate.frame_index)
            points.append(
                TrackPoint(
                    frame_index=candidate.frame_index,
                    time_s=candidate.time_s,
                    x_px=candidate.x,
                    y_px=candidate.y,
                    x_m=float(x_m),
                    z_m=float(z_m),
                )
            )
    points.sort(key=lambda p: p.frame_index)
    return points


def _save_calibration_image(options: Options, calibration, path: str) -> None:
    frame = reference_frame(options.input_path)
    if isinstance(calibration, TableCalibration):
        polygon = np.round(calibration.corners).astype(np.int32)
        cv2.polylines(frame, [polygon], True, (0, 255, 255), 2, cv2.LINE_AA)
        for number, corner in enumerate(polygon, start=1):
            cv2.circle(frame, tuple(int(v) for v in corner), 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                frame, str(number), (int(corner[0]) + 8, int(corner[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
            )  # fmt: skip
        # Net line, to confirm that the long edge was identified correctly.
        net = [calibration.project_table_point(0, y, 0) for y in (-0.7625, 0.7625)]
        top = [calibration.project_table_point(0, y, 0.1525) for y in (-0.7625, 0.7625)]
        outline = np.round(np.array([net[0], net[1], top[1], top[0]])).astype(np.int32)
        cv2.polylines(frame, [outline], True, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, frame)


def _mean_frame_rate(timeline: Sequence[Tuple[int, float]], nominal: float) -> float:
    """Return the measured frame rate, which keeps variable-rate video in sync with audio."""
    if len(timeline) < 2 or timeline[-1][1] <= timeline[0][1]:
        return nominal
    return (len(timeline) - 1) / (timeline[-1][1] - timeline[0][1])


def _render(
    options: Options,
    info: VideoInfo,
    shots: Sequence[Shot],
    path: str,
    fps: float,
    warnings: List[str],
    log: Progress,
) -> None:
    capture, _ = open_video(options.input_path)
    renderer = OverlayRenderer(info.size, shots)
    writer = FrameWriter(path, info, fps, find_ffmpeg())
    warnings.extend(writer.warnings)
    try:
        for index, time_s, frame in iter_frames(capture, info.fps):
            writer.write(renderer.draw(frame, index, time_s))
            if index and index % 300 == 0:
                log(f"Rendering: frame {index}/{info.frame_count}")
    finally:
        writer.close()
        capture.release()


def run(options: Options, log: Progress = print) -> Result:
    if options.slowmo <= 0:
        raise ValueError("--slowmo must be greater than zero.")
    capture, info = open_video(options.input_path)
    capture.release()
    os.makedirs(options.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(options.input_path))[0]
    warnings: List[str] = []

    effective_fps = info.fps * options.slowmo
    log(
        f"Video: {info.width}x{info.height}, {info.fps:.2f} fps, "
        f"{info.frame_count} frames"
    )
    if effective_fps < 60:
        warnings.append(
            f"At {effective_fps:.0f} real frames per second a fast shot spans only a "
            "few frames, so peak speeds are coarse. Record at 120 fps or more if possible."
        )

    calibration = _manual_or_auto_calibration(options, info, warnings, log)
    log("Detecting the ball...")
    frames, timeline = detect_candidates(options, info, calibration, log)

    tracker_config = _tracker_config(options, info)
    if calibration is None:
        calibration = _calibration_from_ball_size(frames, info, tracker_config, warnings)
    warnings.extend(calibration.warnings())
    log(f"Calibration: {calibration.describe()}")

    log("Tracking the ball and splitting it into shots...")
    selected = track_ball(frames, calibration, tracker_config)
    points = _to_points(selected, calibration)
    shots = segment_shots(
        points,
        ShotConfig(
            fps=info.fps * options.slowmo,
            time_scale=options.slowmo,
            min_points=_min_points(options, info),
            min_speed_mps=options.min_speed_kmh / 3.6,
            max_speed_mps=options.max_speed_kmh / 3.6,
            table_half_length_m=(
                TABLE_LENGTH_M / 2.0 if isinstance(calibration, TableCalibration) else None
            ),
            # Without a table to check against, only a longer path can confirm a flight.
            min_path_m=0.15 if isinstance(calibration, TableCalibration) else 0.5,
            flight_model=(
                FlightModel(calibration, info.fps * options.slowmo)
                if isinstance(calibration, TableCalibration)
                else None
            ),
        ),
    )
    if not shots:
        warnings.append(
            "No shots were detected. The camera must stay still and the ball must be "
            "visible. Check the calibration image and try --ball-color."
        )

    paths = {
        "shots_csv": os.path.join(options.output_dir, f"{stem}_shots.csv"),
        "trajectory_csv": os.path.join(options.output_dir, f"{stem}_trajectory.csv"),
        "summary_json": os.path.join(options.output_dir, f"{stem}_summary.json"),
        "calibration_image": os.path.join(options.output_dir, f"{stem}_calibration.jpg"),
    }
    export.write_shots_csv(paths["shots_csv"], shots)
    export.write_trajectory_csv(paths["trajectory_csv"], shots)
    _save_calibration_image(options, calibration, paths["calibration_image"])

    if options.render_video:
        paths["video"] = os.path.join(options.output_dir, f"{stem}_speed.mp4")
        log("Rendering the output video...")
        fps = _mean_frame_rate(timeline, info.fps)
        _render(options, info, shots, paths["video"], fps, warnings, log)

    export.write_summary_json(
        paths["summary_json"],
        video={
            "path": options.input_path,
            "width": info.width,
            "height": info.height,
            "fps": round(info.fps, 3),
            "frames": info.frame_count,
            "slowmo_factor": options.slowmo,
        },
        calibration=calibration.describe(),
        warnings=warnings,
        shots=shots,
    )
    return Result(shots, calibration, warnings, paths)
