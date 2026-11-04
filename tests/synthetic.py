"""Render synthetic training videos with a known ground truth.

A pinhole camera films a table, a net and two moving "players". Racket-to-racket
shots follow simulated ballistic flights with drag, gravity and a table bounce.
Each frame averages sub-frames to reproduce motion blur.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

TABLE_L = 2.74
TABLE_W = 1.525
BALL_RADIUS = 0.02
GRAVITY = 9.81
# Drag acceleration = DRAG * speed^2, a typical value for a 40 mm, 2.7 g ball.
DRAG = 0.13


@dataclass
class Scene:
    width: int = 1280
    height: int = 720
    fps: float = 60.0
    camera_position: Tuple[float, float, float] = (0.0, -4.0, 1.6)
    look_at: Tuple[float, float, float] = (0.0, 0.0, 0.2)
    focal_factor: float = 0.85  # Focal length as a fraction of the frame width.
    exposure_fraction: float = 0.5
    noise_sigma: float = 2.0
    ball_bgr: Tuple[int, int, int] = (245, 245, 245)
    lateral_offset: float = 0.0  # Ball distance from the table's centre line.
    speeds_mps: Sequence[float] = (8.0, 12.0, 16.0, 6.0, 20.0, 10.0)
    players: bool = True
    container_fps: Optional[float] = None  # Playback rate, to simulate slow motion.
    sub_steps: int = 8
    idle_seconds: float = 0.6  # Extra time without a ball at the end of the video.
    yaw_degrees: Sequence[float] = (0.0,)  # Shot angles to the table's axis, cycled through.
    magnus_accel: float = 0.0  # Downward acceleration from topspin, in m/s^2.
    table_bgr: Tuple[int, int, int] = (120, 70, 20)


class Camera:
    def __init__(self, scene: Scene) -> None:
        position = np.array(scene.camera_position, dtype=np.float64)
        forward = np.array(scene.look_at, dtype=np.float64) - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        self.rotation = np.stack([right, down, forward])
        self.position = position
        self.focal = scene.focal_factor * scene.width
        self.center = np.array([scene.width / 2.0, scene.height / 2.0])

    def project(self, points: np.ndarray) -> np.ndarray:
        camera = (self.rotation @ (np.atleast_2d(points) - self.position).T).T
        return self.focal * camera[:, :2] / camera[:, 2:3] + self.center

    def pixels_per_meter(self, point: np.ndarray) -> float:
        depth = float((self.rotation @ (np.asarray(point) - self.position))[2])
        return self.focal / depth


def simulate_shot(
    start_x: float,
    start_z: float,
    direction: int,
    speed: float,
    lateral: float,
    sub_dt: float,
    yaw_degrees: float = 0.0,
    magnus_accel: float = 0.0,
) -> np.ndarray:
    """Simulate one flight. Returns rows of (t, x, y, z, speed)."""
    position = np.array([start_x, lateral, start_z])
    # Launch nearly flat, aimed to bounce on the far half of the table.
    angle = np.radians(6.0 if speed > 10 else 14.0)
    yaw = np.radians(yaw_degrees)
    velocity = speed * np.array(
        [direction * np.cos(angle) * np.cos(yaw), np.cos(angle) * np.sin(yaw), np.sin(angle)]
    )
    rows = []
    time = 0.0
    while abs(position[0]) <= TABLE_L / 2.0 + 0.3 or direction * position[0] < 0:
        current = float(np.linalg.norm(velocity))
        rows.append([time, *position, current])
        velocity = velocity + sub_dt * (
            np.array([0.0, 0.0, -GRAVITY - magnus_accel]) - DRAG * current * velocity
        )
        position = position + sub_dt * velocity
        on_table = abs(position[0]) <= TABLE_L / 2.0 and abs(position[1]) <= TABLE_W / 2.0
        if position[2] < BALL_RADIUS and velocity[2] < 0 and on_table:
            position[2] = BALL_RADIUS
            velocity[2] *= -0.85
            velocity[0] *= 0.9
        time += sub_dt
        if time > 3.0:
            break
    return np.array(rows)


def _draw_scene(scene: Scene, camera: Camera) -> np.ndarray:
    frame = np.full((scene.height, scene.width, 3), (70, 80, 90), dtype=np.uint8)
    frame[: int(scene.height * 0.28)] = (150, 160, 165)  # Wall.
    half_l, half_w = TABLE_L / 2.0, TABLE_W / 2.0
    corners = camera.project(
        np.array(
            [[-half_l, -half_w, 0], [half_l, -half_w, 0], [half_l, half_w, 0], [-half_l, half_w, 0]],
            dtype=np.float64,
        )
    )
    cv2.fillPoly(frame, [np.round(corners).astype(np.int32)], scene.table_bgr, cv2.LINE_AA)
    net = camera.project(
        np.array(
            [[0, -half_w, 0], [0, half_w, 0], [0, half_w, 0.1525], [0, -half_w, 0.1525]],
            dtype=np.float64,
        )
    )
    cv2.fillPoly(frame, [np.round(net).astype(np.int32)], (200, 200, 200), cv2.LINE_AA)
    return frame


def _draw_players(frame: np.ndarray, camera: Camera, time: float) -> None:
    for side in (-1, 1):
        sway = 0.25 * np.sin(2.4 * time + side)
        base = np.array([side * 2.1, sway, 0.0])
        top = camera.project(base + [0, 0, 1.7])[0]
        bottom = camera.project(base + [0, 0, -0.6])[0]
        thickness = int(0.42 * camera.pixels_per_meter(base))
        cv2.line(frame, tuple(np.round(top).astype(int)), tuple(np.round(bottom).astype(int)), (70, 50, 120), thickness)
        # An arm with a racket that swings, the classic false positive.
        swing = 0.5 * np.sin(5.0 * time + side)
        hand = base + np.array([-side * (0.45 + 0.1 * swing), swing, 1.0 + 0.3 * swing])
        shoulder = camera.project(base + [0, 0, 1.3])[0]
        hand_px = camera.project(hand)[0]
        ppm = camera.pixels_per_meter(hand)
        cv2.line(frame, tuple(np.round(shoulder).astype(int)), tuple(np.round(hand_px).astype(int)), (120, 150, 200), int(0.09 * ppm))
        cv2.circle(frame, tuple(np.round(hand_px).astype(int)), int(0.08 * ppm), (40, 40, 180), -1, cv2.LINE_AA)


def generate_video(path: str, scene: Scene, truth_path: Optional[str] = None) -> dict:
    """Render a rally to ``path`` and return its ground truth."""
    camera = Camera(scene)
    background = _draw_scene(scene, camera)
    sub_steps = scene.sub_steps
    frame_dt = 1.0 / scene.fps
    sub_dt = frame_dt * scene.exposure_fraction / sub_steps

    # Build the rally with a fine time step, for interpolating ball positions.
    fine_dt = 0.0005
    flights: List[np.ndarray] = []
    shots: List[dict] = []
    clock = 0.6
    direction = 1
    start_x, start_z = -(TABLE_L / 2.0 + 0.1), 0.25
    for shot_index, speed in enumerate(scene.speeds_mps):
        yaw = scene.yaw_degrees[shot_index % len(scene.yaw_degrees)]
        # Diagonal shots start off-centre, so that they cross the table's centre line.
        lateral = scene.lateral_offset + np.tan(np.radians(yaw)) * direction * start_x
        rows = simulate_shot(
            start_x, start_z, direction, speed, lateral, fine_dt, yaw, scene.magnus_accel
        )
        rows[:, 0] += clock
        flights.append(rows)
        shots.append(
            {
                "start_time_s": float(rows[0, 0]),
                "end_time_s": float(rows[-1, 0]),
                "peak_speed_mps": float(rows[:, 4].max()),
                "mean_speed_mps": float(rows[:, 4].mean()),
            }
        )
        clock = rows[-1, 0]
        direction = -direction
        # The next shot leaves from where this one was hit.
        start_x, start_z = rows[-1, 1], float(np.clip(rows[-1, 3], 0.1, 0.5))
        # A pause after every third shot simulates the start of a new rally.
        if (shot_index + 1) % 3 == 0:
            clock += 2.0
    duration = clock + scene.idle_seconds

    def ball_at(time: float) -> Optional[np.ndarray]:
        for rows in flights:
            if rows[0, 0] <= time <= rows[-1, 0]:
                index = int((time - rows[0, 0]) / fine_dt)
                return rows[min(index, len(rows) - 1), 1:4]
        return None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        path, fourcc, scene.container_fps or scene.fps, (scene.width, scene.height)
    )
    rng = np.random.default_rng(7)
    track = []
    for index in range(int(duration * scene.fps)):
        time = index * frame_dt
        scene_frame = background.copy()
        true_ball = ball_at(time + 0.5 * scene.exposure_fraction * frame_dt)
        if true_ball is not None:
            track.append([index, time, *camera.project(true_ball)[0].tolist(), *true_ball.tolist()])
        if scene.players:
            _draw_players(scene_frame, camera, time)
        accumulated = np.zeros_like(scene_frame, dtype=np.float32)
        for sub in range(sub_steps):
            # Each sub-frame accumulates the static scene with the ball on top.
            layer = scene_frame.copy()
            ball = ball_at(time + sub * sub_dt)
            if ball is not None:
                center = camera.project(ball)[0]
                radius = BALL_RADIUS * camera.pixels_per_meter(ball)
                # Draw with sub-pixel precision so that tiny balls move smoothly.
                shift = 4
                cv2.circle(
                    layer,
                    tuple(np.round(center * (1 << shift)).astype(int)),
                    int(round(radius * (1 << shift))),
                    scene.ball_bgr,
                    -1,
                    cv2.LINE_AA,
                    shift,
                )
            accumulated += layer
        frame = accumulated / sub_steps
        if scene.noise_sigma > 0:
            frame = frame + rng.normal(0, scene.noise_sigma, frame.shape)
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()

    half_l, half_w = TABLE_L / 2.0, TABLE_W / 2.0
    corners = camera.project(
        np.array(
            [[-half_l, -half_w, 0], [half_l, -half_w, 0], [half_l, half_w, 0], [-half_l, half_w, 0]],
            dtype=np.float64,
        )
    )
    truth = {
        "fps": scene.fps,
        "focal_px": camera.focal,
        "corners": corners.tolist(),
        "corners_arg": ";".join(f"{x:.1f},{y:.1f}" for x, y in corners),
        "shots": shots,
        "track": track,
    }
    if truth_path:
        with open(truth_path, "w") as handle:
            json.dump(truth, handle, indent=2)
    return truth
