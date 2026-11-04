"""Tests for speed estimation and the end-to-end pipeline."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthetic import Scene, generate_video  # noqa: E402
from ttspeed.calibration import (  # noqa: E402
    CalibrationError,
    TableCalibration,
    parse_points,
)
from ttspeed.cli import main  # noqa: E402
from ttspeed.pipeline import Options, run  # noqa: E402
from ttspeed.shots import ShotConfig, TrackPoint, estimate_speeds, segment_shots  # noqa: E402
from ttspeed.video import find_ffmpeg  # noqa: E402


def silent(_message: str) -> None:
    pass


@pytest.fixture(scope="module")
def rally(tmp_path_factory):
    work = tmp_path_factory.mktemp("rally")
    video = str(work / "rally.mp4")
    truth = generate_video(video, Scene(width=960, height=540, sub_steps=4))
    return video, truth


def test_speed_estimate_on_constant_motion():
    times = np.arange(10) / 60.0
    positions = np.column_stack([10.0 * times, 2.0 * times])
    speeds = estimate_speeds(times, positions, half_window=1)
    assert speeds == pytest.approx(np.full(10, np.hypot(10.0, 2.0)), rel=1e-6)


def test_speed_estimate_handles_irregular_timestamps():
    times = np.array([0.0, 0.016, 0.034, 0.05, 0.067, 0.083])
    positions = np.column_stack([8.0 * times, np.zeros(6)])
    assert estimate_speeds(times, positions, half_window=1) == pytest.approx(np.full(6, 8.0))


def _line(start_frame, count, start_x, velocity, fps=60.0):
    return [
        TrackPoint(
            frame_index=start_frame + i,
            time_s=(start_frame + i) / fps,
            x_px=0.0,
            y_px=0.0,
            x_m=start_x + velocity * i / fps,
            z_m=0.2,
        )
        for i in range(count)
    ]


def test_reversal_starts_a_new_shot():
    first = _line(0, 12, -1.0, 10.0)
    second = _line(12, 12, first[-1].x_m, -15.0)
    shots = segment_shots(first + second, ShotConfig(fps=60.0))
    assert len(shots) == 2
    assert shots[0].direction == "+x" and shots[1].direction == "-x"
    assert shots[0].peak_speed_mps == pytest.approx(10.0, rel=0.02)
    assert shots[1].peak_speed_mps == pytest.approx(15.0, rel=0.02)


def test_long_pause_starts_a_new_rally():
    first = _line(0, 12, -1.0, 10.0)
    second = _line(300, 12, 1.0, -10.0)
    shots = segment_shots(first + second, ShotConfig(fps=60.0))
    assert [shot.rally_id for shot in shots] == [1, 2]


def test_table_calibration_recovers_known_points():
    from synthetic import Camera, TABLE_L, TABLE_W

    scene = Scene()
    camera = Camera(scene)
    half_l, half_w = TABLE_L / 2, TABLE_W / 2
    corners = camera.project(
        np.array([[-half_l, -half_w, 0], [half_l, -half_w, 0], [half_l, half_w, 0], [-half_l, half_w, 0]])
    )
    calibration = TableCalibration(corners, (scene.width, scene.height), focal_px=camera.focal)
    truth = np.array([[-1.0, 0.0, 0.3], [0.5, 0.0, 0.1], [1.2, 0.0, 0.5]])
    world = calibration.to_world(camera.project(truth))
    assert np.abs(np.abs(world[:, 0]) - np.abs(truth[:, 0])).max() < 0.01
    assert np.abs(world[:, 1] - truth[:, 2]).max() < 0.01


def test_parse_points_rejects_bad_input():
    with pytest.raises(CalibrationError):
        parse_points("1,2;3,4", 4)
    with pytest.raises(CalibrationError):
        parse_points("a,b;c,d", 2)


def test_finds_every_shot_with_accurate_speed(rally, tmp_path):
    video, truth = rally
    result = run(
        Options(video, str(tmp_path), table_corners=truth["corners_arg"], render_video=False),
        log=silent,
    )
    assert len(result.shots) == len(truth["shots"])
    assert [shot.rally_id for shot in result.shots] == [1, 1, 1, 2, 2, 2]
    for shot, expected in zip(result.shots, truth["shots"]):
        assert shot.peak_speed_mps == pytest.approx(expected["peak_speed_mps"], rel=0.15)
        assert shot.start_time_s == pytest.approx(expected["start_time_s"], abs=0.1)


def test_output_files_and_video(rally, tmp_path):
    video, truth = rally
    code = main([video, "-o", str(tmp_path), "--table-corners", truth["corners_arg"]])
    assert code == 0

    with open(tmp_path / "rally_shots.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(truth["shots"])
    assert float(rows[0]["peak_speed_kmh"]) > 0

    with open(tmp_path / "rally_summary.json") as handle:
        summary = json.load(handle)
    assert summary["totals"]["shots"] == len(rows)

    source = cv2.VideoCapture(video)
    output = cv2.VideoCapture(str(tmp_path / "rally_speed.mp4"))
    assert output.isOpened()
    assert output.get(cv2.CAP_PROP_FRAME_COUNT) == source.get(cv2.CAP_PROP_FRAME_COUNT)
    assert output.get(cv2.CAP_PROP_FPS) == pytest.approx(source.get(cv2.CAP_PROP_FPS), rel=0.01)
    source.release()
    output.release()


def test_players_without_a_ball_produce_no_shots(tmp_path):
    video = str(tmp_path / "empty.mp4")
    truth = generate_video(
        video, Scene(width=960, height=540, speeds_mps=(), sub_steps=1, idle_seconds=8.0)
    )
    assert truth["shots"] == []
    result = run(
        Options(video, str(tmp_path), table_corners=truth["corners_arg"], render_video=False),
        log=silent,
    )
    assert result.shots == []


def test_audio_is_preserved(rally, tmp_path):
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        pytest.skip("ffmpeg is not available")
    video, truth = rally
    with_audio = str(tmp_path / "audio.mp4")
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", video, "-f", "lavfi", "-i",
         "sine=frequency=440:duration=2", "-c:v", "copy", "-c:a", "aac", "-shortest", with_audio],
        check=True,
    )  # fmt: skip
    run(Options(with_audio, str(tmp_path), table_corners=truth["corners_arg"]), log=silent)
    probe = subprocess.run(
        [ffmpeg, "-i", str(tmp_path / "audio_speed.mp4")], capture_output=True, text=True
    )
    assert "Audio: aac" in probe.stderr
    assert "Video: h264" in probe.stderr


def test_cli_reports_missing_file(tmp_path, capsys):
    assert main([str(tmp_path / "missing.mp4")]) == 1
    assert "not found" in capsys.readouterr().err


def test_grid_is_saved(rally, tmp_path):
    video, _ = rally
    path = tmp_path / "grid.jpg"
    assert main([video, "--save-grid", str(path)]) == 0
    assert path.exists()
