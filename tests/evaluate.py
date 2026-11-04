"""Measure accuracy on synthetic videos with a known ground truth.

Usage: python tests/evaluate.py [work_dir] [name_filter ...]
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic import Scene, generate_video  # noqa: E402
from ttspeed.pipeline import Options, run  # noqa: E402


@dataclass
class Case:
    name: str
    scene: Scene
    calibration: str = "corners"  # "corners", "auto", "line" or "ball".
    options: Dict = field(default_factory=dict)


# Camera recreated from a real recording: ultra-wide, oblique and close to the table.
OBLIQUE_WIDE = dict(
    width=1920,
    height=1080,
    camera_position=(-2.13, -1.44, 0.85),
    look_at=(0.1, 0.1, 0.0),
    focal_factor=0.47,
    speeds_mps=(5.0, 7.0, 10.0, 6.0, 13.0, 8.0),
    players=False,  # The default players would stand right in front of this camera.
)

CASES: List[Case] = [
    Case("wide-30fps-straight", Scene(fps=30.0, sub_steps=4, **OBLIQUE_WIDE)),
    Case("wide-30fps-diagonal", Scene(fps=30.0, sub_steps=4, yaw_degrees=(14.0, -10.0, 18.0), **OBLIQUE_WIDE)),
    Case("wide-30fps-topspin", Scene(fps=30.0, sub_steps=4, yaw_degrees=(14.0, -10.0, 18.0), magnus_accel=4.0, **OBLIQUE_WIDE)),
    Case("wide-60fps-diagonal", Scene(fps=60.0, sub_steps=4, yaw_degrees=(14.0, -10.0, 18.0), **OBLIQUE_WIDE)),
    Case("side-60fps", Scene()),
    Case("side-30fps", Scene(fps=30.0, exposure_fraction=0.5)),
    Case("side-120fps", Scene(fps=120.0, sub_steps=4)),
    Case("slowmo-240fps-as-30", Scene(fps=240.0, container_fps=30.0, sub_steps=3), options={"slowmo": 8.0}),
    Case("oblique-60fps", Scene(camera_position=(-2.2, -3.4, 1.8))),
    Case("high-camera-60fps", Scene(camera_position=(0.0, -3.0, 2.8))),
    Case("lateral-40cm", Scene(lateral_offset=0.4)),
    Case("orange-ball", Scene(ball_bgr=(30, 120, 250)), options={"ball_color": "orange"}),
    Case("green-table-auto", Scene(table_bgr=(70, 120, 30)), calibration="auto"),
    Case("blue-table-auto", Scene(), calibration="auto"),
    Case("line-calibration", Scene(), calibration="line"),
    Case("ball-size-only", Scene(table_bgr=(70, 80, 90)), calibration="ball"),
]


def evaluate(case_and_dir):
    case, work_dir = case_and_dir
    video = os.path.join(work_dir, f"{case.name}.mp4")
    truth = generate_video(video, case.scene)
    options = Options(video, os.path.join(work_dir, case.name), render_video=False, **case.options)
    if case.calibration == "corners":
        options.table_corners = truth["corners_arg"]
    elif case.calibration == "line":
        # The table's centre line lies on the ball's flight plane.
        c = truth["corners"]
        options.calib_line = (
            f"{(c[0][0] + c[3][0]) / 2},{(c[0][1] + c[3][1]) / 2};"
            f"{(c[1][0] + c[2][0]) / 2},{(c[1][1] + c[2][1]) / 2}"
        )
    result = run(options, log=lambda message: None)

    # Slow motion stretches video time, and ground-truth times are real times.
    time_scale = (case.scene.fps / case.scene.container_fps) if case.scene.container_fps else 1.0
    errors = []
    high_errors = []
    matched = set()
    for expected in truth["shots"]:
        start, end = expected["start_time_s"] * time_scale, expected["end_time_s"] * time_scale
        best, best_overlap = None, 0.0
        for shot in result.shots:
            overlap = min(end, shot.end_time_s) - max(start, shot.start_time_s)
            if overlap > best_overlap:
                best, best_overlap = shot, overlap
        if best is not None and best_overlap > 0.4 * (end - start):
            matched.add(best.shot_id)
            error = (best.peak_speed_mps - expected["peak_speed_mps"]) / expected["peak_speed_mps"]
            errors.append(error)
            if best.confidence == "high":
                high_errors.append(error)
    return {
        "name": case.name,
        "expected": len(truth["shots"]),
        "found": len(result.shots),
        "matched": len(matched),
        "errors": errors,
        "high_errors": high_errors,
        "method": result.calibration.describe()["method"],
    }


def main() -> int:
    work_dir = sys.argv[1] if len(sys.argv) > 1 else "evaluation"
    os.makedirs(work_dir, exist_ok=True)
    names = sys.argv[2:]
    cases = [case for case in CASES if not names or any(n in case.name for n in names)]
    with multiprocessing.Pool(4) as pool:
        reports = pool.map(evaluate, [(case, work_dir) for case in cases])
    print(f"{'case':<22}{'calibration':<15}{'found':>6}{'matched':>9}{'mean err':>10}{'max |err|':>11}{'high: n':>9}{'max |err|':>11}")
    for report in reports:
        errors = report["errors"]
        mean = sum(errors) / len(errors) if errors else float("nan")
        worst = max((abs(e) for e in errors), default=float("nan"))
        print(
            f"{report['name']:<22}{report['method']:<15}"
            f"{report['found']:>3}/{report['expected']:<2}{report['matched']:>6}/{report['expected']:<2}"
            f"{mean * 100:>9.1f}%{worst * 100:>10.1f}%{len(report['high_errors']):>9}{max((abs(e) for e in report['high_errors']), default=float('nan')) * 100:>10.1f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
