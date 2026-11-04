"""Write analysis results to CSV and JSON."""

from __future__ import annotations

import csv
import json
from typing import Sequence

from .shots import MPS_TO_KMH, Shot


def _timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:06.3f}"


def shot_row(shot: Shot) -> dict:
    return {
        "shot_id": shot.shot_id,
        "rally_id": shot.rally_id,
        "start_time_s": round(shot.start_time_s, 3),
        "start_timestamp": _timestamp(shot.start_time_s),
        "end_time_s": round(shot.end_time_s, 3),
        "peak_time_s": round(shot.peak_time_s, 3),
        "peak_speed_kmh": round(shot.peak_speed_mps * MPS_TO_KMH, 1),
        "initial_speed_kmh": round(shot.initial_speed_mps * MPS_TO_KMH, 1),
        "mean_speed_kmh": round(shot.mean_speed_mps * MPS_TO_KMH, 1),
        "direction": shot.direction,
        "bounces": shot.bounces,
        "confidence": shot.confidence,
        "tracked_points": len(shot.points),
    }


def write_shots_csv(path: str, shots: Sequence[Shot]) -> None:
    rows = [shot_row(shot) for shot in shots]
    fields = [
        "shot_id", "rally_id", "start_time_s", "start_timestamp", "end_time_s",
        "peak_time_s", "peak_speed_kmh", "initial_speed_kmh", "mean_speed_kmh",
        "direction", "bounces", "confidence", "tracked_points",
    ]  # fmt: skip
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_trajectory_csv(path: str, shots: Sequence[Shot]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["shot_id", "frame", "time_s", "x_px", "y_px", "x_m", "y_m", "z_m", "speed_kmh"]
        )
        for shot in shots:
            for point in shot.points:
                writer.writerow(
                    [
                        shot.shot_id,
                        point.frame_index,
                        round(point.time_s, 4),
                        round(point.x_px, 1),
                        round(point.y_px, 1),
                        round(point.x_m, 3),
                        round(point.y_m, 3),
                        round(point.z_m, 3),
                        round(point.speed_mps * MPS_TO_KMH, 1),
                    ]
                )


def write_summary_json(
    path: str,
    video: dict,
    calibration: dict,
    warnings: Sequence[str],
    shots: Sequence[Shot],
) -> None:
    rows = [shot_row(shot) for shot in shots]
    peaks = [row["peak_speed_kmh"] for row in rows]
    summary = {
        "video": video,
        "calibration": calibration,
        "warnings": list(warnings),
        "totals": {
            "shots": len(rows),
            "rallies": max((row["rally_id"] for row in rows), default=0),
            "max_peak_speed_kmh": max(peaks, default=None),
            "mean_peak_speed_kmh": round(sum(peaks) / len(peaks), 1) if peaks else None,
        },
        "shots": rows,
    }
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2)
