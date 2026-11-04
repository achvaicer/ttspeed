"""Command-line interface."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .calibration import TABLE_LENGTH_M, CalibrationError
from .pipeline import Options, run, save_coordinate_grid
from .shots import MPS_TO_KMH
from .video import VideoError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ttspeed",
        description=(
            "Measure the ball speed of each shot in a table tennis training video and "
            "generate a new video that shows it. The camera must not move."
        ),
    )
    parser.add_argument("video", help="input video file")
    parser.add_argument("-o", "--output-dir", default="output", help="output directory (default: output)")

    calibration = parser.add_argument_group(
        "calibration", "Without any of these, the table is detected automatically."
    )
    calibration.add_argument(
        "--table-corners",
        metavar="X,Y;X,Y;X,Y;X,Y",
        help="table corners in pixels, in perimeter order, with the first two on a long edge (recommended)",
    )
    calibration.add_argument(
        "--calib-line",
        metavar="X,Y;X,Y",
        help="two pixels spaced a known distance apart, on the plane the ball flies in",
    )
    calibration.add_argument(
        "--calib-length",
        type=float,
        default=TABLE_LENGTH_M,
        metavar="METERS",
        help=f"real length of --calib-line (default: {TABLE_LENGTH_M}, the table's length)",
    )
    calibration.add_argument("--px-per-meter", type=float, help="fixed image scale")
    calibration.add_argument("--focal-px", type=float, help="camera focal length in pixels, if known")
    calibration.add_argument(
        "--save-grid",
        metavar="IMAGE",
        help="save a frame with a pixel grid for reading coordinates, then exit",
    )

    parser.add_argument(
        "--ball-color", choices=["auto", "white", "orange"], default="auto", help="ball colour (default: auto)"
    )
    parser.add_argument(
        "--slowmo",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="how many times slower than real time the video plays, e.g. 8 for 240 fps played at 30 fps",
    )
    parser.add_argument("--max-speed", type=float, default=140.0, metavar="KMH", help="fastest plausible speed (default: 140)")
    parser.add_argument("--min-speed", type=float, default=10.0, metavar="KMH", help="slowest speed counted as a shot (default: 10)")
    parser.add_argument("--no-video", action="store_true", help="only write the data files")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.save_grid:
            path = save_coordinate_grid(args.video, args.save_grid)
            print(f"Saved coordinate grid: {path}")
            return 0
        result = run(
            Options(
                input_path=args.video,
                output_dir=args.output_dir,
                table_corners=args.table_corners,
                calib_line=args.calib_line,
                calib_length_m=args.calib_length,
                px_per_meter=args.px_per_meter,
                focal_px=args.focal_px,
                ball_color=args.ball_color,
                slowmo=args.slowmo,
                max_speed_kmh=args.max_speed,
                min_speed_kmh=args.min_speed,
                render_video=not args.no_video,
            )
        )
    except (VideoError, CalibrationError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print()
    print(f"{'Shot':>4}  {'Rally':>5}  {'Time':>9}  {'Peak km/h':>9}  {'Mean km/h':>9}  {'Bounces':>7}  {'Confidence':>10}")
    for shot in result.shots:
        print(
            f"{shot.shot_id:>4}  {shot.rally_id:>5}  {shot.start_time_s:>8.2f}s  "
            f"{shot.peak_speed_mps * MPS_TO_KMH:>9.1f}  "
            f"{shot.mean_speed_mps * MPS_TO_KMH:>9.1f}  {shot.bounces:>7}  {shot.confidence:>10}"
        )
    print(f"\nShots detected: {len(result.shots)}")
    for message in result.warnings:
        print(f"Warning: {message}")
    print("\nFiles:")
    for path in result.output_paths.values():
        print(f"  {os.path.relpath(path)}")
    return 0
