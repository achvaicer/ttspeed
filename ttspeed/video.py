"""Video input helpers and output muxing."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


class VideoError(RuntimeError):
    """Raised when a video cannot be read or written."""


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height


def open_video(path: str) -> Tuple[cv2.VideoCapture, VideoInfo]:
    if not os.path.isfile(path):
        raise VideoError(f"Video file not found: {path}")
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise VideoError(f"Could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or not np.isfinite(fps):
        capture.release()
        raise VideoError(f"Video reports an invalid frame rate ({fps}): {path}")
    info = VideoInfo(
        path=path,
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(fps),
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    return capture, info


def iter_frames(capture: cv2.VideoCapture, fps: float) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Yield ``(index, time_s, frame)`` using container timestamps when they are sane.

    Phones often record at a variable frame rate, so ``index / fps`` would drift.
    """
    index = 0
    previous_time = -1.0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        time_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        # Some backends return 0 or repeated values: fall back to a nominal time.
        if not np.isfinite(time_s) or time_s <= previous_time:
            time_s = index / fps if previous_time < 0 else previous_time + 1.0 / fps
        yield index, time_s, frame
        previous_time = time_s
        index += 1


def find_ffmpeg() -> Optional[str]:
    """Locate ffmpeg on PATH, or the binary bundled with imageio-ffmpeg."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


class FrameWriter:
    """Write BGR frames to H.264 through ffmpeg, keeping the source audio.

    Falls back to OpenCV's MPEG-4 encoder, without audio, when ffmpeg is missing.
    """

    def __init__(
        self,
        output_path: str,
        info: VideoInfo,
        fps: float,
        ffmpeg: Optional[str],
        crf: int = 20,
    ) -> None:
        self.output_path = output_path
        self.process: Optional[subprocess.Popen] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.warnings: List[str] = []
        self._log_path = output_path + ".ffmpeg.log"
        self._log = None
        # H.264 with yuv420p needs even dimensions.
        self.size = (info.width - info.width % 2, info.height - info.height % 2)
        if ffmpeg:
            self._log = open(self._log_path, "wb")
            command = [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.size[0]}x{self.size[1]}", "-r", f"{fps:.6f}", "-i", "pipe:0",
                "-i", info.path,
                "-map", "0:v:0", "-map", "1:a?",
                "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
                output_path,
            ]  # fmt: skip
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=self._log)
        else:
            self.warnings.append("ffmpeg not found: output uses MPEG-4 and has no audio.")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(output_path, fourcc, fps, self.size)
            if not self.writer.isOpened():
                raise VideoError(f"Could not open output video: {output_path}")

    def write(self, frame: np.ndarray) -> None:
        frame = frame[: self.size[1], : self.size[0]]
        if self.process is not None:
            try:
                self.process.stdin.write(frame.tobytes())
            except BrokenPipeError as error:
                raise VideoError(self._failure_message()) from error
        else:
            self.writer.write(frame)

    def close(self) -> None:
        if self.process is not None:
            self.process.stdin.close()
            code = self.process.wait()
            self._log.close()
            if code != 0:
                raise VideoError(self._failure_message())
            os.remove(self._log_path)
        elif self.writer is not None:
            self.writer.release()

    def _failure_message(self) -> str:
        if self._log is not None and not self._log.closed:
            self._log.close()
        try:
            with open(self._log_path, "r", errors="replace") as handle:
                details = handle.read().strip()
        except OSError:
            details = ""
        return f"ffmpeg failed while writing {self.output_path}: {details}"
