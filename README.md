# ttspeed

Command-line tool that measures the ball speed of each shot in a table tennis training
video and generates a new video showing it.

## Setup

Requires Python 3.9 or newer. ffmpeg is installed with the dependencies.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
.venv/bin/python -m ttspeed training.mp4 -o output
```

The table is detected automatically. Always check the outline in
`output/training_calibration.jpg`: every speed depends on it. Corners given by hand are
usually more exact, and they gave a better fit than automatic detection on real footage.

### Output files

| File | Content |
| --- | --- |
| `training_speed.mp4` | The video with the ball trail and speed panel, keeping the audio |
| `training_shots.csv` | One row per shot: video time, peak, initial and mean speed, rally, bounces, confidence |
| `training_trajectory.csv` | One row per frame: ball position and instantaneous speed |
| `training_summary.json` | Shots, totals, calibration and warnings |
| `training_calibration.jpg` | The calibration drawn on the scene |

A shot runs from one racket contact to the next. A pause of over 1.5 s starts a new rally.

### Calibration

If the automatic outline is wrong, give the table corners by hand, which is also the most
reliable option. To read pixel coordinates, save a frame with a grid:

```bash
.venv/bin/python -m ttspeed training.mp4 --save-grid grid.jpg
```

Then pass the corners in perimeter order, with the first two on a long edge:

```bash
.venv/bin/python -m ttspeed training.mp4 --table-corners "224,494;1056,494;937,346;343,346"
```

| Option | Use |
| --- | --- |
| `--table-corners` | Most reliable. Corrects perspective. |
| (none) | Detects a blue or green table automatically, with the same perspective correction. |
| `--calib-line` with `--calib-length` | Two pixels a known distance apart, on the ball's flight plane (such as the ends of the table's centre line). The camera must be perpendicular to the table. |
| `--px-per-meter` | Fixed scale, with the same restriction. |
| Ball size | Automatic last resort when no table is found. Least reliable. |

### Other options

| Option | Use |
| --- | --- |
| `--slowmo 8` | **Required for slow-motion files** that play slower than real time, or speeds will be wrong. For example, 240 fps played at 30 fps is 8. |
| `--ball-color white\|orange` | Ball colour (default: `auto`) |
| `--max-speed`, `--min-speed` | Plausible speed range, in km/h (defaults: 140 and 10) |
| `--focal-px` | Camera focal length in pixels, if known |
| `--no-video` | Only write the data files |

## Recording for good results

- **Keep the camera still**, on a tripod. Camera motion breaks detection.
- **Film from the side of the table**, with the whole table visible. Oblique views work,
  but views from behind an end line trigger a warning and are unreliable.
- **Use the highest frame rate available**, 120 fps or more if possible.
- **Use bright light and a short exposure**, with a background that contrasts with the ball.

## How it works

1. **Calibration:** the table corners give the camera pose.
2. **Detection:** each frame is compared with a rolling median background, which keeps the
   ball's whole disc even when it barely moves in the image. Blobs are filtered by size
   and ranked by colour and size.
3. **Tracking:** blobs are linked into candidate trajectories. Checks on smoothness,
   direction and blob shape reject anything else, such as moving arms, and dynamic
   programming picks the best set.
4. **Shots:** a reversal along the table marks a racket contact. An abrupt change in the
   ball's vertical motion in the image marks a bounce, which stays within its shot.
5. **3D flight and speed:** each shot gets its own vertical flight plane. A bounce fixes a
   point of that plane, because the ball is known to be on the table. The plane chosen is
   the one where a physical flight (gravity, air drag and spin) best matches the pixels.
   Speed comes from that fit, using real timestamps, so variable-frame-rate video works.

Each shot has a `confidence` value. `high` means a bounce anchors the flight plane and the
fit matches the pixels. Shots marked `low` are shown as `(ESTIMATE)` in the video.

## Accuracy

### Real footage

One real recording has been tested: 2 minutes at 30 fps from an oblique ultra-wide camera,
without any independently measured speeds.

- **Correct detections:** all 16 reported shots were checked by eye and are real ball
  flights. Their speeds, from 14 to 23 km/h, agree with the rough time the ball takes to
  cross the table.
- **Missed shots:** at least 7 stretches of real ball flight are not reported. Fast shots,
  balls near the net and balls in front of a moving player are the most often lost.
- **Speed error on real footage is unknown.** Validating it needs a radar or a similar
  reference.

### Synthetic footage

Peak speed error against the known speed, on 6 shots from 18 to 72 km/h
(reproduce with `python tests/evaluate.py`):

| Scenario | Shots found | Mean error | Worst error |
| --- | --- | --- | --- |
| Oblique ultra-wide, 30 fps, straight shots | 6/6 | -3.0% | 6.4% |
| Oblique ultra-wide, 30 fps, diagonal shots | 6/6 | -3.6% | 10.5% |
| Oblique ultra-wide, 60 fps, diagonal shots | 6/6 | -3.5% | 11.8% |
| Oblique ultra-wide, 30 fps, diagonal shots with strong topspin | 6/6 | -19.3% | 63.9% |
| Side view, 30 fps | 5/6 | -8.7% | 14.0% |
| Side view, 60 fps | 6/6 | -2.4% | 8.4% |
| Side view, 120 fps | 6/6 | -1.6% | 7.1% |
| 240 fps slow motion | 6/6 | -0.1% | 5.8% |
| Oblique camera, 60 fps | 5/6 | -2.2% | 8.6% |
| High camera, 60 fps | 6/6 | -2.9% | 8.6% |
| Ball 40 cm off the centre line, 60 fps | 6/6 | -9.1% | 18.1% |
| Orange ball, 60 fps | 6/6 | -2.3% | 8.4% |
| Automatic table detection, 60 fps | 6/6 | -2.2% | 8.2% |
| Line calibration, 60 fps (one extra false shot) | 6/6 | -8.8% | 12.7% |
| Ball-size calibration, 60 fps (two extra false shots) | 6/6 | -11.3% | 15.0% |

The simulation uses the same physics as the flight model, so these results are a best case.

### Known limitations

- **Spin is only roughly modelled.** Strong topspin produced the largest errors, up to
  64%, and up to 35% among `high` confidence shots.
- **Peak speed reads low.** The true peak happens at contact, but the contact frame is
  discarded as unreliable and the ball has already slowed by the next one. Higher frame
  rates reduce this.
- **Shots without a visible bounce are less reliable.** They are always marked `low`, and
  are dropped unless they cross the net with at least 6 tracked points.
- **Perspective is only corrected with the table.** Line, fixed-scale and ball-size
  calibrations use neither the 3D flight model nor the table checks.
- **One ball at a time.** With several balls in flight, as in multiball drills, only the
  best trajectory at each moment is kept.

## Tests

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests
```
