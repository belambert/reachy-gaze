---
title: Gaze Tracker
emoji: 👁️
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Point Reachy Mini's head at a person, a cat, or a dog
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Gaze Tracker

Reachy Mini follows a named object with its head. Pick a class in the control
panel — `person`, `cat`, `dog`, or any of the other 80 COCO classes — and the
head tracks it across the room, with the body turning to extend the reach.

Detection runs **off-board**, on a machine with a GPU, because the robot's Pi is
already busy with motor control. The robot sends downscaled JPEGs over the
network and gets boxes back.

## How it works

Two loops at different rates, which is what keeps the motion smooth despite a
detector that only answers a handful of times per second:

| Loop           | Rate    | Job                                                      |
| -------------- | ------- | -------------------------------------------------------- |
| Vision thread  | ~12 Hz  | Detect, pick the target, convert its pixel to a head pose |
| Control loop   | 50 Hz   | Slew the head toward that pose                            |

The vision thread turns each detection into an **absolute** head pose, using the
head pose recorded when the frame was captured. That means a detection arriving
80 ms late still points where the target actually was, rather than compounding
into overshoot.

Between detections the control loop eases toward the last goal with a time
constant of 150 ms. Target selection reuses the approach in the SDK's own
`reachy_mini.vision.face_tracking`: acquire the largest box, then follow the
nearest one frame to frame, with a max-jump gate so the head doesn't snap
between two cats and a miss counter so it lets go once the real one leaves.

## Running it

### 1. Start the detector, on your laptop

    uv sync --extra server
    uv run gaze-tracker-server --host 0.0.0.0

It serves on port 8100 and picks up CUDA, MPS, or CPU automatically. The default
model, `yolo11n.pt`, downloads on first run; pass `--model yolo11s.pt` for better
range at some cost in speed.

Note the machine's LAN address — the robot needs to reach it.

### 2. Start the app, on the robot

Install it as a Reachy Mini app, then open the control panel at
<http://localhost:8042>. Set **Detector** to `http://<your-laptop-ip>:8100`,
choose a class, and tick **Tracking enabled**.

The panel shows whether the detector is reachable, the measured detection rate,
and where in frame the tracker currently believes the target is.

## Tuning

Constants live at the top of `gaze_tracker/main.py`:

| Constant     | Default | Effect                                                     |
| ------------ | ------- | ---------------------------------------------------------- |
| `DETECT_HZ`  | 12      | Ceiling on detection requests                               |
| `SLEW_TAU`   | 0.15    | Larger is smoother and laggier; smaller is snappier         |
| `LOST_AFTER` | 1.5     | Seconds without a detection before the head gives up        |

Selection gates — minimum box area, max frame-to-frame jump, misses tolerated —
are constructor arguments on `TargetSelector` in `gaze_tracker/tracking.py`.

If the head settles slightly off-center and stops, that's `CenterFilter`'s dead
zone doing its job: it trades a standing offset of up to 0.02 in normalized
frame coordinates for a head that doesn't dither on detector noise.

## Swapping the detector

`gaze_tracker/detector.py` defines a `Detector` protocol — `classes()` and
`detect(frame, labels, conf)`. `RemoteDetector` is the HTTP implementation. An
on-device backend, or an open-vocabulary model like YOLOE that takes free-text
prompts instead of a fixed 80 classes, only has to satisfy that protocol.

## Development

    uv sync --extra dev
    uv run pytest
    uv run black gaze_tracker tests && uv run isort gaze_tracker tests

The tests cover target selection, smoothing, the slew math, and the detector
wire protocol against a stub server. None of them need a robot.
