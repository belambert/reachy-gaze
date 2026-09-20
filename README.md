---
title: Tracker
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

# Tracker

Reachy Mini follows a named object with its head. Pick a class in the control
panel — `person`, `cat`, `dog`, or any of the other 80 COCO classes — and the
head tracks it across the room, with the body turning to extend the reach.

Detection runs **off-board**, on a machine with a GPU, because the robot's Pi is
already busy with motor control. The robot sends downscaled JPEGs over the
network and gets boxes back.

## How it works

Two loops at different rates, which is what keeps the motion smooth despite a
detector that only answers a handful of times per second:

| Loop          | Rate   | Job                                                       |
| ------------- | ------ | --------------------------------------------------------- |
| Vision thread | ~12 Hz | Detect, pick the target, convert its pixel to a head pose |
| Control loop  | 50 Hz  | Slew the head toward that pose                            |

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
    uv run tracker-server --host 0.0.0.0

It serves on port 8100 and picks up CUDA, MPS, or CPU automatically. The default
model, `yolo11x.pt`, is the most accurate of the family and downloads 109 MB on
first run; pass `--model yolo11s.pt` if the machine is modest. See
[Choosing a model](#choosing-a-model).

Note the machine's LAN address — the robot needs to reach it.

### 2. Start the app, on the robot

Install it as a Reachy Mini app, then open the control panel at
<http://localhost:8042>. Set **Detector** to `http://<your-laptop-ip>:8100`,
choose a class, and tick **Tracking enabled**.

The panel shows whether the detector is reachable, the measured detection rate,
and where in frame the tracker currently believes the target is.

## Choosing a model

All five YOLO11 sizes are COCO-80 and drop in via `--model`; each downloads on
first use. Latency below is **measured** on an Apple M4 Pro (14 core, MPS),
decoding a JPEG payload at quality 75 and timing decode plus inference together,
median of 30 runs after 8 warmups. The robot sends 640 px, so that is the column
that applies; 512 px is kept to show what shrinking the payload would buy. The
mAP column is Ultralytics' published COCO figure, not measured here.

| Model        | Params | Weights | mAP50-95 | 640 px           | 512 px           |
| ------------ | ------ | ------- | -------- | ---------------- | ---------------- |
| `yolo11n.pt` | 2.6 M  | 5.4 MB  | 39.5     | 7.1 ms (140 fps) | 6.6 ms (152 fps) |
| `yolo11s.pt` | 9.4 M  | 18 MB   | 47.0     | 9.8 ms (102 fps) | 8.2 ms (122 fps) |
| `yolo11m.pt` | 20.1 M | 39 MB   | 51.5     | 18.7 ms (53 fps) | 14.3 ms (70 fps) |
| `yolo11l.pt` | 25.3 M | 49 MB   | 53.4     | 22.5 ms (45 fps) | 17.1 ms (59 fps) |
| `yolo11x.pt` | 56.9 M | 109 MB  | 54.7     | 41.3 ms (24 fps) | 29.0 ms (34 fps) |

The app asks for at most `DETECT_HZ` detections per second — an 83 ms budget at
the default of 12 Hz. On this class of hardware **every size fits**, including
`yolo11x` with room to spare. The model is not the bottleneck; the request cap
and the network round trip are.

So pick on accuracy, not speed, which is why `yolo11x.pt` is the default. It
costs +15 mAP over `yolo11n` for 34 ms a frame you were going to spend waiting
anyway, and that accuracy buys range — the whole reason detection is off-board.
Drop down only if the machine running the server is weaker than this one, or is
busy with something else.

These numbers are one machine and one 5-object test image; a slower laptop
reorders the table. Re-run the benchmark before trusting them elsewhere.

## Tuning

Constants live at the top of `tracker/main.py`:

| Constant     | Default | Effect                                               |
| ------------ | ------- | ---------------------------------------------------- |
| `DETECT_HZ`  | 12      | Ceiling on detection requests                        |
| `SLEW_TAU`   | 0.15    | Larger is smoother and laggier; smaller is snappier  |
| `LOST_AFTER` | 1.5     | Seconds without a detection before the head gives up |

Selection gates — minimum box area, max frame-to-frame jump, misses tolerated —
are constructor arguments on `TargetSelector` in `tracker/tracking.py`.

If the head settles slightly off-center and stops, that's `CenterFilter`'s dead
zone doing its job: it trades a standing offset of up to 0.02 in normalized
frame coordinates for a head that doesn't dither on detector noise.

## Swapping the detector

`tracker/detector.py` defines a `Detector` protocol — `classes()` and
`detect(frame, labels, conf)`. `RemoteDetector` is the HTTP implementation. An
on-device backend, or an open-vocabulary model like YOLOE that takes free-text
prompts instead of a fixed 80 classes, only has to satisfy that protocol.

## Development

    uv sync --extra dev
    uv run pytest
    uv run black tracker tests && uv run isort tracker tests

The tests cover target selection, smoothing, the slew math, and the detector
wire protocol against a stub server. None of them need a robot.
