---
title: Tracker
emoji: 👁️
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Point Reachy Mini's head at people, cats, dogs and birds
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Tracker

Reachy Mini follows people, cats, dogs and birds with its head, tracking them
across the room with the body turning to extend the reach. It looks for all
four at once and locks onto whichever makes the better target.

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

Between detections the control loop eases toward the last goal with a critically
damped second-order follower. That matters more than it sounds: the goal steps
with every detection, and a first-order lag reaches a stepped setpoint with a
velocity discontinuity each time, which is exactly what reads as jerky. Carrying
angular velocity as state keeps velocity continuous, and capping the follower's
pull bounds how much it can change per tick.

Target selection reuses the approach in the SDK's own
`reachy_mini.vision.face_tracking`: acquire the largest box, then follow the
nearest one frame to frame, with a max-jump gate so the head doesn't snap
between two cats and a miss counter so it lets go once the real one leaves.
Selection is class-blind: the four labels only decide what gets detected, and
from there the head follows a box, whatever it is labelled. The panel reports
which class the current target came back as.

When detections stop, the head keeps its aim on the last known position for
`LOST_AFTER` seconds before it starts scanning, so a subject that steps behind
something is still being watched when it reappears. The panel distinguishes the
two: **locked** while sightings are arriving, **holding** with the age of the
last one while the head waits it out.

## Running it

### 1. Start the detector, on your laptop

    uv sync --extra server
    uv run tracker-server --host 0.0.0.0

It serves on port 8100 and picks up CUDA, MPS, or CPU automatically. The default
model, `yolo11x.pt`, is the most accurate of the family and downloads 109 MB on
first run; pass `--model yolo11s.pt` if the machine is modest. See
[Choosing a model](#choosing-a-model).

Note the machine's LAN address — the robot needs to reach it.

The server logs first contact, a summary every 10 s, and when a client goes
quiet — enough to tell "the robot isn't reaching me" from "it is, and the
detections are empty" without a line per frame:

    Serving yolo11x.pt on mps at http://0.0.0.0:8100
    first contact from 10.0.0.42
    served class list (80 classes) to 10.0.0.42
    10.0.0.42: 118 req in 10s (11.8/s), 41 ms avg, 1.2 det/req
    10.0.0.42 went quiet after 118 requests

Pass `--verbose` to add uvicorn's per-request access log when debugging.

### 2. Start the app, on the robot

Install it as a Reachy Mini app, then open the control panel at
<http://localhost:8042>. Check that **Detector** points at the machine running
the server.

Tracking is **on from the moment the app starts** — untick **Tracking enabled**
to stop it. That setting is not persisted, so a restart begins tracking again.

The field is prefilled from `DEFAULT_SERVER_URL` in `tracker/main.py`. Set
`TRACKER_SERVER_URL` to change it without editing code — worth doing if the
server's address comes from DHCP and moves.

The panel shows whether the detector is reachable, the measured detection rate,
and where in frame the tracker currently believes the target is.

What it hunts for is `TRACK_LABELS` in `tracker/main.py` — `person`, `cat`,
`dog`, `bird`. Any COCO class works there. Whenever the Detector address
changes the app checks the labels against the server's vocabulary and logs a
warning for any it doesn't know, because the server quietly detects
*everything* when it recognises none of them.

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

| Constant         | Default | Effect                                                    |
| ---------------- | ------- | --------------------------------------------------------- |
| `DETECT_HZ`      | 12      | Ceiling on detection requests                             |
| `SMOOTH_TAU`     | 0.09    | Follower time constant; larger is smoother and laggier    |
| `MAX_HEAD_PULL`  | 10.0    | rad/s² cap on the follower's pull; **lower is gentler**   |
| `MAX_HEAD_SPEED` | 3.5     | rad/s hard ceiling on commanded rotation                  |
| `BLEND_TAU`      | 0.4     | Seconds to ease between searching and locked-on posture   |
| `LOST_AFTER`     | 10.0    | Seconds holding the last aim point before giving up       |
| `STALE_AFTER`    | 1.0     | Seconds before the panel calls a lock held rather than live |
| `SCAN_DEGREES`   | 60.0    | Half-width of the scan                                    |
| `SCAN_HZ`        | 0.08    | Scan rate                                                 |

Each scan is phase-aligned to the head's current yaw, so it picks up from
wherever the head was holding instead of returning to centre first. If the
scan still looks unsteady on hardware, the commanded path is not the cause —
simulated, it peaks at 0.26 rad/s³ of jerk against roughly 10 while tracking.
Look instead at automatic body yaw, which a ±60° scan leans on heavily, or at
the servos, which judder at the very low speeds around each turnaround.

Both constants scale the motion together: peak scan speed is `2π · SCAN_HZ ·
SCAN_DEGREES`, 30 °/s as set. Raising either keeps the servos moving faster and
out of their judder range, at the cost of a brisker scan.

`MAX_HEAD_PULL` is only the starting value — the control panel's
**Responsiveness** slider changes it live, so there is no need to edit code and
reinstall to find a setting you like. It bounds how much commanded velocity can
change in a single tick, which is the dial that trades smoothness against
chasing power:

Measured against a 40° step, with the time taken to settle within a degree of
it:

| Responsiveness | Peak jerk | Peak speed | Settles in |
| -------------- | --------- | ---------- | ---------- |
| 4              | 2.2       | 10 °/s     | 3.80 s     |
| 10 (default)   | 5.6       | 26 °/s     | 1.60 s     |
| 20             | 11.1      | 52 °/s     | 0.92 s     |
| 60             | 33.3      | 150 °/s    | 0.56 s     |

Every setting still converges without overshoot; lower simply takes longer.
Simulated against a brisk subject — 40° of yaw at 0.15 Hz, detected at 12 Hz —
the follower cuts peak jerk about fourfold versus the plain first-order lag it
replaced, 6.5 against 26.2. That costs tracking lag, and this is where the
setting is felt: a mean of 8.2° behind the subject at 10, against 4.2° at 20.
Raise it if the head visibly trails things you care about.

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
wire protocol against a stub server. None of them need a robot. The panel's
state sync has its own suite, which needs Node but no dependencies:

    node tests/test_panel.mjs

Two remotes, and they are not interchangeable: `origin` is the private GitHub
repo, which runs CI, and `space` is the Hugging Face Space the robot installs
from. A change is only on the robot once it has gone to `space`.

    git push origin main && git push space main
