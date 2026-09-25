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
`TRACK_LABELS` is also a **preference order**: a cat in view outranks a person
in view, and the head will leave the one for the other. A preferred class has
to be seen for a few frames running before it takes the lock, so a detection
flickering at the confidence threshold cannot bounce the head between two
subjects, and the minimum-area gate still applies — a stray speck of cat will
not pull the head off a person standing right there. The preference does not
run backwards: once on the cat, a person cannot take it back.

It also does not fixate forever. After `LOCK_TIMEOUT` seconds on one target the
lock is dropped and the head starts scanning again to see what else is around.
Until it locks onto something new, it passes over **everything it has dwelt on
in the last `DWELL_MEMORY` seconds**, not just the target it tired of, so it
doesn't bounce straight back to the cat it watched a minute ago. Which objects
those are comes from the [world model](#the-world-model), which tracks them by
world direction rather than position in frame, so the scanning head cannot
slide the block off them. When nothing else turns up, each old target becomes
fair game again once `DWELL_MEMORY` seconds have passed since it was last
watched — so with a single subject in the room, the head scans for that long
before coming back to it.

When detections stop, the head keeps its aim on the last known position for
`LOST_AFTER` seconds before it starts scanning the room — 60° either side of
centre, with the head held level — so a subject that steps behind something is
still being watched when it reappears. The panel distinguishes the two:
**locked** while sightings are arriving, **holding** with the age of the last
one while the head waits it out.

### The world model

Alongside the one target it aims at, the robot keeps a short-term memory of
everything it has seen lately (`tracker/world.py`). Each entry holds the
object's type, its location, when it was last seen, when the head last dwelt on
it (was locked onto it), and a small picture of it from the last time the head
was locked onto it. It keeps every box of every tracked class, not
only the target.

The location is a **unit direction** in the world frame (+X forward, +Y left, +Z
up), not a point in space: a single camera gives a bearing but no range. It is
absolute, so an object stays put in memory while the head turns away from it. A
full direction, not just a yaw, so two subjects on the same bearing at different
heights stay distinct.

Each frame, a sighting within 15° of a remembered object with the same label
counts as that object. The object moves to the new direction and its age
resets. Anything else becomes a new entry. Closest pairs are matched first, so
two cats side by side keep their own entries. An object not seen for
`FORGET_AFTER` seconds is dropped. Nothing removes an object early when the
head looks where it was and finds nothing, so something that has moved on
lingers until it ages out.

The panel shows the world as a table under the aim readout, newest first: each
object's picture and type, its bearing as arrows (e.g. "←30° ↓5°" for 30° left and 5° down),
and how long ago it was seen and watched. The row the head is locked onto is highlighted. `/state`
returns it as `world`, with each object's `id`, `label`, `direction`,
`yaw`/`pitch` in degrees, `age` and `dwelt_ago` in seconds (`dwelt_ago` is null
if it has never been the target), `target`, and `thumb`.

The picture is taken only while the head is locked onto the object, since it is
centred and steady then rather than a blurred box at the edge of a scan. Each
frame's target box is cropped and scaled to fill 48×36 px, then shown at half
that so it sits on the text's line without making the row any taller. It is
sent as a JPEG data URI of 1–2 KB, and takes about 0.1 ms to make on an M4
laptop; the robot's Pi will be slower, but not measured. An object never locked
onto has no picture, and shows its emoji instead.

A badge alongside the lock counts down to boredom ("bored in 7s") while the head
holds a target, then reads "bored: avoiding recent targets" until something new
takes the lock. `/state` carries these as `bored_in` and `bored`.

## Running it

### 1. Start a detector

There are three detection backends, and you need one of them running somewhere
the robot can reach. The control panel picks between them.

| Backend                     | Address            | Speed               | Labels          |
| --------------------------- | ------------------ | ------------------- | --------------- |
| **Triton** (`triton`)       | `host:8101` (gRPC) | Full `DETECT_HZ`    | COCO-80         |
| **Built-in** (`builtin`)    | `http://host:8100` | Full, on a LAN      | COCO-80         |
| **VLM** (`vlm`)             | `http://host:8000` | A frame every 1–2 s | Any (prompted)  |

**Triton** is the default and the fastest. The **vision-server** is a Triton
Inference Server that takes a JPEG and returns boxes. Set it up on the Spark
from its own repo; the short version is:

    scripts/build_engine.sh    # one-off, builds the TensorRT plan
    scripts/serve.sh           # serves gRPC on 8101 (and HTTP 8100, metrics 8102)

Check it is up:

    curl -sf <spark>:8100/v2/health/ready && echo READY || echo "NOT READY"

The client uses **gRPC on 8101**, not HTTP JSON on 8100: JSON spends a decimal
number per JPEG byte and inflates each frame ~4.6x, enough to blow the wifi
budget at 12 Hz. HTTP 8100 is for probing by hand.

**Built-in** is the bundled FastAPI + Ultralytics server — no Triton or TensorRT
needed, and happy on CPU, CUDA or Apple MPS, so it suits a laptop on the same
LAN:

    uv sync --extra server
    uv run tracker-server              # serves on 0.0.0.0:8100
    uv run tracker-server --model yolo11s.pt --port 8100

See [Choosing a model](#choosing-a-model) for which model to pass.

**VLM** prompts a vision-language model served by vLLM's OpenAI-compatible API
for boxes. It is open-vocabulary, so `TRACK_LABELS` isn't limited to COCO, but
it is far slower than a detector network. Serve a grounding-capable model (the
Qwen-VL family's `[0, 1000]` box convention is what the client expects), e.g.:

    vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000

The address is just the server; the model id is discovered from `/v1/models`.
See [Swapping the detector](#swapping-the-detector) for how each backend works.

### 2. Start the app, on the robot

Install it as a Reachy Mini app, then open the control panel at
<http://localhost:8042>. Pick the **Backend** — `Triton (vision-server)`,
`Built-in server` or `VLM (vLLM)` — and check its **Detector** address, in the
form the table above gives. Switching backend fills in that backend's default
address, which you can then edit.

Tracking is **on from the moment the app starts** — untick **Tracking enabled**
to stop it. That setting is not persisted, so a restart begins tracking again.

The backend and address are prefilled from `DEFAULT_BACKEND` and
`DEFAULT_SERVER_URL` in `tracker/main.py` (Triton on `spark-10cf:8101`). Set
`TRACKER_BACKEND` (`triton`, `builtin` or `vlm`) and `TRACKER_SERVER_URL` to
change them without editing code; with only the backend set, the address
defaults to that backend's.

The panel shows whether the detector is reachable, the measured detection rate,
and a view of what's in frame: the tracked target as a green disc and every
other box alongside it in a second colour, each labelled with its type. Dashed
rings mark every 10° off the camera's axis, so a position in frame reads as an
angle. They come from the camera's focal length (radius f·tan θ) and ignore lens
distortion, so they are approximate toward the edges. Below
that it reads out where the head is aimed — e.g. "Aimed ←20° ↑5°" for 20° left
and 5° up — with ←/→ the yaw and ↑/↓ the pitch. This is the absolute
aim in the world: the head pose comes back from forward kinematics over all the
joints, so the body's turntable yaw is already folded in.

What it hunts for is `TRACK_LABELS` in `tracker/main.py` — `cat`, `dog`,
`bird`, `person`, most preferred first. Any COCO class works there, and
reordering the list is how you change what it would rather watch. Whenever the Detector address
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
| `MAX_HEAD_PULL`  | 5.0     | rad/s² cap on the follower's pull; **lower is gentler**   |
| `MAX_HEAD_SPEED` | 3.5     | rad/s hard ceiling on commanded rotation                  |
| `BLEND_TAU`      | 0.4     | Seconds to ease between searching and locked-on posture   |
| `LOST_AFTER`     | 10.0    | Seconds holding the last aim point before giving up       |
| `LOCK_TIMEOUT`   | 30.0    | Seconds on one target before breaking off to scan for others |
| `STALE_AFTER`    | 5.0     | Seconds before the panel calls a lock held rather than live |
| `FORGET_AFTER`   | 120.0   | Seconds before an unseen object leaves the world model    |
| `DWELL_MEMORY`   | 60.0    | Seconds a watched object stays shunned once bored         |
| `SCAN_DEGREES`   | 90.0    | Half-width of the scan                                    |
| `SCAN_HZ`        | 0.04    | Scan rate                                                 |

### The scan

The scan is a slow side-to-side sweep in yaw only; the head is held level, with
no up-and-down motion. One cycle takes 25 s and covers **180° of yaw**.

That reaches 90° to either side. The body's `yaw_body` joint would allow ±160°,
so there is still room to widen `SCAN_DEGREES`, but as it stands a subject that
leaves further round or behind has to come back into view on its own.

Each scan is phase-aligned to the head's current yaw, so it picks up from
wherever the head was holding instead of snapping to centre first. Any pitch
left over from tracking is eased back to level once by the follower.

### Why it never whips round

A target can be acquired well off to one side of where the head is pointing —
at the far end of a scan, say — and the head must not lunge at it. It can't, and the reason is `MAX_HEAD_PULL`
rather than anything in the scan: once the spring term saturates, the follower
settles at the speed where the capped pull balances damping,

    terminal speed = MAX_HEAD_PULL * SMOOTH_TAU / 2

which is **0.225 rad/s, about 13°/s** at the defaults. Simulated, the peak speed
closing a 20°, 60°, 120° or 179° gap is the same 13°/s every time — distance
changes how long it takes, never how fast it gets there. `MAX_HEAD_SPEED` is a
backstop that never binds at these settings.

The catch is that this ceiling is the **Responsiveness** slider's, not a fixed
one. At 60 the terminal speed is 155°/s, and the robot will whip. If you raise
it for snappier tracking, that is what you are trading away.

For comparison, over a full scan cycle starting from rest:

| Motion                          | Peak speed | Peak acceleration |
| ------------------------------- | ---------- | ----------------- |
| Scan, 90° yaw                   | 23 °/s     | 97 °/s²           |
| Tracking a 40° step (pull 5)    | 13 °/s     | 159 °/s²          |

The scan moves faster than tracking does at the default, but it accelerates
more gently, so if it looks unsteady on hardware the commanded path is not the
cause. Look instead at automatic body yaw, if the
scan is widened far enough to lean on it, or at the servos, which judder at the
very low speeds around each turnaround.

`MAX_HEAD_PULL` is only the starting value — the control panel's
**Responsiveness** slider changes it live, so there is no need to edit code and
reinstall to find a setting you like. It caps the follower's spring term, which is
the dial that trades smoothness against chasing power — and, as above, it sets
the speed ceiling for any large movement.

Measured against a 40° step, with the time taken to settle within a degree of
it, and the mean lag behind a brisk subject — swinging ±40° of yaw at 0.15 Hz,
detected at 12 Hz, averaged after the first 10 s:

| Responsiveness | Peak speed | Peak acceleration | Settles in | Mean lag |
| -------------- | ---------- | ----------------- | ---------- | -------- |
| 4              | 10 °/s     | 127 °/s²          | 3.80 s     | 23.5°    |
| 5 (default)    | 13 °/s     | 159 °/s²          | 3.06 s     | 22.2°    |
| 10             | 26 °/s     | 318 °/s²          | 1.60 s     | 10.1°    |
| 20             | 52 °/s     | 637 °/s²          | 0.92 s     | 4.2°     |
| 60             | 150 °/s    | 1910 °/s²         | 0.56 s     | 4.2°     |

Every setting still converges without overshoot; lower simply takes longer.
Against that same subject at 10, the follower cut peak acceleration about
fourfold versus the plain first-order lag it replaced, and lower settings cut
it further. That costs tracking lag, and this is where the setting is felt:
at 5 the head's top speed of 13°/s is below the subject's peak of 38°/s, so it
trails by 22° on average, against 10° at 10 and 4° at 20. Past 20 the lag is set
by the detection rate, not the follower. Raise it if the head visibly trails
things you care about.

Selection gates — minimum box area, max frame-to-frame jump, misses tolerated —
are constructor arguments on `TargetSelector` in `tracker/tracking.py`.

If the head settles slightly off-center and stops, that's `CenterFilter`'s dead
zone doing its job: it trades a standing offset of up to 0.02 in normalized
frame coordinates for a head that doesn't dither on detector noise.

## Swapping the detector

`tracker/detector.py` defines a `Detector` protocol — `classes()` and
`detect(frame, labels, conf)` — and the backends that implement it:

- **`TritonDetector`** — the vision-server's `tracker` ensemble over gRPC.
- **`BuiltinDetector`** — the built-in FastAPI server over HTTP.
- **`VlmDetector`** — a vLLM server running a vision-language model, prompted for
  boxes. Open-vocabulary: the labels go into the prompt, and boxes come back as
  `[0, 1000]`-normalised coordinates (the Qwen grounding convention). Decoding is
  constrained to a JSON schema (`response_format`), so the reply is schema-valid
  JSON with labels from the requested set — no reasoning prose to parse around,
  and faster than letting a thinking model narrate first. Still far slower than a
  detector network: a frame every second or few, not a stream.

Backends are registered in the `BACKENDS` table at the bottom of that file, and
the control panel selects between them — `snapshot()` reports the list. **To add
one:** write a class satisfying the protocol and add a line to `BACKENDS` giving
its key, label, default address, and constructor.

## Development

    uv sync --extra dev
    uv run pytest
    uv run black tracker tests && uv run isort tracker tests

The tests cover target selection, the world model, smoothing, the slew math,
and the detector wire protocol against a stub server. None of them need a robot. The panel's
state sync has its own suite, which needs Node but no dependencies:

    node tests/test_panel.mjs

Two remotes, and they are not interchangeable: `origin` is the private GitHub
repo, which runs CI, and `space` is the Hugging Face Space the robot installs
from. A change is only on the robot once it has gone to `space`.

    git push origin main && git push space main
