---
title: Reachy Gaze
emoji: 👁️
colorFrom: red
colorTo: blue
sdk: static
pinned: false
license: mit
short_description: Make Reachy Mini look at people and pets
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Reachy Gaze

A Reachy Mini app that makes the robot look at people and pets. It scans the
room, follows whatever it finds, and once its target has sat still long enough
to get boring, ignores it for a while and scans for something else.

Detection runs **off-board** on a machine with a GPU, because the robot's Pi is
busy with motor control. The robot streams downscaled JPEGs to a detector and
gets boxes back.

## Running It

### 1. Start a Detector

Run one of these somewhere the robot can reach:

| Backend              | Address            | Notes                                         |
| -------------------- | ------------------ | --------------------------------------------- |
| Built-in (`builtin`) | `http://host:8100` | Bundled FastAPI + YOLO11; CPU, CUDA or MPS    |
| Triton (`triton`)    | `host:8101` (gRPC) | Fastest; needs a private Triton server        |
| VLM (`vlm`)          | `http://host:8000` | A vLLM vision model; slow, any labels         |

The app defaults to the built-in server. The Triton backend is faster but needs
a private server; pick it in the control panel or set
`REACHY_GAZE_BACKEND=triton`.

To run the built-in server:

    uv sync --extra server
    uv run reachy-gaze-server                 # serves on 0.0.0.0:8100
    uv run reachy-gaze-server --model yolo11s.pt

### 2. Start the App

Install it on the robot as a Reachy Mini app, then open the control panel at
<http://localhost:8042> to pick the backend and detector address. The defaults
come from `REACHY_GAZE_BACKEND` and `REACHY_GAZE_SERVER_URL` if set.

The panel's view draws the target's bounding box filled with the frame it was
captured from (served at `/crop`), with the class icon beside it.

Tuning constants, including which classes it looks for, live at the top of
`reachy_gaze/main.py`. New detector backends go in the `BACKENDS` table in
`reachy_gaze/detector.py`.

## How It Behaves

The head runs a cycle of phases, published as `phase` in the panel's `/state`
endpoint (<http://localhost:8042/state>):

| Phase      | What the head does                                                    | Ends when                                                     |
| ---------- | --------------------------------------------------------------------- | ------------------------------------------------------------- |
| `idle`     | Rests at neutral; tracking is switched off                            | Tracking is switched on                                       |
| `scanning` | Sweeps side to side from where it is                                  | It locks onto a target                                        |
| `tracking` | Follows the target, moving or not, holding its aim through short gaps | The target is still for the boredom time, or lost for 10 s    |

Both ways out of `tracking` lead back to `scanning`. A target counts as still
while it stays within a few degrees of where it settled, so a moving target is
followed indefinitely. The boredom time is the panel's Boredom slider (20 s by
default). Once bored, the head ignores anything within 15° of where that
target sat for a minute, so the scan finds something else rather than the
same target again.

The Travel slider limits how far the head turns left and right of straight
ahead (90° by default, up to 160°). It bounds the scan and tracking alike.

Alongside `phase`, `/state` reports `phase_for` (seconds in the current
phase), and while tracking, `still_for` and `bored_in` (seconds).

## Development

See [arch.md](arch.md) for how the code is put together: threads, the
look-around state machine, detector backends and the control panel.

    uv sync --extra dev
    uv run pytest
    node tests/test_panel.mjs
    uv run black reachy_gaze tests && uv run isort reachy_gaze tests

## Deployment

The robot installs the app from a Hugging Face Space. The Deploy workflow
mirrors `main` to the Space each time the tests pass on a push to `main`, and
can also be run by hand from the Actions tab. It needs two settings on the
GitHub repo (Settings → Secrets and variables → Actions):

| Name       | Kind     | Value                                                      |
| ---------- | -------- | ---------------------------------------------------------- |
| `HF_TOKEN` | Secret   | A Hugging Face access token with write access to the Space |
| `HF_SPACE` | Variable | The Space as `owner/name`                                  |

The workflow pushes without force, so if someone pushes to the Space
directly, its history diverges and the deploy fails until the two are
reconciled.

## License

MIT; see [LICENSE](LICENSE).
