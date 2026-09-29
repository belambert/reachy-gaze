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

A Reachy Mini app that makes the robot look at people and pets. It moves its
head to a random pose, scans the room from there, follows whatever it finds,
and starts over once its target has sat still long enough to get boring.

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

Tuning constants, including which classes it looks for, live at the top of
`reachy_gaze/main.py`. New detector backends go in the `BACKENDS` table in
`reachy_gaze/detector.py`.

## How It Behaves

The head runs a cycle of phases, published as `phase` in the panel's `/state`
endpoint (<http://localhost:8042/state>):

| Phase      | What the head does                                                    | Ends when                                                                                 |
| ---------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `idle`     | Rests at neutral; tracking is switched off                            | Tracking is switched on                                                                   |
| `moving`   | Goes to a random orientation and position                             | It arrives (or 5 s pass)                                                                  |
| `holding`  | Holds still at that pose, ignoring what it sees                       | 1.5 s pass                                                                                |
| `scanning` | Sweeps side to side from where it is                                  | It locks onto a target, or 30 s pass (new cycle)                                          |
| `tracking` | Follows the target, moving or not, holding its aim through short gaps | The target is still for the boredom time (new cycle), or lost for 10 s (back to scanning) |

A target counts as still while it stays within a few degrees of where it
settled, so a moving target is followed indefinitely. The boredom time is the
panel's Boredom slider (20 s by default).

Alongside `phase`, `/state` reports `cycle` (a count that goes up with each
random pose), `phase_for` (seconds in the current phase), and while tracking,
`still_for` and `bored_in` (seconds).

## Development

    uv sync --extra dev
    uv run pytest
    node tests/test_panel.mjs
    uv run black reachy_gaze tests && uv run isort reachy_gaze tests

There are two remotes: `origin` is the GitHub repo, which runs CI, and `space`
is the Hugging Face Space the robot installs from. A change only reaches the
robot once it's pushed to `space`.

    git push origin main && git push space main

## License

MIT; see [LICENSE](LICENSE).
