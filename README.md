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

A Reachy Mini app that makes the robot look at people and pets. It follows
whatever it finds with its head, keeps a short memory of what it has seen, and
looks around the room when there's nothing to watch or it gets bored.

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
