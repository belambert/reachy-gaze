"""COCO detection service: run this on the machine with the GPU, not on the robot.

uv run --extra server tracker-server --host 0.0.0.0
"""

from __future__ import annotations

import io
import logging

import numpy as np
import torch
import typer
import uvicorn
from fastapi import FastAPI, Query, Request
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

app = FastAPI(title="tracker detector")
cli = typer.Typer(add_completion=False)

_model: YOLO | None = None
_device = "cpu"


def pick_device() -> str:
    """Best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@app.get("/classes")
def classes() -> dict:
    """The model's full vocabulary."""
    assert _model is not None
    return {"classes": list(_model.names.values())}


@app.get("/health")
def health() -> dict:
    """Liveness plus what the model is running on."""
    return {"ok": _model is not None, "device": _device}


@app.post("/detect")
async def detect(
    request: Request,
    labels: str = Query("", description="Comma-separated class names to keep"),
    conf: float = Query(0.4, ge=0.0, le=1.0),
) -> dict:
    """Detect objects in a posted JPEG, returning boxes in that JPEG's pixels."""
    assert _model is not None
    body = await request.body()
    image = Image.open(io.BytesIO(body)).convert("RGB")

    wanted = [name.strip() for name in labels.split(",") if name.strip()]
    names = {name: idx for idx, name in _model.names.items()}
    # Filtering in the model is far cheaper than filtering the results.
    keep = [names[name] for name in wanted if name in names] or None

    result = _model.predict(
        np.asarray(image), conf=conf, classes=keep, device=_device, verbose=False
    )[0]

    detections = [
        {
            "label": _model.names[int(cls)],
            "conf": float(score),
            "box": [float(v) for v in box],
        }
        for box, score, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        )
    ]
    return {"detections": detections, "width": image.width, "height": image.height}


@cli.command()
def main(
    host: str = "0.0.0.0",
    port: int = 8100,
    model: str = "yolo11x.pt",
    device: str = "",
) -> None:
    """Serve COCO detection over HTTP."""
    global _model, _device
    logging.basicConfig(level=logging.INFO)

    _device = device or pick_device()
    _model = YOLO(model)
    _model.to(_device)
    logger.info("Serving %s on %s at http://%s:%d", model, _device, host, port)

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    cli()
