"""Object detection backends: the robot sends a frame, something else finds the boxes."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import requests
from PIL import Image

# The 80 COCO classes, bundled so the control panel can populate its picker
# before the detection server has ever been reached.
# fmt: off
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]
# fmt: on


@dataclass(frozen=True)
class Detection:
    """One detected object, boxed in full-resolution frame pixels."""

    label: str
    conf: float
    box: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        """Box center as (u, v) pixels."""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class Detector(Protocol):
    """Anything that can name its vocabulary and find boxes in a frame."""

    def classes(self) -> list[str]:
        """Labels this detector can recognize."""
        ...

    def detect(
        self, frame: npt.NDArray[np.uint8], labels: list[str], conf: float
    ) -> list[Detection]:
        """Find `labels` in a BGR frame, keeping detections above `conf`."""
        ...


class DetectorUnavailable(RuntimeError):
    """The detection service could not be reached or returned garbage."""


class RemoteDetector:
    """Detector backed by an HTTP service, typically a laptop with a GPU.

    Frames are downscaled and JPEG-encoded before going over the network; boxes
    come back in downscaled coordinates and are rescaled to the source frame so
    callers can use them against the camera intrinsics.

    The default width matches YOLO's native 640 px, so the model isn't handed an
    upscaled image; at JPEG quality 75 a frame is only tens of kilobytes.
    """

    def __init__(
        self,
        url: str,
        width: int = 640,
        quality: int = 75,
        timeout: float = 2.0,
    ) -> None:
        """Point the detector at a service base URL, e.g. http://192.168.1.20:8100."""
        self.url = url.rstrip("/")
        self.width = width
        self.quality = quality
        self.timeout = timeout
        self._session = requests.Session()

    def classes(self) -> list[str]:
        """Ask the service for its vocabulary.

        Raises:
            DetectorUnavailable: If the service cannot be reached or answers
                with something other than a class list.

        """
        try:
            resp = self._session.get(f"{self.url}/classes", timeout=self.timeout)
            resp.raise_for_status()
            return list(resp.json()["classes"])
        except requests.RequestException as e:
            raise DetectorUnavailable(f"{self.url}: {e}") from e
        except (KeyError, ValueError) as e:
            raise DetectorUnavailable(f"{self.url}: bad class list") from e

    def detect(
        self, frame: npt.NDArray[np.uint8], labels: list[str], conf: float
    ) -> list[Detection]:
        """POST one frame and return the boxes, in source-frame pixels."""
        jpeg, scale = self._encode(frame)
        try:
            resp = self._session.post(
                f"{self.url}/detect",
                params={"labels": ",".join(labels), "conf": conf},
                data=jpeg,
                headers={"Content-Type": "image/jpeg"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as e:
            raise DetectorUnavailable(f"{self.url}: {e}") from e
        except ValueError as e:
            raise DetectorUnavailable(f"{self.url}: bad JSON response") from e

        return [
            Detection(
                label=d["label"],
                conf=float(d["conf"]),
                box=tuple(float(c) * scale for c in d["box"]),  # type: ignore[arg-type]
            )
            for d in payload.get("detections", [])
        ]

    def _encode(self, frame: npt.NDArray[np.uint8]) -> tuple[bytes, float]:
        """Return (JPEG bytes, factor mapping encoded pixels back to source)."""
        height, width = frame.shape[:2]
        rgb = Image.fromarray(frame[..., ::-1])  # camera gives BGR
        if width > self.width:
            target = (self.width, max(1, round(height * self.width / width)))
            rgb = rgb.resize(target, Image.BILINEAR)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=self.quality)
        return buf.getvalue(), width / rgb.width
