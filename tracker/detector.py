"""Object detection backends: the robot sends a frame, something else finds the boxes."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import tritonclient.grpc as grpcclient
from PIL import Image
from tritonclient.utils import InferenceServerException

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


# The ensemble the robot talks to; see the vision-server repo. Detectors swap
# behind this name, so the client never learns which one is loaded.
TRACKER_MODEL = "tracker"


class RemoteDetector:
    """Detector backed by the Triton vision-server, over gRPC.

    Frames are downscaled and JPEG-encoded before going over the network; the
    server runs pre/post-processing and answers with boxes in the pixels of the
    JPEG it was sent, which are rescaled to the source frame so callers can use
    them against the camera intrinsics.

    gRPC rather than HTTP JSON on purpose: JSON spends a decimal number per JPEG
    byte and inflates each frame ~4.6x, enough to blow the robot's wifi budget
    at the detection rate. The default width matches YOLO's native 640 px, so
    the model isn't handed an upscaled image; at JPEG quality 75 a frame is only
    tens of kilobytes.
    """

    def __init__(
        self,
        url: str,
        width: int = 640,
        quality: int = 75,
        timeout: float = 2.0,
    ) -> None:
        """Point the detector at a Triton gRPC endpoint, e.g. spark-10cf:8101."""
        # tritonclient wants a bare host:port; tolerate a pasted scheme anyway
        self.url = url.split("://", 1)[-1].rstrip("/")
        self.width = width
        self.quality = quality
        self.timeout = timeout
        self._client = grpcclient.InferenceServerClient(url=self.url)

    def classes(self) -> list[str]:
        """Confirm the ensemble is loaded and report its vocabulary.

        The contract exposes no class-list endpoint, so this checks the model is
        ready — which doubles as the startup reachability probe — and returns
        the COCO labels the ensemble is built against.

        Raises:
            DetectorUnavailable: If the server cannot be reached or the model
                is not loaded.

        """
        try:
            if not self._client.is_model_ready(TRACKER_MODEL):
                raise DetectorUnavailable(f"{self.url}: {TRACKER_MODEL} not ready")
        except InferenceServerException as e:
            raise DetectorUnavailable(f"{self.url}: {e}") from e
        return list(COCO_CLASSES)

    def detect(
        self, frame: npt.NDArray[np.uint8], labels: list[str], conf: float
    ) -> list[Detection]:
        """Infer one frame and return the boxes, in source-frame pixels."""
        jpeg, scale = self._encode(frame)
        inputs = [
            self._input("JPEG", np.frombuffer(jpeg, dtype=np.uint8), "UINT8"),
            self._input("KEEP", np.array(labels, dtype=object), "BYTES"),
            self._input("CONF", np.array([conf], dtype=np.float32), "FP32"),
        ]
        outputs = [
            grpcclient.InferRequestedOutput(n) for n in ("BOXES", "SCORES", "LABELS")
        ]
        try:
            result = self._client.infer(
                TRACKER_MODEL,
                inputs=inputs,
                outputs=outputs,
                client_timeout=self.timeout,
            )
        except InferenceServerException as e:
            raise DetectorUnavailable(f"{self.url}: {e}") from e

        boxes = result.as_numpy("BOXES")
        if boxes is None or boxes.size == 0:
            return []
        scores, out_labels = result.as_numpy("SCORES"), result.as_numpy("LABELS")
        return [
            Detection(
                label=lbl.decode() if isinstance(lbl, bytes) else str(lbl),
                conf=float(score),
                box=tuple(float(c) * scale for c in box),  # type: ignore[arg-type]
            )
            for box, score, lbl in zip(boxes.reshape(-1, 4), scores, out_labels)
        ]

    @staticmethod
    def _input(name: str, data: np.ndarray, dtype: str) -> grpcclient.InferInput:
        """A Triton input tensor carrying `data`."""
        inp = grpcclient.InferInput(name, list(data.shape), dtype)
        inp.set_data_from_numpy(data)
        return inp

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
