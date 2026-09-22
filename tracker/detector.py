"""Object detection backends: the robot sends a frame, something else finds the boxes."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import numpy.typing as npt
import requests
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


# --- Detectors ---------------------------------------------------------------
#
# One class per detection service, each satisfying the Detector protocol and
# sharing frame encoding through _JpegDetector. The BACKENDS registry at the
# bottom is what the app selects between.

TRACKER_MODEL = "tracker"  # the ensemble name the Triton vision-server exposes


class _JpegDetector:
    """Frame encoding shared by detectors that send a JPEG over the network.

    Frames are downscaled to `width` and JPEG-encoded; the returned factor maps
    the encoded pixels back to the source frame, so boxes an off-board detector
    returns line up with the camera intrinsics. The default width matches YOLO's
    native 640 px, so the model isn't handed an upscale; at quality 75 a frame
    is only tens of kilobytes.
    """

    def __init__(self, width: int = 640, quality: int = 75, timeout: float = 2.0):
        self.width = width
        self.quality = quality
        self.timeout = timeout

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


class TritonDetector(_JpegDetector):
    """Detector backed by the Triton vision-server, over gRPC.

    The server runs pre/post-processing and answers with boxes in the pixels of
    the JPEG it was sent. gRPC rather than HTTP JSON on purpose: JSON spends a
    decimal number per JPEG byte and inflates each frame ~4.6x, enough to blow
    the robot's wifi budget at the detection rate.
    """

    def __init__(
        self,
        url: str,
        width: int = 640,
        quality: int = 75,
        timeout: float = 2.0,
    ) -> None:
        """Point the detector at a Triton gRPC endpoint, e.g. spark-10cf:8101."""
        super().__init__(width, quality, timeout)
        # tritonclient wants a bare host:port; tolerate a pasted scheme anyway
        self.url = url.split("://", 1)[-1].rstrip("/")
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


class BuiltinDetector(_JpegDetector):
    """Detector backed by the built-in FastAPI server (see tracker.server).

    A frame is POSTed as JPEG and boxes come back in downscaled coordinates,
    which are rescaled to the source frame. Simpler than Triton and happy on
    CPU or MPS, but JSON over HTTP, so meant for a laptop on the same LAN.
    """

    def __init__(
        self,
        url: str,
        width: int = 640,
        quality: int = 75,
        timeout: float = 2.0,
    ) -> None:
        """Point the detector at a server base URL, e.g. http://192.168.1.20:8100."""
        super().__init__(width, quality, timeout)
        self.url = url.rstrip("/")
        self._session = requests.Session()

    def classes(self) -> list[str]:
        """Ask the server for its vocabulary.

        Raises:
            DetectorUnavailable: If the server cannot be reached or answers with
                something other than a class list.

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


# --- Backends ----------------------------------------------------------------
#
# The registry the app selects between. Add a backend by writing its Detector
# above and adding one line here; the control panel picks it up from snapshot().


@dataclass(frozen=True)
class Backend:
    """A selectable detection backend: how to label it and how to build it."""

    key: str
    label: str
    default_url: str
    factory: Callable[[str], Detector]


BACKENDS: dict[str, Backend] = {
    "triton": Backend(
        "triton", "Triton (vision-server)", "spark-10cf:8101", TritonDetector
    ),
    "builtin": Backend(
        "builtin", "Built-in server", "http://10.0.0.206:8100", BuiltinDetector
    ),
}


def make_detector(backend: str, url: str) -> Detector:
    """Build the client for a named backend, pointed at `url`."""
    try:
        spec = BACKENDS[backend]
    except KeyError:
        raise ValueError(f"unknown backend {backend!r}") from None
    return spec.factory(url)
