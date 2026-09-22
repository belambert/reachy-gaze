import io

import numpy as np
import pytest
from PIL import Image
from tritonclient.utils import InferenceServerException

import tracker.detector as detector
from tracker.detector import COCO_CLASSES, DetectorUnavailable, RemoteDetector

# What the fake server sends back, per output tensor; the request it received
# lands in `seen` so tests can assert on what actually went over the wire.
reply: dict = {}
seen: dict = {}


class FakeInput:
    """Stands in for grpcclient.InferInput, recording the array it is given."""

    def __init__(self, name, shape, datatype):
        self.name, self.shape, self.datatype = name, shape, datatype

    def set_data_from_numpy(self, data):
        self.data = data


class FakeResult:
    def as_numpy(self, name):
        return reply.get(name)


class FakeClient:
    def __init__(self, url, **kwargs):
        seen["url"] = url

    def is_model_ready(self, model):
        if reply.get("_down"):
            raise InferenceServerException("connection refused")
        return not reply.get("_not_ready")

    def infer(self, model, inputs, outputs, client_timeout=None):
        if reply.get("_down"):
            raise InferenceServerException("connection refused")
        seen["model"] = model
        seen["timeout"] = client_timeout
        seen["inputs"] = {inp.name: inp for inp in inputs}
        seen["outputs"] = [o for o in outputs]
        return FakeResult()


@pytest.fixture(autouse=True)
def fake_triton(monkeypatch):
    reply.clear()
    seen.clear()
    monkeypatch.setattr(detector.grpcclient, "InferInput", FakeInput)
    monkeypatch.setattr(detector.grpcclient, "InferRequestedOutput", lambda name: name)
    monkeypatch.setattr(detector.grpcclient, "InferenceServerClient", FakeClient)


@pytest.fixture
def frame():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def sent_image():
    """Decode the JPEG the detector put on the wire."""
    return Image.open(io.BytesIO(seen["inputs"]["JPEG"].data.tobytes()))


def test_boxes_are_rescaled_to_source_frame(frame):
    # Server answers in 512-wide space; 640/512 = 1.25.
    reply["BOXES"] = np.array([[80, 40, 160, 120]], dtype=np.float32)
    reply["SCORES"] = np.array([0.8], dtype=np.float32)
    reply["LABELS"] = np.array([b"cat"], dtype=object)

    det = RemoteDetector("spark:8101", width=512).detect(frame, ["cat"], 0.4)[0]
    assert det.box == pytest.approx((100, 50, 200, 150))
    assert det.center == pytest.approx((150, 100))
    assert det.label == "cat"


def test_small_frames_are_not_upscaled():
    reply["BOXES"] = np.array([[10, 10, 20, 20]], dtype=np.float32)
    reply["SCORES"] = np.array([0.8], dtype=np.float32)
    reply["LABELS"] = np.array([b"cat"], dtype=object)

    small = np.zeros((120, 160, 3), dtype=np.uint8)
    det = RemoteDetector("spark:8101", width=512).detect(small, ["cat"], 0.4)[0]
    assert sent_image().width == 160, "must not waste bytes upscaling"
    assert det.box == pytest.approx((10, 10, 20, 20))


def test_request_carries_labels_and_confidence(frame):
    RemoteDetector("spark:8101", width=512).detect(frame, ["cat", "dog"], 0.65)

    keep = [
        s.decode() if isinstance(s, bytes) else s for s in seen["inputs"]["KEEP"].data
    ]
    assert keep == ["cat", "dog"]
    assert seen["inputs"]["CONF"].data == pytest.approx([0.65])
    assert seen["model"] == "tracker"
    assert sent_image().size == (512, 384), "aspect ratio must be preserved"


def test_empty_detections(frame):
    reply["BOXES"] = np.empty((0,), dtype=np.float32)
    reply["SCORES"] = np.empty((0,), dtype=np.float32)
    reply["LABELS"] = np.empty((0,), dtype=object)
    assert RemoteDetector("spark:8101", width=512).detect(frame, ["cat"], 0.4) == []


def test_unreachable_server_raises(frame):
    reply["_down"] = True
    with pytest.raises(DetectorUnavailable):
        RemoteDetector("spark:8101").detect(frame, ["cat"], 0.4)


def test_classes_raises_when_server_is_down():
    # The caller keeps the last known list; the detector does not invent one.
    reply["_down"] = True
    with pytest.raises(DetectorUnavailable):
        RemoteDetector("spark:8101").classes()


def test_classes_raises_when_the_model_is_not_loaded():
    reply["_not_ready"] = True
    with pytest.raises(DetectorUnavailable):
        RemoteDetector("spark:8101").classes()


def test_classes_reports_the_coco_vocabulary():
    assert RemoteDetector("spark:8101").classes() == list(COCO_CLASSES)


def test_a_pasted_scheme_is_tolerated():
    RemoteDetector("http://spark:8101/").classes()
    assert seen["url"] == "spark:8101"


def test_default_payload_matches_the_model_resolution(frame):
    # 640 is YOLO's native size; sending less means handing it an upscale.
    RemoteDetector("spark:8101").detect(frame, ["cat"], 0.4)
    assert sent_image().width == 640
