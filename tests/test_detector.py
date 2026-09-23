import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
from PIL import Image
from tritonclient.utils import InferenceServerException

import tracker.detector as detector
from tracker.detector import (
    BACKENDS,
    COCO_CLASSES,
    BuiltinDetector,
    DetectorUnavailable,
    TritonDetector,
    VlmDetector,
    make_detector,
)


@pytest.fixture
def frame():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


# --- Triton backend, over a fake gRPC client ---------------------------------

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


def sent_image():
    """Decode the JPEG the detector put on the wire."""
    return Image.open(io.BytesIO(seen["inputs"]["JPEG"].data.tobytes()))


def test_triton_boxes_are_rescaled_to_source_frame(frame):
    # Server answers in 512-wide space; 640/512 = 1.25.
    reply["BOXES"] = np.array([[80, 40, 160, 120]], dtype=np.float32)
    reply["SCORES"] = np.array([0.8], dtype=np.float32)
    reply["LABELS"] = np.array([b"cat"], dtype=object)

    det = TritonDetector("spark:8101", width=512).detect(frame, ["cat"], 0.4)[0]
    assert det.box == pytest.approx((100, 50, 200, 150))
    assert det.center == pytest.approx((150, 100))
    assert det.label == "cat"


def test_triton_small_frames_are_not_upscaled():
    reply["BOXES"] = np.array([[10, 10, 20, 20]], dtype=np.float32)
    reply["SCORES"] = np.array([0.8], dtype=np.float32)
    reply["LABELS"] = np.array([b"cat"], dtype=object)

    small = np.zeros((120, 160, 3), dtype=np.uint8)
    det = TritonDetector("spark:8101", width=512).detect(small, ["cat"], 0.4)[0]
    assert sent_image().width == 160, "must not waste bytes upscaling"
    assert det.box == pytest.approx((10, 10, 20, 20))


def test_triton_request_carries_labels_and_confidence(frame):
    TritonDetector("spark:8101", width=512).detect(frame, ["cat", "dog"], 0.65)

    keep = [
        s.decode() if isinstance(s, bytes) else s for s in seen["inputs"]["KEEP"].data
    ]
    assert keep == ["cat", "dog"]
    assert seen["inputs"]["CONF"].data == pytest.approx([0.65])
    assert seen["model"] == "tracker"
    assert sent_image().size == (512, 384), "aspect ratio must be preserved"


def test_triton_empty_detections(frame):
    reply["BOXES"] = np.empty((0,), dtype=np.float32)
    reply["SCORES"] = np.empty((0,), dtype=np.float32)
    reply["LABELS"] = np.empty((0,), dtype=object)
    assert TritonDetector("spark:8101", width=512).detect(frame, ["cat"], 0.4) == []


def test_triton_unreachable_server_raises(frame):
    reply["_down"] = True
    with pytest.raises(DetectorUnavailable):
        TritonDetector("spark:8101").detect(frame, ["cat"], 0.4)


def test_triton_classes_raises_when_server_is_down():
    # The caller keeps the last known list; the detector does not invent one.
    reply["_down"] = True
    with pytest.raises(DetectorUnavailable):
        TritonDetector("spark:8101").classes()


def test_triton_classes_raises_when_the_model_is_not_loaded():
    reply["_not_ready"] = True
    with pytest.raises(DetectorUnavailable):
        TritonDetector("spark:8101").classes()


def test_triton_classes_reports_the_coco_vocabulary():
    assert TritonDetector("spark:8101").classes() == list(COCO_CLASSES)


def test_triton_a_pasted_scheme_is_tolerated():
    TritonDetector("http://spark:8101/").classes()
    assert seen["url"] == "spark:8101"


def test_triton_default_payload_matches_the_model_resolution(frame):
    # 640 is YOLO's native size; sending less means handing it an upscale.
    TritonDetector("spark:8101").detect(frame, ["cat"], 0.4)
    assert sent_image().width == 640


# --- Built-in backend, over a real local HTTP server -------------------------

# Filled in by each test with what the stub should return; captured requests
# land in `http_seen` so tests can assert on what went over the wire.
http_reply: dict = {}
http_seen: dict = {}


class Stub(BaseHTTPRequestHandler):
    def do_GET(self):
        self._send({"classes": ["person", "cat"]})

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        http_seen["query"] = parse_qs(urlparse(self.path).query)
        http_seen["image"] = Image.open(io.BytesIO(body))
        self._send(http_reply)

    def _send(self, payload):
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    http_reply.clear()
    http_seen.clear()
    httpd = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_builtin_boxes_are_rescaled_to_source_frame(server, frame):
    # Stub answers in 512-wide space; 640/512 = 1.25.
    http_reply["detections"] = [
        {"label": "cat", "conf": 0.8, "box": [80, 40, 160, 120]}
    ]

    det = BuiltinDetector(server, width=512).detect(frame, ["cat"], 0.4)[0]
    assert det.box == pytest.approx((100, 50, 200, 150))
    assert det.label == "cat"


def test_builtin_request_carries_labels_and_confidence(server, frame):
    http_reply["detections"] = []

    BuiltinDetector(server, width=512).detect(frame, ["cat", "dog"], 0.65)
    assert http_seen["query"]["labels"] == ["cat,dog"]
    assert http_seen["query"]["conf"] == ["0.65"]
    assert http_seen["image"].size == (512, 384), "aspect ratio must be preserved"


def test_builtin_empty_detections(server, frame):
    http_reply["detections"] = []
    assert BuiltinDetector(server, width=512).detect(frame, ["cat"], 0.4) == []


def test_builtin_unreachable_server_raises(frame):
    with pytest.raises(DetectorUnavailable):
        BuiltinDetector("http://127.0.0.1:1", timeout=0.3).detect(frame, ["cat"], 0.4)


def test_builtin_classes_prefers_the_server(server):
    assert BuiltinDetector(server).classes() == ["person", "cat"]


def test_builtin_classes_raises_when_server_is_down():
    with pytest.raises(DetectorUnavailable):
        BuiltinDetector("http://127.0.0.1:1", timeout=0.3).classes()


# --- VLM backend, over a fake OpenAI-compatible server -----------------------

# The model's reply text and the ids it lists; the last request body lands in
# `vlm_seen` so tests can assert on what was asked.
vlm_reply: dict = {}
vlm_seen: dict = {}


class VlmStub(BaseHTTPRequestHandler):
    def do_GET(self):
        ids = vlm_reply.get("models", ["served-model"])
        self._send({"object": "list", "data": [{"id": i} for i in ids]})

    def do_POST(self):
        vlm_seen["body"] = json.loads(
            self.rfile.read(int(self.headers["Content-Length"]))
        )
        content = vlm_reply.get("content", "[]")
        self._send({"choices": [{"message": {"content": content}}]})

    def _send(self, payload):
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture
def vlm_server():
    vlm_reply.clear()
    vlm_seen.clear()
    httpd = HTTPServer(("127.0.0.1", 0), VlmStub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_vlm_maps_normalised_boxes_to_source_pixels(vlm_server, frame):
    # frame is 640x480; [0,1000] maps 500->320 in x and 1000->480 in y.
    vlm_reply["content"] = '[{"label": "cat", "box": [0, 0, 500, 1000]}]'
    det = VlmDetector(vlm_server).detect(frame, ["cat"], 0.4)[0]
    assert det.box == pytest.approx((0, 0, 320, 480))
    assert det.label == "cat"


def test_vlm_reads_fenced_json_and_bbox_2d(vlm_server, frame):
    vlm_reply["content"] = (
        '```json\n[{"bbox_2d": [0, 0, 1000, 1000], "label": "dog"}]\n```'
    )
    det = VlmDetector(vlm_server).detect(frame, ["dog"], 0.4)[0]
    assert det.box == pytest.approx((0, 0, 640, 480))


def test_vlm_keeps_only_requested_labels(vlm_server, frame):
    vlm_reply["content"] = (
        '[{"label": "cat", "box": [0,0,100,100]},'
        ' {"label": "sofa", "box": [0,0,100,100]}]'
    )
    dets = VlmDetector(vlm_server).detect(frame, ["cat"], 0.4)
    assert [d.label for d in dets] == ["cat"]


def test_vlm_applies_the_confidence_threshold(vlm_server, frame):
    # a scored box below the threshold drops; an unscored one is kept.
    vlm_reply["content"] = (
        '[{"label": "cat", "box": [0,0,100,100], "confidence": 0.2},'
        ' {"label": "cat", "box": [0,0,100,100]}]'
    )
    dets = VlmDetector(vlm_server).detect(frame, ["cat"], 0.4)
    assert [d.conf for d in dets] == [1.0]


def test_vlm_prompts_the_discovered_model(vlm_server, frame):
    vlm_reply["models"] = ["Qwen/Qwen3.5-0.8B", "alias"]
    VlmDetector(vlm_server).detect(frame, ["cat"], 0.4)
    assert vlm_seen["body"]["model"] == "Qwen/Qwen3.5-0.8B"
    content = vlm_seen["body"]["messages"][0]["content"]
    assert any(part["type"] == "image_url" for part in content)
    assert "cat" in next(p["text"] for p in content if p["type"] == "text")


def test_vlm_junk_reply_yields_no_detections(vlm_server, frame):
    vlm_reply["content"] = "I could not find anything in this image."
    assert VlmDetector(vlm_server).detect(frame, ["cat"], 0.4) == []


def test_vlm_unreachable_server_raises(frame):
    with pytest.raises(DetectorUnavailable):
        VlmDetector("http://127.0.0.1:1", timeout=0.3).detect(frame, ["cat"], 0.4)


def test_vlm_url_tolerates_a_trailing_v1(vlm_server):
    assert VlmDetector(vlm_server + "/v1").classes() == list(COCO_CLASSES)


# --- Backend registry --------------------------------------------------------


def test_make_detector_builds_each_backend():
    assert isinstance(make_detector("triton", "spark:8101"), TritonDetector)
    assert isinstance(make_detector("builtin", "http://x:8100"), BuiltinDetector)
    assert isinstance(make_detector("vlm", "http://x:8000"), VlmDetector)


def test_make_detector_rejects_an_unknown_backend():
    with pytest.raises(ValueError):
        make_detector("nope", "x")


def test_every_backend_has_a_label_and_default_url():
    for key, backend in BACKENDS.items():
        assert backend.key == key
        assert backend.label and backend.default_url
