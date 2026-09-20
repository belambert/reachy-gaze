import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
from PIL import Image

from tracker.detector import DetectorUnavailable, RemoteDetector

# Filled in by each test with what the stub should return; captured requests land
# in `seen` so tests can assert on what actually went over the wire.
reply: dict = {}
seen: dict = {}


class Stub(BaseHTTPRequestHandler):
    def do_GET(self):
        self._send({"classes": ["person", "cat"]})

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        seen["query"] = parse_qs(urlparse(self.path).query)
        seen["image"] = Image.open(io.BytesIO(body))
        self._send(reply)

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
    httpd = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture
def frame():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def test_boxes_are_rescaled_to_source_frame(server, frame):
    # Stub answers in 512-wide space; 640/512 = 1.25.
    reply.clear()
    reply.update(
        {"detections": [{"label": "cat", "conf": 0.8, "box": [80, 40, 160, 120]}]}
    )

    det = RemoteDetector(server, width=512).detect(frame, ["cat"], 0.4)[0]
    assert det.box == pytest.approx((100, 50, 200, 150))
    assert det.center == pytest.approx((150, 100))
    assert det.label == "cat"


def test_small_frames_are_not_upscaled(server):
    reply.clear()
    reply.update(
        {"detections": [{"label": "cat", "conf": 0.8, "box": [10, 10, 20, 20]}]}
    )

    small = np.zeros((120, 160, 3), dtype=np.uint8)
    det = RemoteDetector(server, width=512).detect(small, ["cat"], 0.4)[0]
    assert seen["image"].width == 160, "must not waste bytes upscaling"
    assert det.box == pytest.approx((10, 10, 20, 20))


def test_request_carries_labels_and_confidence(server, frame):
    reply.clear()
    reply.update({"detections": []})

    RemoteDetector(server, width=512).detect(frame, ["cat", "dog"], 0.65)
    assert seen["query"]["labels"] == ["cat,dog"]
    assert seen["query"]["conf"] == ["0.65"]
    assert seen["image"].size == (512, 384), "aspect ratio must be preserved"


def test_empty_detections(server, frame):
    reply.clear()
    reply.update({"detections": []})
    assert RemoteDetector(server, width=512).detect(frame, ["cat"], 0.4) == []


def test_unreachable_server_raises(frame):
    detector = RemoteDetector("http://127.0.0.1:1", timeout=0.3)
    with pytest.raises(DetectorUnavailable):
        detector.detect(frame, ["cat"], 0.4)


def test_classes_raises_when_server_is_down():
    # The caller keeps the last known list; the detector does not invent one.
    with pytest.raises(DetectorUnavailable):
        RemoteDetector("http://127.0.0.1:1", timeout=0.3).classes()


def test_classes_prefers_the_server(server):
    assert RemoteDetector(server).classes() == ["person", "cat"]


def test_default_payload_matches_the_model_resolution(server, frame):
    # 640 is YOLO's native size; sending less means handing it an upscale.
    reply.clear()
    reply.update({"detections": []})

    RemoteDetector(server).detect(frame, ["cat"], 0.4)
    assert seen["image"].width == 640
