"""Track a named COCO object with Reachy Mini's head.

Detection runs off-board (see ``tracker.server``) because the Pi's CPU is
already busy with motor control. Two rates keep that from showing: the vision
thread re-anchors the target a handful of times a second, while the control loop
slews the head toward that anchor at 50 Hz.
"""

from __future__ import annotations

import logging
import math
import threading
import time

import numpy as np
from pydantic import BaseModel
from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.vision.look_at import look_at_image_pose
from scipy.spatial.transform import Rotation

from tracker.detector import COCO_CLASSES, DetectorUnavailable, RemoteDetector
from tracker.tracking import (
    CenterFilter,
    TargetSelector,
    norm_center,
    pixel_center,
    pose_matrix,
    slew,
)

CONTROL_HZ = 50.0
DETECT_HZ = 12.0  # request ceiling; the server is usually quicker than this
SLEW_TAU = 0.15  # seconds to close ~63% of the angular error
LOST_AFTER = 1.5  # seconds without a detection before giving up the lock
RETRY_AFTER = 2.0  # seconds to wait out an unreachable detection server

logger = logging.getLogger(__name__)


class Config(BaseModel):
    """Settings the control panel can change while the app runs."""

    enabled: bool | None = None
    label: str | None = None
    conf: float | None = None
    server_url: str | None = None
    scan: bool | None = None


class State:
    """Shared state between the vision thread and the control loop."""

    def __init__(self) -> None:
        """Start disabled, aimed at nothing, with no server contacted yet."""
        self.lock = threading.Lock()

        self.enabled = False
        self.label = "person"
        self.conf = 0.4
        self.server_url = "http://192.168.1.10:8100"
        self.scan = True

        self.goal: Rotation | None = None
        self.head_pose = np.eye(4)
        self.last_seen = 0.0
        self.detector_ok = False
        self.error = ""
        self.fps = 0.0
        self.center: tuple[float, float] | None = None

    def snapshot(self) -> dict:
        """Everything the control panel polls, in one consistent read."""
        with self.lock:
            locked = (
                self.goal is not None and time.monotonic() - self.last_seen < LOST_AFTER
            )
            return {
                "enabled": self.enabled,
                "label": self.label,
                "conf": self.conf,
                "server_url": self.server_url,
                "scan": self.scan,
                "locked": locked,
                "detector_ok": self.detector_ok,
                "error": self.error,
                "fps": round(self.fps, 1),
                "center": self.center,
            }


class Tracker(ReachyMiniApp):
    """Point the head at whichever COCO class the control panel asks for."""

    custom_app_url: str | None = "http://0.0.0.0:8042"
    request_media_backend: str | None = "gstreamer"

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Serve the control panel, run the vision thread, and drive the head."""
        camera = reachy_mini.media.camera
        if camera is None or camera.K is None:
            raise RuntimeError("Tracking needs a calibrated camera.")

        state = State()
        self._mount_api(state)

        # Let the body carry the head past its own yaw limit.
        reachy_mini.set_automatic_body_yaw(True)

        vision = threading.Thread(
            target=self._track,
            args=(reachy_mini, state, stop_event),
            daemon=True,
            name="tracker-vision",
        )
        vision.start()
        try:
            self._drive(reachy_mini, state, stop_event)
        finally:
            vision.join(timeout=2.0)

    def _drive(
        self, mini: ReachyMini, state: State, stop_event: threading.Event
    ) -> None:
        """Slew the head toward the current goal at a steady rate."""
        period = 1.0 / CONTROL_HZ
        current = Rotation.identity()
        t0 = time.monotonic()
        last = t0
        tick = 0

        while not stop_event.is_set():
            now = time.monotonic()
            dt = min(now - last, 0.1)  # a scheduling hiccup must not cause a lurch
            last = now

            # Refresh the cached actual pose at ~10 Hz; the vision thread aims
            # against it and every SDK call belongs on this thread.
            if tick % 5 == 0:
                with state.lock:
                    state.head_pose = mini.get_current_head_pose()
            tick += 1

            with state.lock:
                goal = state.goal
                fresh = goal is not None and now - state.last_seen < LOST_AFTER
                scanning = state.enabled and state.scan and not fresh

            if not fresh:
                goal = self._idle_pose(now - t0) if scanning else Rotation.identity()

            assert goal is not None
            current = slew(current, goal, 1.0 - math.exp(-dt / SLEW_TAU))
            mini.set_target(
                head=pose_matrix(current),
                antennas=self._antennas(fresh, now - t0),
            )
            time.sleep(period)

    def _track(
        self, mini: ReachyMini, state: State, stop_event: threading.Event
    ) -> None:
        """Detect the requested class and turn each hit into an absolute head pose."""
        camera = mini.media.camera
        assert camera is not None
        K, D = camera.K, camera.D
        T_head_cam = getattr(mini, "T_head_cam", None)

        selector = TargetSelector()
        smoother = CenterFilter()
        detector: RemoteDetector | None = None
        detector_url = ""
        period = 1.0 / DETECT_HZ

        while not stop_event.is_set():
            started = time.monotonic()
            with state.lock:
                enabled, label, conf, url = (
                    state.enabled,
                    state.label,
                    state.conf,
                    state.server_url,
                )
                head_pose = state.head_pose

            if not enabled or not label:
                selector.reset()
                smoother.reset()
                stop_event.wait(0.2)
                continue

            if detector is None or url != detector_url:
                detector = RemoteDetector(url)
                detector_url = url
                selector.reset()
                smoother.reset()

            frame = mini.media.get_frame()
            if frame is None:
                stop_event.wait(period)
                continue

            try:
                dets = detector.detect(frame, [label], conf)
            except DetectorUnavailable as e:
                with state.lock:
                    state.detector_ok = False
                    state.error = str(e)
                    state.fps = 0.0
                logger.warning("Detector unreachable: %s", e)
                stop_event.wait(RETRY_AFTER)
                continue

            height, width = frame.shape[:2]
            det = selector.select(dets, width, height)
            if det is None and not selector.has_target:
                smoother.reset()

            with state.lock:
                state.detector_ok = True
                state.error = ""
                # Measured rate matters more than the cap; the panel shows it.
                state.fps += 0.2 * (
                    1.0 / max(time.monotonic() - started, 1e-3) - state.fps
                )
                if det is not None:
                    center = smoother.update(norm_center(det, width, height))
                    u, v = pixel_center(center, width, height)
                    # Aim against the pose the frame was captured at, so a late
                    # detection still points where the target actually was.
                    state.goal = Rotation.from_matrix(
                        look_at_image_pose(u, v, K, D, head_pose, T_head_cam)[:3, :3]
                    )
                    state.last_seen = time.monotonic()
                    state.center = center

            elapsed = time.monotonic() - started
            if elapsed < period:
                stop_event.wait(period - elapsed)

    @staticmethod
    def _idle_pose(t: float) -> Rotation:
        """A slow yaw sweep, so a lost target has a chance of wandering back in."""
        return Rotation.from_euler(
            "z", 35.0 * math.sin(2 * math.pi * 0.08 * t), degrees=True
        )

    @staticmethod
    def _antennas(locked: bool, t: float) -> np.ndarray:
        """Perked up when locked on, idly wagging when searching."""
        if locked:
            return np.deg2rad([20.0, -20.0])
        a = 12.0 * math.sin(2 * math.pi * 0.3 * t)
        return np.deg2rad([a, -a])

    def _mount_api(self, state: State) -> None:
        """Expose the control panel's read and write endpoints."""
        if self.settings_app is None:
            return

        @self.settings_app.get("/state")
        def get_state() -> dict:
            return state.snapshot()

        @self.settings_app.get("/classes")
        def get_classes() -> dict:
            return {"classes": COCO_CLASSES}

        @self.settings_app.post("/config")
        def set_config(config: Config) -> dict:
            with state.lock:
                for field, value in config.model_dump(exclude_none=True).items():
                    setattr(state, field, value)
                # A new class means the old lock is meaningless.
                if config.label is not None or config.enabled is False:
                    state.goal = None
                    state.center = None
            return state.snapshot()


if __name__ == "__main__":
    app = Tracker()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
