"""Track a named COCO object with Reachy Mini's head.

Detection runs off-board (see ``tracker.server``) because the Pi's CPU is
already busy with motor control. Two rates keep that from showing: the vision
thread re-anchors the target a handful of times a second, while the control loop
slews the head toward that anchor at 50 Hz.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time

import numpy as np
from pydantic import BaseModel, Field
from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.vision.look_at import look_at_image_pose
from scipy.spatial.transform import Rotation

from tracker.detector import COCO_CLASSES, DetectorUnavailable, RemoteDetector
from tracker.tracking import (
    CenterFilter,
    PoseSmoother,
    TargetSelector,
    norm_center,
    pixel_center,
    pose_matrix,
)

CONTROL_HZ = 50.0
DETECT_HZ = 12.0  # request ceiling; the server is usually quicker than this
SMOOTH_TAU = 0.09  # follower time constant; larger is smoother and laggier
MAX_HEAD_SPEED = 3.5  # rad/s ceiling on commanded head rotation
MAX_HEAD_PULL = 20.0  # rad/s^2 ceiling on the follower's pull; lower is gentler
BLEND_TAU = 0.4  # seconds to ease between searching and locked-on posture
LOST_AFTER = 10.0  # seconds holding the last aim point before giving up
STALE_AFTER = 1.0  # seconds before the panel calls the lock stale rather than live
RETRY_AFTER = 2.0  # seconds to wait out an unreachable detection server

# Prefilled in the control panel. Override without editing code by setting
# TRACKER_SERVER_URL; a DHCP lease will eventually make this one wrong.
DEFAULT_SERVER_URL = os.environ.get("TRACKER_SERVER_URL", "http://10.0.0.206:8100")

logger = logging.getLogger(__name__)


class Config(BaseModel):
    """Settings the control panel can change while the app runs."""

    enabled: bool | None = None
    label: str | None = None
    conf: float | None = None
    server_url: str | None = None
    scan: bool | None = None
    # A pull of zero would freeze the head, so refuse it rather than obey it.
    pull: float | None = Field(None, gt=0.0, le=200.0)


class State:
    """Shared state between the vision thread and the control loop."""

    def __init__(self) -> None:
        """Start disabled, aimed at nothing, with no server contacted yet."""
        self.lock = threading.Lock()

        self.enabled = False
        self.label = "person"
        self.conf = 0.4
        self.server_url = DEFAULT_SERVER_URL
        self.scan = True
        self.pull = MAX_HEAD_PULL

        self.goal: Rotation | None = None
        self.head_pose = np.eye(4)
        self.last_seen = 0.0
        self.detector_ok = False
        self.error = ""
        self.fps = 0.0
        self.center: tuple[float, float] | None = None
        # Seeded so the picker works before any server has been reached.
        self.classes = list(COCO_CLASSES)
        # Bumped whenever the list actually changes, so the panel knows to
        # re-read it rather than polling 80 strings several times a second.
        self.classes_version = 0

    def snapshot(self) -> dict:
        """Everything the control panel polls, in one consistent read."""
        with self.lock:
            seen_ago = (
                time.monotonic() - self.last_seen if self.goal is not None else None
            )
            locked = seen_ago is not None and seen_ago < LOST_AFTER
            return {
                "enabled": self.enabled,
                "label": self.label,
                "conf": self.conf,
                "server_url": self.server_url,
                "scan": self.scan,
                "pull": self.pull,
                "locked": locked,
                "detector_ok": self.detector_ok,
                "error": self.error,
                "fps": round(self.fps, 1),
                "center": self.center,
                "classes_version": self.classes_version,
                "seen_ago": round(seen_ago, 1) if locked else None,
            }


class Tracker(ReachyMiniApp):
    """Point the head at whichever COCO class the control panel asks for."""

    custom_app_url: str | None = "http://0.0.0.0:8042"
    request_media_backend: str | None = "gstreamer"

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Serve the control panel, run the vision thread, and drive the head."""
        # Mount before anything that can fail: a panel whose endpoints 404 looks
        # to the browser like an app with no classes and no state at all.
        state = State()
        self._mount_api(state)

        camera = reachy_mini.media.camera
        if camera is None or camera.K is None:
            state.error = "Tracking needs a calibrated camera."
            raise RuntimeError(state.error)

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
        """Follow the current goal, keeping commanded velocity continuous."""
        logger.info("Control loop running at %.0f Hz", CONTROL_HZ)
        period = 1.0 / CONTROL_HZ
        smoother = PoseSmoother(SMOOTH_TAU, MAX_HEAD_SPEED, MAX_HEAD_PULL)
        lock_level = 0.0
        t0 = time.monotonic()
        last = t0
        next_tick = t0

        while not stop_event.is_set():
            now = time.monotonic()
            dt = min(now - last, 0.1)  # a scheduling hiccup must not cause a lurch
            last = now

            # Cheap: the SDK's receive thread keeps this up to date for us.
            # It asserts until the daemon has published a first pose, though,
            # which at startup would otherwise take the whole app down.
            try:
                pose = mini.get_current_head_pose()
            except AssertionError:
                pose = None

            with state.lock:
                if pose is not None:
                    state.head_pose = pose
                goal = state.goal
                fresh = goal is not None and now - state.last_seen < LOST_AFTER
                scanning = state.enabled and state.scan and not fresh
                smoother.max_pull = state.pull  # tunable live from the panel

            # One eased scalar drives both postures, so nothing steps on a
            # transition: antennas blend, and the sweep grows in rather than
            # starting at full swing.
            lock_level += (float(fresh) - lock_level) * (
                1.0 - math.exp(-dt / BLEND_TAU)
            )

            if not fresh:
                goal = (
                    self._idle_pose(now - t0, 1.0 - lock_level)
                    if scanning
                    else Rotation.identity()
                )

            assert goal is not None
            mini.set_target(
                head=pose_matrix(smoother.step(goal, dt)),
                antennas=self._antennas(lock_level, now - t0),
            )

            # Absolute deadlines: sleeping a fixed period would let the loop
            # drift by however long the work took, jittering the command rate.
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()  # fell behind; resync, don't spin

    def _track(
        self, mini: ReachyMini, state: State, stop_event: threading.Event
    ) -> None:
        """Detect the requested class and turn each hit into an absolute head pose."""
        try:
            self._track_forever(mini, state, stop_event)
        except Exception as e:
            logger.exception("Vision thread died")
            with state.lock:
                state.error = f"Vision thread died: {e}"
            raise

    def _track_forever(
        self, mini: ReachyMini, state: State, stop_event: threading.Event
    ) -> None:
        """Body of the vision thread."""
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

            # Ahead of the enabled check: pointing at a new server should
            # refresh the picker straight away, not on the next tracking run.
            if detector is None or url != detector_url:
                detector = RemoteDetector(url)
                detector_url = url
                selector.reset()
                smoother.reset()
                self._refresh_classes(detector, state)

            if not enabled or not label:
                selector.reset()
                smoother.reset()
                stop_event.wait(0.2)
                continue

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
    def _refresh_classes(detector: RemoteDetector, state: State) -> None:
        """Adopt the server's vocabulary, keeping the last known one on failure."""
        try:
            classes = detector.classes()
        except DetectorUnavailable as e:
            logger.warning("Could not fetch class list: %s", e)
            return
        with state.lock:
            if classes != state.classes:
                state.classes = classes
                state.classes_version += 1
        logger.info("Fetched %d classes from %s", len(classes), detector.url)

    @staticmethod
    def _idle_pose(t: float, scale: float) -> Rotation:
        """A slow yaw sweep, so a lost target has a chance of wandering back in."""
        return Rotation.from_euler(
            "z", scale * 35.0 * math.sin(2 * math.pi * 0.08 * t), degrees=True
        )

    @staticmethod
    def _antennas(lock_level: float, t: float) -> np.ndarray:
        """Wagging while searching, perked up when locked, blended in between."""
        wag = 12.0 * math.sin(2 * math.pi * 0.3 * t)
        angle = wag + lock_level * (20.0 - wag)
        return np.deg2rad([angle, -angle])

    def _mount_api(self, state: State) -> None:
        """Expose the control panel's read and write endpoints."""
        if self.settings_app is None:
            return

        @self.settings_app.get("/state")
        def get_state() -> dict:
            return state.snapshot()

        @self.settings_app.get("/classes")
        def get_classes() -> dict:
            with state.lock:
                return {"classes": list(state.classes)}

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
