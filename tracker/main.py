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
from pydantic import BaseModel, Field, field_validator
from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.vision.look_at import look_at_image_pose
from scipy.spatial.transform import Rotation

from tracker.detector import BACKENDS, Detector, DetectorUnavailable, make_detector
from tracker.tracking import (
    CenterFilter,
    PoseSmoother,
    TargetSelector,
    norm_center,
    pixel_center,
    pose_matrix,
)

# Hunted for together, and in preference order: a cat in view outranks a person
# in view, and the head will leave the one for the other.
TRACK_LABELS = ["cat", "dog", "bird", "person"]

CONTROL_HZ = 50.0
DETECT_HZ = 12.0  # request ceiling; the server is usually quicker than this
SMOOTH_TAU = 0.09  # follower time constant; larger is smoother and laggier
MAX_HEAD_SPEED = 3.5  # rad/s ceiling on commanded head rotation
MAX_HEAD_PULL = 10.0  # rad/s^2 ceiling on the follower's pull; lower is gentler
BLEND_TAU = 0.4  # seconds to ease between searching and locked-on posture
# The body's yaw joint allows +/-160 degrees; this keeps the scan to the front
# of the robot, well inside that.
SCAN_DEGREES = 90.0  # half-width of the yaw scan
SCAN_HZ = 0.04  # yaw scan rate; peak speed is 2*pi*SCAN_HZ*SCAN_DEGREES
LOST_AFTER = 10.0  # seconds holding the last aim point before giving up
LOCK_TIMEOUT = 15.0  # seconds on one target before breaking off to scan for others
LOOK_AWAY = 10.0  # seconds steering clear of the abandoned target while scanning
STALE_AFTER = 5.0  # seconds before the panel calls the lock stale rather than live
RETRY_AFTER = 2.0  # seconds to wait out an unreachable detection server

# Which detection backend to use, and where to reach it. Both are prefilled in
# the control panel and overridable from the environment without editing code.
DEFAULT_BACKEND = os.environ.get("TRACKER_BACKEND", "triton")
DEFAULT_SERVER_URL = os.environ.get(
    "TRACKER_SERVER_URL", BACKENDS[DEFAULT_BACKEND].default_url
)

logger = logging.getLogger(__name__)


def look_yaw_pitch(head_pose: np.ndarray) -> tuple[float, float]:
    """Head aim as (yaw, pitch) degrees: +yaw is left, +pitch is up, 0 is ahead.

    Reads the head's forward axis (the pose rotation's +X column, which the SDK
    aims at the target) in the world frame, where +X is forward, +Y left, +Z up.
    This is the absolute aim: `get_current_head_pose` is the forward kinematics
    of all seven joints, body yaw included, so the turntable's contribution is
    already in here — do not add body yaw again.
    """
    fwd = head_pose[:3, 0]
    yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, fwd[2]))))
    return yaw, pitch


class Config(BaseModel):
    """Settings the control panel can change while the app runs."""

    enabled: bool | None = None
    conf: float | None = None
    server_url: str | None = None
    backend: str | None = None
    scan: bool | None = None
    # A pull of zero would freeze the head, so refuse it rather than obey it.
    pull: float | None = Field(None, gt=0.0, le=200.0)

    @field_validator("backend")
    @classmethod
    def _known_backend(cls, v: str | None) -> str | None:
        if v is not None and v not in BACKENDS:
            raise ValueError(f"unknown backend {v!r}")
        return v


class State:
    """Shared state between the vision thread and the control loop."""

    def __init__(self) -> None:
        """Start tracking, aimed at nothing, with no server contacted yet."""
        self.lock = threading.Lock()

        # On by default: starting the app is the instruction to track, and a
        # restart used to come back silently unticked.
        self.enabled = True
        # What the head is currently locked onto, not what it is looking for.
        self.label = ""
        self.conf = 0.75
        self.server_url = DEFAULT_SERVER_URL
        self.backend = DEFAULT_BACKEND
        self.scan = True
        self.pull = MAX_HEAD_PULL

        self.goal: Rotation | None = None
        self.head_pose = np.eye(4)  # rotation & position
        self.last_seen = 0.0
        self.detector_ok = False
        self.error = ""
        self.fps = 0.0
        self.center: tuple[float, float] | None = None
        # Every other box in view, so the panel can show what the head is
        # ignoring: {"label", "center": [x, y]} in the same normalized coords.
        self.targets: list[dict] = []

    def snapshot(self) -> dict:
        """Everything the control panel polls, in one consistent read."""
        with self.lock:
            seen_ago = (
                time.monotonic() - self.last_seen if self.goal is not None else None
            )
            locked = seen_ago is not None and seen_ago < LOST_AFTER
            yaw, pitch = look_yaw_pitch(self.head_pose)
            return {
                "enabled": self.enabled,
                "labels": TRACK_LABELS,
                "label": self.label,
                "conf": self.conf,
                "server_url": self.server_url,
                "backend": self.backend,
                "backends": [
                    {"key": k, "label": b.label, "default_url": b.default_url}
                    for k, b in BACKENDS.items()
                ],
                "scan": self.scan,
                "pull": self.pull,
                "locked": locked,
                "detector_ok": self.detector_ok,
                "error": self.error,
                "fps": round(self.fps, 1),
                "center": self.center,
                "targets": self.targets,
                "aim": {"yaw": round(yaw, 1), "pitch": round(pitch, 1)},
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
        perk = 0.0
        was_scanning = False
        scan_t0 = 0.0
        scan_phase = 0.0
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
                # Not "seen just now": the aim is held for the whole of
                # LOST_AFTER, so this stays true long after the last sighting.
                aimed = goal is not None and now - state.last_seen < LOST_AFTER
                scanning = state.enabled and state.scan and not aimed
                smoother.max_pull = state.pull  # tunable live from the panel

            # 0 searching, 1 locked on, eased so the antennas never snap.
            perk += (float(aimed) - perk) * (1.0 - math.exp(-dt / BLEND_TAU))

            # Start each scan from wherever the head already is, carrying on the
            # way it was turning. Running the sine off a fixed epoch meant it
            # began at an arbitrary phase, so losing a target swung the head to
            # centre and then out again; ignoring the direction meant it always
            # set off the same way and left one side of the room unswept.
            if scanning and not was_scanning:  # a scan starts on this tick
                scan_t0 = now
                scan_phase = self._scan_phase(smoother.rotation, smoother.omega[2])
            was_scanning = scanning

            if not aimed:
                goal = (
                    self._scan_pose(now - scan_t0, scan_phase)
                    if scanning
                    else Rotation.identity()
                )

            assert goal is not None
            mini.set_target(
                head=pose_matrix(smoother.step(goal, dt)),
                antennas=self._antennas(perk, now - t0),
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

        selector = TargetSelector(
            TRACK_LABELS, max_lock=LOCK_TIMEOUT, look_away=LOOK_AWAY
        )
        smoother = CenterFilter()
        detector: Detector | None = None
        detector_key: tuple[str, str] | None = None
        period = 1.0 / DETECT_HZ

        while not stop_event.is_set():
            started = time.monotonic()
            with state.lock:
                enabled, conf = state.enabled, state.conf
                backend, url = state.backend, state.server_url
                head_pose = state.head_pose

            # Ahead of the enabled check: a new backend or server should be
            # vetted straight away, not on the next tracking run.
            if detector is None or (backend, url) != detector_key:
                detector = make_detector(backend, url)
                detector_key = (backend, url)
                selector.reset()
                smoother.reset()
                self._check_vocabulary(detector)

            if not enabled:
                selector.reset()
                smoother.reset()
                with state.lock:
                    state.targets = []
                stop_event.wait(0.2)
                continue

            frame = mini.media.get_frame()
            if frame is None:
                stop_event.wait(period)
                continue

            try:
                dets = detector.detect(frame, TRACK_LABELS, conf)
            except DetectorUnavailable as e:
                with state.lock:
                    state.detector_ok = False
                    state.error = str(e)
                    state.fps = 0.0
                    state.targets = []
                logger.warning("Detector unreachable: %s", e)
                stop_event.wait(RETRY_AFTER)
                continue

            height, width = frame.shape[:2]
            # World direction of each box, so the selector can shun a target it
            # tired of by where it is rather than by a pixel the scan moves.
            dirs = [self._direction(d, K, D, head_pose, T_head_cam) for d in dets]
            det = selector.select(dets, width, height, dirs)
            if det is None and not selector.has_target:
                smoother.reset()

            with state.lock:
                state.detector_ok = True
                state.error = ""
                # Measured rate matters more than the cap; the panel shows it.
                state.fps += 0.2 * (
                    1.0 / max(time.monotonic() - started, 1e-3) - state.fps
                )
                # Every box but the one we aim at, for the panel to show as the
                # others in view; the primary is carried by `center` instead.
                state.targets = [
                    {"label": d.label, "center": list(norm_center(d, width, height))}
                    for d in dets
                    if d is not det
                ]
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
                    state.label = det.label
                elif selector.looking_away:
                    # Bored of the last target: drop the aim now so the head
                    # scans, rather than holding it for the whole LOST_AFTER grace.
                    state.goal = None
                    state.center = None
                    state.label = ""

            elapsed = time.monotonic() - started
            if elapsed < period:
                stop_event.wait(period - elapsed)

    @staticmethod
    def _direction(det, K, D, head_pose, T_head_cam) -> tuple[float, float, float]:
        """Unit world direction to a detection: where the head would face it.

        The look-at pose's forward axis (its rotation's first column) is that
        direction; carried as a full vector, two targets at the same yaw but
        different height stay distinct.
        """
        u, v = det.center
        pose = look_at_image_pose(u, v, K, D, head_pose, T_head_cam)
        fwd = pose[:3, 0]
        return (float(fwd[0]), float(fwd[1]), float(fwd[2]))

    @staticmethod
    def _check_vocabulary(detector: Detector) -> None:
        """Warn if the server's model cannot see what we are looking for.

        The server silently detects *everything* when it recognises none of the
        requested labels, so an unknown one would look like a wildly distracted
        robot rather than a configuration mistake.
        """
        try:
            classes = detector.classes()
        except DetectorUnavailable as e:
            logger.warning("Could not fetch class list: %s", e)
            return
        where = getattr(detector, "url", "detector")
        missing = [label for label in TRACK_LABELS if label not in classes]
        if missing:
            logger.warning("%s does not detect %s", where, ", ".join(missing))
        else:
            logger.info("%s detects all of %s", where, ", ".join(TRACK_LABELS))

    @staticmethod
    def _scan_pose(t: float, phase: float = 0.0) -> Rotation:
        """A slow, level look from side to side, to find a target again.

        `t` is seconds since this scan began, not since the app started: the
        phase is chosen per scan so it picks up from the head's current yaw.
        The head is held level; a pitch left over from tracking is eased out
        once by the follower rather than swept up and down.
        """
        yaw = SCAN_DEGREES * math.sin(2 * math.pi * SCAN_HZ * t + phase)
        return Rotation.from_euler("Z", yaw, degrees=True)

    @staticmethod
    def _scan_phase(rotation: Rotation, rate: float = 0.0) -> float:
        """The scan phase whose starting yaw matches `rotation`, heading on.

        `rate` is the head's current yaw velocity. The scan carries on the way
        the head is already turning, so losing a target that was moving keeps
        the head chasing it rather than reversing. Once the head has settled
        (`rate` ~ 0, e.g. after holding a lost aim), it presses on outward from
        centre instead — never straight back through ground already covered.
        """
        yaw, _, _ = rotation.as_euler("ZYX", degrees=True)
        heading = rate if abs(rate) > 1e-3 else yaw
        return Tracker._phase_at(yaw, SCAN_DEGREES, heading)

    @staticmethod
    def _phase_at(angle: float, amplitude: float, heading: float = 1.0) -> float:
        """Phase of a sine of `amplitude` at `angle`, on the branch `heading` picks.

        A given `angle` sits at two phases, one rising and one falling; `heading`
        chooses which, so the scan can leave that point in either direction.
        """
        if amplitude <= 0.0:  # an axis turned off must not become a NaN pose
            return 0.0
        base = math.asin(max(-1.0, min(1.0, angle / amplitude)))
        # asin is the rising branch (yaw increasing); pi - asin is the falling
        # one. Pick whichever leaves the current yaw going the way we want.
        return base if heading >= 0 else math.pi - base

    @staticmethod
    def _antennas(perk: float, t: float) -> np.ndarray:
        """Wagging while searching, perked up when locked, blended in between."""
        wag = 12.0 * math.sin(2 * math.pi * 0.3 * t)
        angle = wag + perk * (20.0 - wag)
        return np.deg2rad([angle, -angle])

    def _mount_api(self, state: State) -> None:
        """Expose the control panel's read and write endpoints."""
        if self.settings_app is None:
            return

        @self.settings_app.get("/state")
        def get_state() -> dict:
            return state.snapshot()

        @self.settings_app.post("/config")
        def set_config(config: Config) -> dict:
            with state.lock:
                for field, value in config.model_dump(exclude_none=True).items():
                    setattr(state, field, value)
                if config.enabled is False:
                    state.goal = None
                    state.center = None
                    state.targets = []
            return state.snapshot()


if __name__ == "__main__":
    app = Tracker()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
