"""Track COCO objects with Reachy Mini's head, in a look-around cycle.

Each cycle moves the head to a random pose, holds it, scans until something
turns up, and follows that until it has sat still long enough to get boring
(see ``reachy_gaze.cycle``). Detection runs off-board (see ``reachy_gaze.server``) because the Pi's CPU is
already busy with motor control. Two rates keep that from showing: the vision
thread re-anchors the target a handful of times a second, while the control loop
slews the head toward that anchor at 50 Hz.
"""

from __future__ import annotations

import logging
import math
import os
import random
import threading
import time

import numpy as np
from pydantic import BaseModel, Field, field_validator
from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.vision.look_at import look_at_image_pose
from scipy.spatial.transform import Rotation

from reachy_gaze.cycle import Cycle, Phase, Stillness, random_pose
from reachy_gaze.detector import BACKENDS, Detector, DetectorUnavailable, make_detector
from reachy_gaze.tracking import (
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
MAX_HEAD_PULL = 5.0  # rad/s^2 ceiling on the follower's pull; lower is gentler
BLEND_TAU = 0.4  # seconds to ease between searching and locked-on posture
# The body's yaw joint allows +/-160 degrees; this keeps the scan to the front
# of the robot, well inside that.
SCAN_DEGREES = 90.0  # half-width of the yaw scan
SCAN_HZ = 0.04  # yaw scan rate; peak speed is 2*pi*SCAN_HZ*SCAN_DEGREES
LOST_AFTER = 10.0  # seconds holding the last aim point before giving up
STALE_AFTER = 5.0  # seconds before the panel calls the lock stale rather than live
RETRY_AFTER = 2.0  # seconds to wait out an unreachable detection server

# The look-around cycle.
BORED_AFTER = 20.0  # default seconds a still target holds attention; a slider
STILL_DEGREES = 8.0  # a target within this of where it settled counts as still
HOLD_SECONDS = 1.5  # pause at each random pose before scanning
MOVE_TIMEOUT = 5.0  # stop waiting for the head to settle at a random pose
IDLE_RESET = 30.0  # seconds of fruitless scanning before a new random pose
ARRIVE_DEGREES = 2.0  # a random pose is reached once the head is this close...
ARRIVE_SPEED = 0.1  # ...and turning slower than this, in rad/s
POSITION_TAU = 0.3  # seconds for the head's position to ease to a new one
# Random poses keep well inside the head's reach, so the kinematics never has
# to refuse one; the body turns to help with yaw.
RANDOM_YAW = 90.0  # degrees either side of straight ahead
RANDOM_TILT = 15.0  # degrees of pitch and of roll, either way
RANDOM_SHIFT = 0.01  # metres along each axis, either way

# Phases in which the head takes a lock on what it sees.
LOOKING = (Phase.SCANNING, Phase.TRACKING)

# Which detection backend to use, and where to reach it. Both are prefilled in
# the control panel and overridable from the environment without editing code.
DEFAULT_BACKEND = os.environ.get("REACHY_GAZE_BACKEND", "builtin")
DEFAULT_SERVER_URL = os.environ.get(
    "REACHY_GAZE_SERVER_URL", BACKENDS[DEFAULT_BACKEND].default_url
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
    x, y, z = head_pose[:3, 0]
    return math.degrees(math.atan2(y, x)), math.degrees(
        math.asin(max(-1.0, min(1.0, z)))
    )


class Config(BaseModel):
    """Settings the control panel can change while the app runs."""

    enabled: bool | None = None
    conf: float | None = None
    server_url: str | None = None
    backend: str | None = None
    scan: bool | None = None
    # A pull of zero would freeze the head, so refuse it rather than obey it.
    pull: float | None = Field(None, gt=0.0, le=200.0)
    bored_after: float | None = Field(None, gt=0.0, le=600.0)

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
        self.bored_after = BORED_AFTER

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
        # Camera intrinsics and frame size, so the panel can draw rings of equal
        # angle off the camera axis; None until the first frame arrives.
        self.lens: dict | None = None

        # Where the head is in the look-around cycle, mirrored from the control
        # loop's Cycle; `cycle` counts random poses, one per cycle.
        self.phase = Phase.IDLE
        self.phase_since = time.monotonic()
        self.cycle = 0
        # How long the target has stayed put, which is what boredom times.
        self.stillness = Stillness(STILL_DEGREES)

    def snapshot(self) -> dict:
        """Everything the control panel polls, in one consistent read."""
        with self.lock:
            now = time.monotonic()
            seen_ago = now - self.last_seen if self.goal is not None else None
            locked = seen_ago is not None and seen_ago < LOST_AFTER
            yaw, pitch = look_yaw_pitch(self.head_pose)
            tracking = self.phase is Phase.TRACKING
            still_for = self.stillness.still_for(now) if tracking else None
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
                "lens": self.lens,
                "seen_ago": round(seen_ago, 1) if locked else None,
                "phase": self.phase.value,
                "phase_for": round(now - self.phase_since, 1),
                "cycle": self.cycle,
                "bored_after": self.bored_after,
                "still_for": None if still_for is None else round(still_for, 1),
                "bored_in": (
                    None
                    if still_for is None
                    else round(max(0.0, self.bored_after - still_for), 1)
                ),
            }


class ReachyGaze(ReachyMiniApp):
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
            name="gaze-vision",
        )
        vision.start()
        try:
            self._drive(reachy_mini, state, stop_event)
        finally:
            vision.join(timeout=2.0)

    def _drive(
        self, mini: ReachyMini, state: State, stop_event: threading.Event
    ) -> None:
        """Run the look-around cycle, keeping commanded velocity continuous."""
        logger.info("Control loop running at %.0f Hz", CONTROL_HZ)
        period = 1.0 / CONTROL_HZ
        smoother = PoseSmoother(SMOOTH_TAU, MAX_HEAD_SPEED, MAX_HEAD_PULL)
        rng = random.Random()
        cycle = Cycle(
            lambda: random_pose(rng, RANDOM_YAW, RANDOM_TILT, RANDOM_SHIFT),
            HOLD_SECONDS,
            MOVE_TIMEOUT,
            IDLE_RESET,
        )
        position = np.zeros(3)
        perk = 0.0
        was_scanning = False
        scan_t0 = 0.0
        scan_phase = 0.0
        scan_from = Rotation.identity()
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
                # Not "seen just now": the aim is held for the whole of
                # LOST_AFTER, so this stays true long after the last sighting.
                aimed = state.goal is not None and now - state.last_seen < LOST_AFTER
                still_for = state.stillness.still_for(now)
                changed = cycle.update(
                    now,
                    enabled=state.enabled,
                    arrived=self._arrived(smoother, cycle.pose[0]),
                    aimed=aimed,
                    bored=still_for is not None and still_for >= state.bored_after,
                )
                if changed:
                    self._enter_phase(state, cycle, now)
                goal, scan_on = state.goal, state.scan
                smoother.max_pull = state.pull  # tunable live from the panel
            phase = cycle.phase

            # 0 searching, 1 locked on, eased so the antennas never snap.
            locked = float(phase is Phase.TRACKING)
            perk += (locked - perk) * (1.0 - math.exp(-dt / BLEND_TAU))

            # Start each scan from wherever the head already is, carrying on the
            # way it was turning. Running the sine off a fixed epoch meant it
            # began at an arbitrary phase, so losing a target swung the head to
            # centre and then out again; ignoring the direction meant it always
            # set off the same way and left one side of the room unswept.
            scanning = phase is Phase.SCANNING
            if scanning and not was_scanning:  # a scan starts on this tick
                scan_t0, scan_from = now, smoother.rotation
                scan_phase = self._scan_phase(smoother.rotation, smoother.omega[2])
            was_scanning = scanning

            if phase is Phase.TRACKING and goal is not None:
                target = goal
            elif scanning:
                # with the sweep off, wait where the scan would have begun
                target = (
                    self._scan_pose(now - scan_t0, scan_phase) if scan_on else scan_from
                )
            else:  # moving to, or holding, the random pose; neutral when idle
                target = cycle.pose[0]

            # The position has no velocity to keep continuous, and moves only a
            # centimetre or two, so a plain exponential ease is enough.
            position += (cycle.pose[1] - position) * (
                1.0 - math.exp(-dt / POSITION_TAU)
            )
            mini.set_target(
                head=pose_matrix(smoother.step(target, dt), position),
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

    @staticmethod
    def _enter_phase(state: State, cycle: Cycle, now: float) -> None:
        """Publish a new phase, dropping what no longer applies; caller locks."""
        state.phase, state.phase_since, state.cycle = cycle.phase, now, cycle.count
        if cycle.phase not in LOOKING:
            # A fresh cycle must not re-lock onto the target it just tired of.
            state.goal = state.center = None
            state.label = ""
        if cycle.phase is not Phase.TRACKING:
            state.stillness.reset()

    @staticmethod
    def _arrived(smoother: PoseSmoother, goal: Rotation) -> bool:
        """Whether the commanded head has settled on `goal`.

        This is the command, not the measured head, which trails it a little;
        the hold that follows covers the difference.
        """
        off = (goal * smoother.rotation.inv()).magnitude()
        return off < math.radians(ARRIVE_DEGREES) and smoother.speed < ARRIVE_SPEED

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

        selector = TargetSelector(TRACK_LABELS)
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
                looking = state.phase in LOOKING

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
            # Moving to and holding a random pose are meant to be still, so
            # nothing seen then takes the lock; the scan that follows will.
            det = selector.select(dets, width, height) if looking else None
            if not looking:
                selector.reset()
            if det is None and not selector.has_target:
                smoother.reset()

            with state.lock:
                state.lens = {
                    "fx": float(K[0, 0]),
                    "fy": float(K[1, 1]),
                    "cx": float(K[0, 2]),
                    "cy": float(K[1, 2]),
                    "width": width,
                    "height": height,
                }
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
                # Re-checked here: the phase may have moved on during detection,
                # and a fresh cycle must not inherit this lock.
                if det is not None and state.phase in LOOKING:
                    center = smoother.update(norm_center(det, width, height))
                    u, v = pixel_center(center, width, height)
                    # Aim against the pose the frame was captured at, so a late
                    # detection still points where the target actually was.
                    state.goal = Rotation.from_matrix(
                        look_at_image_pose(u, v, K, D, head_pose, T_head_cam)[:3, :3]
                    )
                    state.last_seen = time.monotonic()
                    state.stillness.update(
                        state.goal.as_matrix()[:, 0], state.last_seen
                    )
                    state.center = center
                    state.label = det.label

            elapsed = time.monotonic() - started
            if elapsed < period:
                stop_event.wait(period - elapsed)

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
        return ReachyGaze._phase_at(yaw, SCAN_DEGREES, heading)

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
    app = ReachyGaze()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
