"""Target selection and motion smoothing.

Adapted from ``reachy_mini.vision.face_tracking``, which solves the same problem
for faces: pick one detection out of many, stay locked onto it across frames, and
smooth its center before it reaches the control loop.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from tracker.detector import Detection


def norm_center(det: Detection, width: int, height: int) -> tuple[float, float]:
    """Box center normalized to [-1, 1] on both axes."""
    u, v = det.center
    return (u / max(width - 1, 1) * 2 - 1, v / max(height - 1, 1) * 2 - 1)


def pixel_center(
    center: tuple[float, float], width: int, height: int
) -> tuple[float, float]:
    """Inverse of `norm_center`."""
    x, y = center
    return ((x + 1) / 2 * max(width - 1, 1), (y + 1) / 2 * max(height - 1, 1))


class PoseSmoother:
    """Critically damped second-order follower for an orientation.

    A first-order lag reaches a stepped setpoint with a velocity discontinuity
    at every step, and the goal here steps with every detection — which is what
    reads as jerk. Carrying angular velocity as state makes velocity continuous,
    and critical damping still gets there without overshooting.
    """

    def __init__(
        self, tau: float, max_speed: float = 4.0, max_pull: float = 20.0
    ) -> None:
        """Follow with time constant `tau`, bounded by `max_speed` and `max_pull`."""
        self.omega_n = 1.0 / tau
        self.max_speed = max_speed
        self.max_pull = max_pull
        self.rotation = Rotation.identity()
        self.omega = np.zeros(3)

    def step(self, goal: Rotation, dt: float) -> Rotation:
        """Advance toward `goal` by `dt` seconds, returning the new orientation."""
        # Explicit integration diverges once dt is large next to the time
        # constant, so take several small steps rather than one long one.
        steps = max(1, int(np.ceil(dt * self.omega_n / 0.25)))
        h = dt / steps

        for _ in range(steps):
            err = (goal * self.rotation.inv()).as_rotvec()

            # Capping the pull bounds how much velocity can change in one tick,
            # which is what stops a big goal step reading as a lurch. Damping is
            # deliberately left uncapped, so it always retains the authority to
            # stop the head and critical damping still means no overshoot.
            pull = self.omega_n**2 * err
            magnitude = float(np.linalg.norm(pull))
            if magnitude > self.max_pull:
                pull *= self.max_pull / magnitude

            accel = pull - 2.0 * self.omega_n * self.omega
            self.omega = self.omega + accel * h

            speed = float(np.linalg.norm(self.omega))
            if speed > self.max_speed:
                self.omega *= self.max_speed / speed

            self.rotation = Rotation.from_rotvec(self.omega * h) * self.rotation

        return self.rotation

    @property
    def speed(self) -> float:
        """Current angular speed in rad/s."""
        return float(np.linalg.norm(self.omega))

    def reset(self) -> None:
        """Return to the neutral pose, at rest."""
        self.rotation = Rotation.identity()
        self.omega = np.zeros(3)


def pose_matrix(rot: Rotation) -> npt.NDArray[np.float64]:
    """Wrap a rotation as the 4x4 head pose the SDK expects."""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot.as_matrix()
    return pose


class TargetSelector:
    """Lock onto one detection: acquire the best, then follow it across frames.

    The max-jump gate is what stops the head snapping between two cats; the miss
    counter is what lets it let go once the real one has left the room.

    `priority` ranks the classes: earlier is preferred, and a class we would
    rather watch takes the lock off one we are already watching. It has to be
    seen for `upgrade_after` frames running to do that, so a detection
    flickering at the confidence threshold cannot bounce the head between two
    subjects.

    `max_lock` is the boredom timer: hold one target that long and the lock is
    dropped so the scan can look for others. For `look_away` seconds after that,
    a detection near where the abandoned one sat is passed over, so the head
    turns to something else rather than snapping straight back.
    """

    def __init__(
        self,
        priority: list[str] | None = None,
        min_area_frac: float = 0.002,
        max_jump: float = 0.5,
        max_misses: int = 12,
        upgrade_after: int = 3,
        max_lock: float | None = None,
        look_away: float = 4.0,
        avoid_radius: float = 0.35,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a selector with the given preference and association gates."""
        self._priority = list(priority or [])
        self._min_area_frac = min_area_frac
        self._max_jump = max_jump
        self._max_misses = max_misses
        self._upgrade_after = upgrade_after
        self._max_lock = max_lock
        self._look_away = look_away
        self._avoid_radius = avoid_radius
        self._time = time_fn
        self._center: tuple[float, float] | None = None
        self._label: str | None = None
        self._misses = 0
        self._better = 0
        self._locked_at: float | None = None
        self._avoid: tuple[float, float] | None = None
        self._avoid_until = 0.0

    def select(
        self, dets: list[Detection], width: int, height: int
    ) -> Detection | None:
        """Pick the detection to aim at, or None when none is plausible."""
        now = self._time()

        # Bored of a target held too long: drop it and steer clear of it for a
        # moment, so the scan turns up something else instead of re-grabbing it.
        if (
            self._max_lock is not None
            and self._locked_at is not None
            and now - self._locked_at >= self._max_lock
        ):
            self._avoid, self._avoid_until = self._center, now + self._look_away
            self._center = self._label = self._locked_at = None

        if not dets:
            self._miss()
            return None

        if self._avoid is not None and now < self._avoid_until:
            dets = [d for d in dets if self._far_from_avoided(d, width, height)]
            if not dets:
                self._miss()
                return None

        best = min(self._rank(det.label) for det in dets)
        self._better = self._better + 1 if best < self._rank(self._label) else 0
        wanted = [det for det in dets if self._rank(det.label) == best]

        acquiring = self._center is None or self._better >= self._upgrade_after
        if acquiring:
            det = max(wanted, key=lambda d: d.area)
            if det.area < self._min_area_frac * width * height:
                self._miss()
                return None
        else:
            anchor = self._center
            det = min(
                wanted, key=lambda d: _dist2(norm_center(d, width, height), anchor)
            )
            if _dist2(norm_center(det, width, height), anchor) > self._max_jump**2:
                self._miss()
                return None

        if acquiring:  # a fresh lock restarts the boredom timer and ends look-away
            self._locked_at = now
            self._avoid = None
        self._center = norm_center(det, width, height)
        self._label = det.label
        self._misses = 0
        self._better = 0
        return det

    def _far_from_avoided(self, det: Detection, width: int, height: int) -> bool:
        assert self._avoid is not None
        return (
            _dist2(norm_center(det, width, height), self._avoid) > self._avoid_radius**2
        )

    def _rank(self, label: str | None) -> int:
        """Lower is preferred; anything unlisted comes last."""
        if label in self._priority:
            return self._priority.index(label)
        return len(self._priority)

    def _miss(self) -> None:
        self._misses += 1
        if self._misses > self._max_misses:
            self._center = None
            self._label = None

    @property
    def has_target(self) -> bool:
        """Whether the selector is still associated with a target."""
        return self._center is not None

    @property
    def looking_away(self) -> bool:
        """Whether we just dropped a target out of boredom and are avoiding it."""
        return self._avoid is not None and self._time() < self._avoid_until

    @property
    def label(self) -> str | None:
        """Class of the current target, if there is one."""
        return self._label

    def reset(self) -> None:
        """Forget the current lock, e.g. after the tracked class changes."""
        self._center = None
        self._label = None
        self._misses = 0
        self._better = 0
        self._locked_at = None
        self._avoid = None


class CenterFilter:
    """Adaptive EMA on the target center: gentle when still, quick when it bolts."""

    _ALPHA = 0.3
    _FAST_ALPHA = 0.6
    _MOVEMENT_THRESHOLD = 0.15
    _DEAD_ZONE = 0.02

    def __init__(self) -> None:
        """Create an unseeded filter; the first sample passes through."""
        self._value: npt.NDArray[np.float64] | None = None
        self._previous: npt.NDArray[np.float64] | None = None

    def update(self, center: tuple[float, float]) -> tuple[float, float]:
        """Consume one raw center and return the filtered one."""
        current = np.asarray(center, dtype=np.float64)
        if self._value is None or self._previous is None:
            self._value = current.copy()
            self._previous = current.copy()
            return center

        movement = float(np.linalg.norm(current - self._previous))
        self._previous = current.copy()
        delta = current - self._value
        # Below the dead zone this is detector noise, not the target moving.
        if float(np.linalg.norm(delta)) < self._DEAD_ZONE:
            return (float(self._value[0]), float(self._value[1]))

        alpha = self._FAST_ALPHA if movement > self._MOVEMENT_THRESHOLD else self._ALPHA
        self._value += alpha * delta
        return (float(self._value[0]), float(self._value[1]))

    def reset(self) -> None:
        """Forget history so the next sample is accepted immediately."""
        self._value = None
        self._previous = None


def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
