"""The look-around cycle: random pose, hold, scan, track, and boredom.

Each cycle starts by moving the head to a random pose and holding it, so a
recording of the search that follows begins from somewhere new. The head then
scans until it finds something, follows it for as long as it moves, and tires
of it once it has stayed put for long enough, which starts the next cycle.
"""

from __future__ import annotations

import math
import random
from enum import Enum
from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

Pose = tuple[Rotation, npt.NDArray[np.float64]]  # orientation, position in metres


class Phase(str, Enum):
    """Where the head is in the cycle; published as-is in /state."""

    IDLE = "idle"  # tracking is off; the head rests at neutral
    MOVING = "moving"  # heading for this cycle's random pose
    HOLDING = "holding"  # still at the random pose, briefly
    SCANNING = "scanning"  # sweeping for something to look at
    TRACKING = "tracking"  # following a target, held through short losses


def random_pose(rng: random.Random, yaw: float, tilt: float, shift: float) -> Pose:
    """Uniform within +/-`yaw` and +/-`tilt` degrees, and +/-`shift` m per axis."""
    y, p, r = (rng.uniform(-a, a) for a in (yaw, tilt, tilt))
    position = np.array([rng.uniform(-shift, shift) for _ in range(3)])
    return Rotation.from_euler("ZYX", [y, p, r], degrees=True), position


class Cycle:
    """Phase transitions, fed the facts the control loop observes each tick.

    Time is passed in rather than read, so the machine can be driven through a
    whole cycle in a test without waiting.
    """

    def __init__(
        self,
        pose_fn: Callable[[], Pose],
        hold: float,
        move_timeout: float,
        idle_reset: float,
    ) -> None:
        """Hold each random pose `hold` s; scan `idle_reset` s before a new one.

        `move_timeout` ends a move that never settles, e.g. a pose the
        kinematics can't quite reach, so the cycle can't stall there.
        """
        self.pose_fn = pose_fn
        self.hold = hold
        self.move_timeout = move_timeout
        self.idle_reset = idle_reset
        self.phase = Phase.IDLE
        self.since = 0.0
        self.count = 0  # cycles started; a new random pose each time
        self.pose: Pose = (Rotation.identity(), np.zeros(3))

    def update(
        self, now: float, *, enabled: bool, arrived: bool, aimed: bool, bored: bool
    ) -> bool:
        """Advance on this tick's facts; True if the phase changed."""
        nxt = self._next(now, enabled, arrived, aimed, bored)
        if nxt is self.phase:
            return False
        self.phase, self.since = nxt, now
        if nxt is Phase.MOVING:
            self.count += 1
            self.pose = self.pose_fn()
        elif nxt is Phase.IDLE:
            self.pose = (Rotation.identity(), np.zeros(3))
        return True

    def _next(
        self, now: float, enabled: bool, arrived: bool, aimed: bool, bored: bool
    ) -> Phase:
        elapsed = now - self.since
        match self.phase:
            case _ if not enabled:
                return Phase.IDLE
            case Phase.IDLE:
                return Phase.MOVING
            case Phase.MOVING if arrived or elapsed >= self.move_timeout:
                return Phase.HOLDING
            case Phase.HOLDING if elapsed >= self.hold:
                return Phase.SCANNING
            case Phase.SCANNING if aimed:
                return Phase.TRACKING
            case Phase.SCANNING if elapsed >= self.idle_reset:
                return Phase.MOVING
            case Phase.TRACKING if bored:
                return Phase.MOVING
            case Phase.TRACKING if not aimed:
                return Phase.SCANNING
        return self.phase


class Stillness:
    """How long a target has stayed within `degrees` of where it settled.

    Directions are in the world frame, so the head's own turning doesn't count
    as the target moving. Any move past the threshold re-anchors the clock: a
    target that keeps moving never gets old.
    """

    def __init__(self, degrees: float) -> None:
        """Count a target as still while it stays inside `degrees`."""
        self.limit = math.radians(degrees)
        self.anchor: npt.NDArray[np.float64] | None = None
        self.since: float | None = None

    def update(self, direction: npt.NDArray[np.float64], now: float) -> None:
        """Note the target's unit world direction at `now`."""
        if self.anchor is None or angle_between(self.anchor, direction) > self.limit:
            self.anchor, self.since = direction, now

    def still_for(self, now: float) -> float | None:
        """Seconds the target has been still, or None with no target."""
        return None if self.since is None else now - self.since

    def reset(self) -> None:
        """Forget the target."""
        self.anchor = self.since = None


def angle_between(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Angle in radians between two unit vectors."""
    return math.acos(max(-1.0, min(1.0, float(np.dot(a, b)))))
