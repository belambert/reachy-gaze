"""The look-around cycle: scan, track, and boredom.

The head scans until it finds something, follows it for as long as it moves,
and tires of it once it has stayed put for long enough. A target it tired of is
then ignored for a while, so the scan that follows looks for something else
rather than coming straight back to it.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
import numpy.typing as npt


class Phase(str, Enum):
    """Where the head is in the cycle; published as-is in /state."""

    IDLE = "idle"  # tracking is off; the head rests at neutral
    SCANNING = "scanning"  # sweeping for something to look at
    TRACKING = "tracking"  # following a target, held through short losses


class Cycle:
    """Phase transitions, fed the facts the control loop observes each tick.

    Time is passed in rather than read, so the machine can be driven through a
    whole cycle in a test without waiting.
    """

    def __init__(self) -> None:
        """Start idle."""
        self.phase = Phase.IDLE
        self.since = 0.0

    def update(self, now: float, *, enabled: bool, aimed: bool, bored: bool) -> bool:
        """Advance on this tick's facts; True if the phase changed."""
        nxt = self._next(enabled, aimed, bored)
        if nxt is self.phase:
            return False
        self.phase, self.since = nxt, now
        return True

    def _next(self, enabled: bool, aimed: bool, bored: bool) -> Phase:
        match self.phase:
            case _ if not enabled:
                return Phase.IDLE
            case Phase.IDLE:
                return Phase.SCANNING
            case Phase.SCANNING if aimed:
                return Phase.TRACKING
            case Phase.TRACKING if bored or not aimed:
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


class Ignored:
    """World directions to look past for a while: targets the head tired of.

    A bored target usually hasn't gone anywhere, so without this the next scan
    would find it again at once. Directions are in the world frame, so the
    head's own turning doesn't move them.
    """

    def __init__(self, degrees: float, seconds: float) -> None:
        """Ignore detections within `degrees` of a direction for `seconds`."""
        self.limit = math.radians(degrees)
        self.seconds = seconds
        self._until: list[tuple[npt.NDArray[np.float64], float, str]] = []

    def add(
        self, direction: npt.NDArray[np.float64], now: float, label: str = ""
    ) -> None:
        """Start ignoring the unit world `direction`, where a `label` sat, from `now`."""
        self._until.append((direction, now + self.seconds, label))

    def covers(self, direction: npt.NDArray[np.float64], now: float) -> bool:
        """Whether `direction` is currently ignored."""
        return any(
            angle_between(d, direction) <= self.limit for d, _, _ in self.active(now)
        )

    def active(self, now: float) -> list[tuple[npt.NDArray[np.float64], str, float]]:
        """Each ignored (direction, label, seconds left), dropping expired ones."""
        self._until = [entry for entry in self._until if entry[1] > now]
        return [(d, label, t - now) for d, t, label in self._until]

    def clear(self) -> None:
        """Ignore nothing."""
        self._until = []


def angle_between(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Angle in radians between two unit vectors."""
    return math.acos(max(-1.0, min(1.0, float(np.dot(a, b)))))
