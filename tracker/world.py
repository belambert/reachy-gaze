"""Short-term memory of what the robot has seen, and in which direction.

The camera gives bearings, not range, so each object's location is a unit
direction in the world frame (+X forward, +Y left, +Z up), the same frame the
head pose is expressed in. Directions are absolute, so an object stays put in
memory while the head turns away from it.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Callable

from tracker.detector import Detection
from tracker.tracking import Vec3, angle_between


@dataclass
class WorldObject:
    """One object the robot believes is out there."""

    id: int
    label: str
    direction: Vec3
    seen_at: float  # monotonic time of the last sighting


class WorldModel:
    """Objects seen recently, matched across sightings by label and direction.

    A sighting within `match_angle` of a remembered object of the same label is
    taken to be that object, which moves to the new direction; anything else is
    a new object. Objects not seen for `forget_after` seconds are dropped.
    """

    def __init__(
        self,
        match_angle: float = math.radians(15.0),
        forget_after: float = 60.0,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        """Start with an empty world."""
        self._match_angle = match_angle
        self._forget_after = forget_after
        self._time = time_fn
        self._ids = itertools.count(1)
        self._objects: list[WorldObject] = []

    def observe(self, dets: list[Detection], directions: list[Vec3]) -> None:
        """Record one frame's detections, parallel to their world `directions`."""
        now = self._time()
        self._forget(now)

        # Greedy, closest pairs first, so two cats side by side each keep
        # their own entry rather than both matching whichever was listed first.
        pairs = sorted(
            (angle_between(vec, obj.direction), i, j)
            for i, (det, vec) in enumerate(zip(dets, directions))
            for j, obj in enumerate(self._objects)
            if obj.label == det.label
        )
        matched: dict[int, int] = {}
        for gap, i, j in pairs:
            if gap > self._match_angle:
                break
            if i not in matched and j not in matched.values():
                matched[i] = j

        for i, (det, vec) in enumerate(zip(dets, directions)):
            if i in matched:
                obj = self._objects[matched[i]]
                obj.direction, obj.seen_at = vec, now
            else:
                self._objects.append(WorldObject(next(self._ids), det.label, vec, now))

    def objects(self) -> list[WorldObject]:
        """Everything still remembered, most recently seen first."""
        self._forget(self._time())
        return sorted(self._objects, key=lambda o: o.seen_at, reverse=True)

    def snapshot(self) -> list[dict]:
        """The world as the control panel reads it, ages in seconds."""
        now = self._time()
        return [
            {
                "id": o.id,
                "label": o.label,
                "direction": [round(c, 4) for c in o.direction],
                "yaw": round(yaw, 1),
                "pitch": round(pitch, 1),
                "age": round(now - o.seen_at, 1),
            }
            for o in self.objects()
            for yaw, pitch in [yaw_pitch(o.direction)]
        ]

    def _forget(self, now: float) -> None:
        self._objects = [
            o for o in self._objects if now - o.seen_at <= self._forget_after
        ]


def yaw_pitch(direction: Vec3) -> tuple[float, float]:
    """A world direction as (yaw, pitch) degrees: +yaw is left, +pitch is up."""
    x, y, z = direction
    return math.degrees(math.atan2(y, x)), math.degrees(
        math.asin(max(-1.0, min(1.0, z)))
    )
