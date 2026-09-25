"""Short-term memory of what the robot has seen, and in which direction.

The camera gives bearings, not range, so each object's location is a unit
direction in the world frame (+X forward, +Y left, +Z up), the same frame the
head pose is expressed in. Directions are absolute, so an object stays put in
memory while the head turns away from it.
"""

from __future__ import annotations

import base64
import io
import itertools
import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps

from tracker.detector import Detection

# A unit direction in the world frame.
Vec3 = tuple[float, float, float]

# Thumbnail size in pixels: twice what the panel shows, so it stays crisp on a
# high-density screen.
THUMB_SIZE = (48, 36)


@dataclass
class WorldObject:
    """One object the robot believes is out there."""

    id: int
    label: str
    direction: Vec3
    seen_at: float  # monotonic time of the last sighting
    dwelt_at: float | None = None  # last time the head was looking its way
    focus_since: float | None = None  # start of the current stretch in focus
    thumb: str | None = None  # JPEG data URI, taken while it was the target


class WorldModel:
    """Objects seen recently, matched across sightings by label and direction.

    A sighting within `match_angle` of a remembered object of the same label is
    taken to be that object, which moves to the new direction; anything else is
    a new object. Objects not seen for `forget_after` seconds are dropped.

    It also follows where the head is looking. Every object within
    `focus_angle` of the aim is in focus: it counts as seen and watched, even on
    frames the detector misses it, and each keeps the start of its current
    stretch in focus. That stretch is what boredom is measured against, so an
    intermittent target still wears out its welcome, and a moving one does too:
    the stretch follows the object, not the direction.
    """

    def __init__(
        self,
        match_angle: float = math.radians(15.0),
        forget_after: float = 60.0,
        focus_angle: float = math.radians(20.0),
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        """Start with an empty world, looking nowhere in particular."""
        self._match_angle = match_angle
        self._forget_after = forget_after
        self._focus_angle = focus_angle
        self._time = time_fn
        self._ids = itertools.count(1)
        self._objects: list[WorldObject] = []
        self._aim: Vec3 | None = None

    def observe(self, dets: list[Detection], directions: list[Vec3]) -> list[int]:
        """Record one frame's detections, returning the object id of each.

        `directions` is each detection's world direction, parallel to `dets`.
        """
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

        ids = []
        for i, (det, vec) in enumerate(zip(dets, directions)):
            if i in matched:
                obj = self._objects[matched[i]]
                obj.direction, obj.seen_at = vec, now
            else:
                obj = WorldObject(next(self._ids), det.label, vec, now)
                self._objects.append(obj)
            ids.append(obj.id)
        return ids

    def look(self, aim: Vec3 | None) -> None:
        """Note where the head is aimed now, or None while it is scanning."""
        now = self._time()
        self._aim = aim
        for obj in self._objects:
            if self._in_focus(obj):
                obj.seen_at = obj.dwelt_at = now
                if obj.focus_since is None:
                    obj.focus_since = now
            else:
                obj.focus_since = None

    @property
    def focused_since(self) -> float | None:
        """Start of the longest current stretch in focus, if anything is in focus.

        The longest, because it is the first to run out: once one object there
        has been watched long enough, the head is bored of all of them.
        """
        starts = [o.focus_since for o in self._objects if o.focus_since is not None]
        return min(starts, default=None)

    def photograph(self, id: int, thumb: str) -> None:
        """Give object `id` a new picture.

        Only the target's is taken: the head is aimed at it then, so it is
        centred and steady rather than a blurred box at the edge of a scan.
        """
        self._by_id(id).thumb = thumb

    def dwelt_within(self, id: int, seconds: float) -> bool:
        """Whether the head was looking at object `id` in the last `seconds`."""
        dwelt_at = self._by_id(id).dwelt_at
        return dwelt_at is not None and self._time() - dwelt_at <= seconds

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
                "dwelt_ago": (
                    None if o.dwelt_at is None else round(now - o.dwelt_at, 1)
                ),
                "thumb": o.thumb,
                "focused": self._in_focus(o),
                "focused_for": (
                    None if o.focus_since is None else round(now - o.focus_since, 1)
                ),
            }
            for o in self.objects()
            for yaw, pitch in [yaw_pitch(o.direction)]
        ]

    def _in_focus(self, obj: WorldObject) -> bool:
        return (
            self._aim is not None
            and angle_between(obj.direction, self._aim) <= self._focus_angle
        )

    def _by_id(self, id: int) -> WorldObject:
        return next(o for o in self._objects if o.id == id)

    def _forget(self, now: float) -> None:
        self._objects = [
            o for o in self._objects if now - o.seen_at <= self._forget_after
        ]


def thumbnail(frame: npt.NDArray[np.uint8], det: Detection) -> str:
    """A small JPEG of `det`'s box in a BGR `frame`, as a data URI.

    Cropped to fill `THUMB_SIZE` exactly, so every row of the panel's table gets
    the same footprint whatever the box's shape.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = det.box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, max(x1 + 1, int(x2))), min(height, max(y1 + 1, int(y2)))

    # Stride down to about twice the target first: resampling a whole person
    # box to 48 px costs far more on the Pi than skipping pixels does.
    tw, th = THUMB_SIZE
    step = max(1, min((x2 - x1) // (2 * tw), (y2 - y1) // (2 * th)))
    crop = np.ascontiguousarray(frame[y1:y2:step, x1:x2:step, ::-1])  # BGR to RGB
    img = ImageOps.fit(Image.fromarray(crop), THUMB_SIZE, Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def angle_between(a: Vec3, b: Vec3) -> float:
    """Angle in radians between two unit direction vectors."""
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    return math.acos(max(-1.0, min(1.0, dot)))


def yaw_pitch(direction: Vec3) -> tuple[float, float]:
    """A world direction as (yaw, pitch) degrees: +yaw is left, +pitch is up."""
    x, y, z = direction
    return math.degrees(math.atan2(y, x)), math.degrees(
        math.asin(max(-1.0, min(1.0, z)))
    )
