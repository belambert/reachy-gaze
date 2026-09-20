"""Target selection and motion smoothing.

Adapted from ``reachy_mini.vision.face_tracking``, which solves the same problem
for faces: pick one detection out of many, stay locked onto it across frames, and
smooth its center before it reaches the control loop.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from gaze_tracker.detector import Detection


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


def slew(current: Rotation, goal: Rotation, alpha: float) -> Rotation:
    """Rotate `current` a fraction `alpha` of the way toward `goal`."""
    return Rotation.from_rotvec((goal * current.inv()).as_rotvec() * alpha) * current


def pose_matrix(rot: Rotation) -> npt.NDArray[np.float64]:
    """Wrap a rotation as the 4x4 head pose the SDK expects."""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot.as_matrix()
    return pose


class TargetSelector:
    """Lock onto one detection: acquire the largest, then follow the nearest.

    The max-jump gate is what stops the head snapping between two cats; the miss
    counter is what lets it let go once the real one has left the room.
    """

    def __init__(
        self,
        min_area_frac: float = 0.002,
        max_jump: float = 0.5,
        max_misses: int = 12,
    ) -> None:
        """Create a selector with the given acquisition and association gates."""
        self._min_area_frac = min_area_frac
        self._max_jump = max_jump
        self._max_misses = max_misses
        self._center: tuple[float, float] | None = None
        self._misses = 0

    def select(
        self, dets: list[Detection], width: int, height: int
    ) -> Detection | None:
        """Pick the detection to aim at, or None when none is plausible."""
        if not dets:
            self._miss()
            return None

        if self._center is None:
            det = max(dets, key=lambda d: d.area)
            if det.area < self._min_area_frac * width * height:
                self._miss()
                return None
        else:
            anchor = self._center
            det = min(dets, key=lambda d: _dist2(norm_center(d, width, height), anchor))
            if _dist2(norm_center(det, width, height), anchor) > self._max_jump**2:
                self._miss()
                return None

        self._center = norm_center(det, width, height)
        self._misses = 0
        return det

    def _miss(self) -> None:
        self._misses += 1
        if self._misses > self._max_misses:
            self._center = None

    @property
    def has_target(self) -> bool:
        """Whether the selector is still associated with a target."""
        return self._center is not None

    def reset(self) -> None:
        """Forget the current lock, e.g. after the tracked class changes."""
        self._center = None
        self._misses = 0


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
