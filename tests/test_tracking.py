import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tracker.detector import Detection
from tracker.tracking import (
    CenterFilter,
    TargetSelector,
    norm_center,
    pixel_center,
    pose_matrix,
    slew,
)

W, H = 640, 480


def box(cx, cy, size=100, label="cat", conf=0.9):
    half = size / 2
    return Detection(label, conf, (cx - half, cy - half, cx + half, cy + half))


class TestSelector:
    def test_acquires_largest(self):
        small, large = box(100, 100, size=40), box(500, 300, size=200)
        assert TargetSelector().select([small, large], W, H) is large

    def test_ignores_specks(self):
        # A 4px box is noise, not a cat.
        assert TargetSelector().select([box(320, 240, size=4)], W, H) is None

    def test_follows_nearest_once_locked(self):
        sel = TargetSelector()
        sel.select([box(100, 240, size=200)], W, H)

        # The bigger box is the wrong one; proximity to the lock wins.
        nearby, bigger = box(130, 240, size=100), box(600, 240, size=300)
        assert sel.select([nearby, bigger], W, H) is nearby

    def test_rejects_implausible_jump(self):
        sel = TargetSelector()
        sel.select([box(60, 240, size=150)], W, H)
        assert sel.select([box(620, 240, size=150)], W, H) is None

    def test_releases_lock_after_sustained_misses(self):
        sel = TargetSelector(max_misses=3)
        sel.select([box(320, 240, size=150)], W, H)
        assert sel.has_target

        for _ in range(4):
            sel.select([], W, H)
        assert not sel.has_target

        # Free again, it re-acquires anywhere in frame.
        assert sel.select([box(600, 100, size=150)], W, H) is not None

    def test_brief_dropout_keeps_lock(self):
        sel = TargetSelector(max_misses=5)
        sel.select([box(320, 240, size=150)], W, H)
        for _ in range(3):
            sel.select([], W, H)
        assert sel.has_target


class TestCenterFilter:
    def test_first_sample_passes_through(self):
        assert CenterFilter().update((0.3, -0.2)) == (0.3, -0.2)

    def test_suppresses_jitter(self):
        filt = CenterFilter()
        filt.update((0.0, 0.0))
        assert filt.update((0.005, 0.005)) == (0.0, 0.0)

    def test_converges_without_overshoot(self):
        filt = CenterFilter()
        filt.update((0.0, 0.0))
        seen = [filt.update((0.5, 0.0))[0] for _ in range(40)]

        assert seen[0] < 0.5, "must lag, not snap"
        assert max(seen) <= 0.5 + 1e-9, "must not overshoot"
        # The dead zone trades a standing offset for a head that doesn't dither.
        assert seen[-1] == pytest.approx(0.5, abs=CenterFilter._DEAD_ZONE)

    def test_reacts_faster_to_large_movement(self):
        slow, fast = CenterFilter(), CenterFilter()
        for filt in (slow, fast):
            filt.update((0.0, 0.0))

        slow.update((0.05, 0.0))  # below the movement threshold
        fast.update((0.9, 0.0))  # a bolt across frame
        assert fast.update((0.9, 0.0))[0] > slow.update((0.9, 0.0))[0]

    def test_reset_forgets_history(self):
        filt = CenterFilter()
        filt.update((0.5, 0.5))
        filt.reset()
        assert filt.update((-0.5, -0.5)) == (-0.5, -0.5)


class TestSlew:
    def test_zero_alpha_holds_and_one_arrives(self):
        current = Rotation.identity()
        goal = Rotation.from_euler("z", 30, degrees=True)

        assert slew(current, goal, 0.0).magnitude() == pytest.approx(0.0)
        assert slew(current, goal, 1.0).approx_equal(goal, atol=1e-9)

    def test_partial_step_moves_proportionally(self):
        goal = Rotation.from_euler("z", 40, degrees=True)
        stepped = slew(Rotation.identity(), goal, 0.25)
        assert np.degrees(stepped.magnitude()) == pytest.approx(10.0, abs=1e-6)

    def test_repeated_steps_converge(self):
        current = Rotation.identity()
        goal = Rotation.from_euler("zy", [35, -10], degrees=True)
        for _ in range(200):
            current = slew(current, goal, 0.1)
        assert current.approx_equal(goal, atol=1e-6)


def test_pose_matrix_is_a_pure_rotation():
    pose = pose_matrix(Rotation.from_euler("z", 25, degrees=True))
    assert pose.shape == (4, 4)
    assert pose[:3, 3] == pytest.approx([0, 0, 0])
    assert pose[3] == pytest.approx([0, 0, 0, 1])


@pytest.mark.parametrize("cx,cy", [(0, 0), (320, 240), (639, 479), (100, 400)])
def test_center_normalization_round_trips(cx, cy):
    u, v = pixel_center(norm_center(box(cx, cy), W, H), W, H)
    assert (u, v) == pytest.approx((cx, cy))


def test_frame_center_maps_to_origin():
    assert norm_center(box((W - 1) / 2, (H - 1) / 2), W, H) == pytest.approx((0.0, 0.0))


class TestClassRefresh:
    """The picker follows the server's vocabulary, not a hardcoded list."""

    class _Stub:
        url = "http://stub:8100"

        def __init__(self, result):
            self.result = result

        def classes(self):
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    def test_server_vocabulary_is_adopted(self):
        from tracker.main import State, Tracker

        state = State()
        Tracker._refresh_classes(self._Stub(["cat", "robot", "mug"]), state)
        assert state.classes == ["cat", "robot", "mug"]

    def test_unreachable_server_keeps_the_last_list(self):
        from tracker.detector import DetectorUnavailable
        from tracker.main import State, Tracker

        state = State()
        before = list(state.classes)
        Tracker._refresh_classes(self._Stub(DetectorUnavailable("down")), state)
        assert state.classes == before, "the picker must not empty itself"

    def test_a_later_failure_does_not_undo_a_good_fetch(self):
        from tracker.detector import DetectorUnavailable
        from tracker.main import State, Tracker

        state = State()
        Tracker._refresh_classes(self._Stub(["cat"]), state)
        Tracker._refresh_classes(self._Stub(DetectorUnavailable("down")), state)
        assert state.classes == ["cat"]
