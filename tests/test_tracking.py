import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tracker.detector import Detection
from tracker.tracking import (
    CenterFilter,
    PoseSmoother,
    TargetSelector,
    norm_center,
    pixel_center,
    pose_matrix,
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


class TestPoseSmoother:
    """Smoothness is the point: velocity must stay continuous under a stepping goal."""

    DT = 1 / 50

    def run(self, smoother, goal, ticks):
        """Step `ticks` times, returning the angle to goal and speed at each."""
        out = []
        for _ in range(ticks):
            smoother.step(goal, self.DT)
            err = float((goal * smoother.rotation.inv()).magnitude())
            out.append((err, smoother.speed))
        return out

    def test_converges_to_the_goal(self):
        smoother = PoseSmoother(tau=0.09)
        goal = Rotation.from_euler("zy", [30, -12], degrees=True)
        assert self.run(smoother, goal, 200)[-1][0] == pytest.approx(0.0, abs=1e-3)

    def test_starts_from_rest_rather_than_jumping(self):
        # The first-order version commanded err/tau immediately; this must not.
        smoother = PoseSmoother(tau=0.09)
        goal = Rotation.from_euler("z", 40, degrees=True)
        first_speed = self.run(smoother, goal, 1)[0][1]
        assert first_speed < 0.1 * (goal.magnitude() / 0.09)

    def test_velocity_stays_continuous_when_the_goal_steps(self):
        smoother = PoseSmoother(tau=0.09)
        near = Rotation.from_euler("z", 5, degrees=True)
        self.run(smoother, near, 60)  # settle

        before = smoother.speed
        far = Rotation.from_euler("z", 45, degrees=True)
        smoother.step(far, self.DT)
        # A 40 degree jump must not translate into an instant velocity jump.
        assert abs(smoother.speed - before) < 0.5

    def test_does_not_overshoot(self):
        smoother = PoseSmoother(tau=0.09)
        goal = Rotation.from_euler("z", 30, degrees=True)
        angles = [
            np.degrees(smoother.step(goal, self.DT).as_euler("zyx")[0])
            for _ in range(300)
        ]
        assert max(angles) <= 30.0 + 1e-6, "critical damping must not overshoot"

    def test_approach_is_monotonic(self):
        smoother = PoseSmoother(tau=0.09)
        goal = Rotation.from_euler("z", 25, degrees=True)
        errs = [e for e, _ in self.run(smoother, goal, 200)]
        assert all(b <= a + 1e-9 for a, b in zip(errs, errs[1:]))

    def test_speed_is_clamped(self):
        smoother = PoseSmoother(tau=0.02, max_speed=1.0)
        goal = Rotation.from_euler("z", 170, degrees=True)
        assert max(s for _, s in self.run(smoother, goal, 100)) <= 1.0 + 1e-9

    def test_a_long_frame_does_not_diverge(self):
        # Sub-stepping must keep explicit integration stable when dt is big.
        smoother = PoseSmoother(tau=0.05)
        goal = Rotation.from_euler("z", 40, degrees=True)
        for _ in range(50):
            smoother.step(goal, 0.1)
        assert smoother.speed < 10.0
        assert float((goal * smoother.rotation.inv()).magnitude()) == pytest.approx(
            0.0, abs=1e-3
        )

    def test_reset_returns_to_rest(self):
        smoother = PoseSmoother(tau=0.09)
        self.run(smoother, Rotation.from_euler("z", 30, degrees=True), 20)
        smoother.reset()
        assert smoother.speed == 0.0
        assert smoother.rotation.magnitude() == pytest.approx(0.0)


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


class TestVocabularyCheck:
    """A server that cannot see our labels detects everything instead, silently."""

    class _Stub:
        url = "http://stub:8100"

        def __init__(self, result):
            self.result = result

        def classes(self):
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    def test_a_missing_label_is_named(self, caplog):
        from tracker.main import TRACK_LABELS, Tracker

        with caplog.at_level("WARNING"):
            Tracker._check_vocabulary(self._Stub([TRACK_LABELS[0]]))
        for label in TRACK_LABELS[1:]:
            assert label in caplog.text

    def test_a_complete_vocabulary_does_not_warn(self, caplog):
        from tracker.main import TRACK_LABELS, Tracker

        with caplog.at_level("WARNING"):
            Tracker._check_vocabulary(self._Stub(list(TRACK_LABELS) + ["mug"]))
        assert caplog.text == ""

    def test_an_unreachable_server_is_not_fatal(self, caplog):
        from tracker.detector import DetectorUnavailable
        from tracker.main import Tracker

        with caplog.at_level("WARNING"):
            Tracker._check_vocabulary(self._Stub(DetectorUnavailable("down")))
        assert "down" in caplog.text


class TestPosture:
    """Lock and loss must not step the commanded antennas or the scan."""

    def test_antennas_are_continuous_across_the_transition(self):
        from tracker.main import Tracker

        t = 3.0
        # The blend scalar moves gradually, so neighbouring levels must too.
        a = Tracker._antennas(0.50, t)[0]
        b = Tracker._antennas(0.51, t)[0]
        assert abs(b - a) < np.deg2rad(1.0)

    def test_antenna_endpoints_are_wag_and_perk(self):
        from tracker.main import Tracker

        t = 0.0  # sine is zero here, so the wag term vanishes
        assert Tracker._antennas(1.0, t)[0] == pytest.approx(np.deg2rad(20.0))
        assert Tracker._antennas(0.0, t)[0] == pytest.approx(0.0)

    def angles(self, rotation):
        """Yaw and pitch in degrees, intrinsic, the way the scan builds them."""
        yaw, pitch, _ = rotation.as_euler("ZYX", degrees=True)
        return yaw, pitch

    def yaw(self, rotation):
        return self.angles(rotation)[0]

    @pytest.mark.parametrize(
        "held", [(0.0, 0.0), (12.0, 5.0), (-20.0, -9.0), (140.0, 17.0)]
    )
    def test_a_scan_begins_where_the_head_already_is(self, held):
        # Regression: the sine ran off a fixed epoch, so starting a scan threw
        # the head to centre and then out to an arbitrary phase.
        from tracker.main import Tracker

        start = Rotation.from_euler("ZY", held, degrees=True)
        phase = Tracker._scan_phase(start)
        assert self.angles(Tracker._scan_pose(0.0, phase)) == pytest.approx(
            held, abs=1e-6
        )

    def test_the_scan_is_continuous_from_its_first_instant(self):
        from tracker.main import (
            CONTROL_HZ,
            SCAN_DEGREES,
            SCAN_HZ,
            SCAN_PITCH_DEGREES,
            SCAN_PITCH_HZ,
            Tracker,
        )

        start = Rotation.from_euler("ZY", [25.0, 8.0], degrees=True)
        phase = Tracker._scan_phase(start)
        moved = Rotation.from_euler(
            "ZY", [25.0, 8.0], degrees=True
        ).inv() * Tracker._scan_pose(1 / CONTROL_HZ, phase)
        # One tick at each axis's own top speed; more than that is a jump, and a
        # fixed bound would only be measuring the amplitude.
        fastest = (
            2 * math.pi * SCAN_HZ * SCAN_DEGREES
            + 2 * math.pi * SCAN_PITCH_HZ * SCAN_PITCH_DEGREES
        ) / CONTROL_HZ
        assert np.degrees(moved.magnitude()) <= fastest, "no jump into a scan"

    def test_a_scan_heads_outward_not_back_to_centre(self):
        from tracker.main import Tracker

        phase = Tracker._scan_phase(Rotation.from_euler("Z", 20.0, degrees=True))
        assert self.yaw(Tracker._scan_pose(0.5, phase)) > 20.0

    def test_an_angle_beyond_the_scan_is_clamped_not_undefined(self):
        from tracker.main import SCAN_DEGREES, SCAN_PITCH_DEGREES, Tracker

        beyond = Rotation.from_euler("ZY", [179.0, 40.0], degrees=True)
        yaw_phase, pitch_phase = Tracker._scan_phase(beyond)
        assert math.isfinite(yaw_phase) and math.isfinite(pitch_phase)

        yaw, pitch = self.angles(Tracker._scan_pose(0.0, (yaw_phase, pitch_phase)))
        assert yaw == pytest.approx(SCAN_DEGREES)
        assert pitch == pytest.approx(SCAN_PITCH_DEGREES)

    def test_an_axis_of_zero_amplitude_is_not_a_nan(self):
        # Setting SCAN_PITCH_DEGREES to 0 to turn pitch off must not divide by it
        # and hand the control loop a NaN pose.
        from tracker.main import Tracker

        assert Tracker._phase_at(12.0, 0.0) == 0.0

    def test_the_scan_looks_all_the_way_round_and_up_and_down(self):
        from tracker.main import SCAN_DEGREES, SCAN_HZ, SCAN_PITCH_DEGREES, Tracker

        seen = [
            self.angles(Tracker._scan_pose(t / 20, (0.0, 0.0)))
            for t in range(int(20 / SCAN_HZ))
        ]
        yaws, pitches = [y for y, _ in seen], [p for _, p in seen]

        assert max(yaws) == pytest.approx(SCAN_DEGREES, abs=0.5)
        assert min(yaws) == pytest.approx(-SCAN_DEGREES, abs=0.5)
        assert max(pitches) == pytest.approx(SCAN_PITCH_DEGREES, abs=0.5)
        assert min(pitches) == pytest.approx(-SCAN_PITCH_DEGREES, abs=0.5)

    def test_the_two_axes_do_not_retrace_one_line(self):
        # Commensurate rates would close the figure and sweep the same stripe
        # forever, leaving most of the room unscanned.
        from tracker.main import SCAN_HZ, SCAN_PITCH_HZ

        ratio = SCAN_PITCH_HZ / SCAN_HZ
        assert abs(ratio - round(ratio)) > 0.1, "pitch must not be a multiple of yaw"


class TestPullTuning:
    """The panel's responsiveness slider must actually change the motion."""

    def profile(self, pull):
        """Peak jerk and peak speed for a 40 degree step at this pull."""
        smoother = PoseSmoother(0.09, 3.5, pull)
        goal = Rotation.from_euler("z", 40, degrees=True)
        speeds = []
        for _ in range(150):
            previous = smoother.rotation
            smoother.step(goal, 1 / 50)
            speeds.append(float((smoother.rotation * previous.inv()).magnitude()) * 50)
        jerk = np.abs(np.diff(speeds)) / (1 / 50)
        return jerk.max(), max(speeds)

    def test_lower_pull_is_gentler(self):
        gentle, brisk = self.profile(4.0), self.profile(60.0)
        assert gentle[0] < brisk[0], "less pull must mean less jerk"
        assert gentle[1] < brisk[1], "less pull must mean less speed"

    def test_every_setting_still_reaches_the_goal(self):
        goal = Rotation.from_euler("z", 40, degrees=True)
        for pull in (4.0, 20.0, 60.0):
            smoother = PoseSmoother(0.09, 3.5, pull)
            for _ in range(600):
                smoother.step(goal, 1 / 50)
            assert float((goal * smoother.rotation.inv()).magnitude()) == pytest.approx(
                0.0, abs=1e-2
            ), f"pull={pull} must still converge"

    def test_no_setting_overshoots(self):
        for pull in (4.0, 20.0, 60.0):
            smoother = PoseSmoother(0.09, 3.5, pull)
            goal = Rotation.from_euler("z", 30, degrees=True)
            angles = [
                np.degrees(smoother.step(goal, 1 / 50).as_euler("zyx")[0])
                for _ in range(400)
            ]
            assert max(angles) <= 30.0 + 1e-6, f"pull={pull} overshot"


class TestNoWhip:
    """A target spotted far off must be approached, never lunged at.

    The scan reaches almost all the way around, so a target can be acquired
    most of a turn away from where the head is pointing.
    """

    def peak_speed(self, offset, pull=None, seconds=30):
        """Fastest the head moves, in deg/s, closing an `offset` degree gap."""
        from tracker.main import MAX_HEAD_PULL, MAX_HEAD_SPEED, SMOOTH_TAU

        smoother = PoseSmoother(SMOOTH_TAU, MAX_HEAD_SPEED, pull or MAX_HEAD_PULL)
        goal = Rotation.from_euler("Z", offset, degrees=True)
        speeds, previous = [], smoother.rotation
        for _ in range(int(seconds * 50)):
            smoother.step(goal, 1 / 50)
            speeds.append(
                np.degrees(float((smoother.rotation * previous.inv()).magnitude())) * 50
            )
            previous = smoother.rotation
        return max(speeds), float((goal * smoother.rotation.inv()).magnitude())

    def terminal(self, pull=None):
        """Speed at which the capped pull balances damping: the real ceiling."""
        from tracker.main import MAX_HEAD_PULL, SMOOTH_TAU

        return np.degrees((pull or MAX_HEAD_PULL) * SMOOTH_TAU / 2)

    @pytest.mark.parametrize("offset", [20, 60, 120, 179])
    def test_speed_does_not_grow_with_distance(self, offset):
        speed, _ = self.peak_speed(offset)
        assert speed <= self.terminal() * 1.05, "a distant target must not whip"

    def test_a_distant_target_is_still_reached(self):
        # Gentle must not mean stuck.
        _, remaining = self.peak_speed(179)
        assert np.degrees(remaining) < 1.0

    def test_raising_responsiveness_gives_up_the_guarantee(self):
        # Worth knowing: the ceiling is the slider's, not a fixed one.
        gentle, _ = self.peak_speed(179, pull=10)
        brisk, _ = self.peak_speed(179, pull=60)
        assert brisk > 4 * gentle
