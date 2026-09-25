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


class Clock:
    """A hand-advanced stand-in for time.monotonic."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class TestBoredom:
    """Holding one target too long must break the lock so the head scans again.

    What to shun while looking away is the caller's decision (the world model's,
    in the app); the tests pass the per-box mask directly.
    """

    def selector(self, clock, **kw):
        return TargetSelector(max_lock=10.0, time_fn=clock, **kw)

    def test_no_max_lock_holds_the_lock_forever(self):
        # The default (no boredom timer) must not change: it stays locked.
        sel = TargetSelector()
        sel.select([box(320, 240, size=150)], W, H)
        for _ in range(1000):
            sel.select([box(320, 240, size=150)], W, H)
        assert sel.has_target

    def test_keeps_the_lock_until_the_timer_expires(self):
        clock = Clock()
        sel = self.selector(clock)
        cat = box(320, 240, size=150)
        sel.select([cat], W, H, [True])

        clock.t = 9.9
        assert sel.select([cat], W, H, [True]) is cat, "shunning waits for boredom"
        assert sel.has_target and not sel.bored

    def test_drops_the_lock_once_bored(self):
        clock = Clock()
        sel = self.selector(clock)
        cat = box(320, 240, size=150)
        sel.select([cat], W, H)

        clock.t = 10.0
        assert sel.select([cat], W, H, [True]) is None, "the sole target is shunned"
        assert not sel.has_target and sel.bored

    def test_without_a_mask_nothing_is_shunned(self):
        clock = Clock()
        sel = self.selector(clock)
        cat = box(320, 240, size=150)
        sel.select([cat], W, H)

        clock.t = 10.0
        assert sel.select([cat], W, H) is cat, "dropped, but promptly re-taken"

    def test_keeps_shunning_while_it_drifts_across_the_frame(self):
        clock = Clock()
        sel = self.selector(clock)
        sel.select([box(100, 240)], W, H)

        clock.t = 10.0
        for cx in range(100, 620, 30):
            clock.t += 0.1
            assert sel.select([box(cx, 240)], W, H, [True]) is None, f"took it at {cx}"
        assert sel.bored

    def test_skips_every_shunned_box_for_one_that_is_not(self):
        clock = Clock()
        sel = self.selector(clock)
        old, older, fresh = box(100, 240), box(320, 240, size=200), box(560, 240)
        sel.select([old], W, H)

        clock.t = 10.0
        # The biggest box is shunned too: only `fresh` may be taken.
        assert sel.select([old, older, fresh], W, H, [True, True, False]) is fresh

    def test_stays_bored_for_as_long_as_the_target_is_shunned(self):
        clock = Clock()
        sel = self.selector(clock)
        cat = box(320, 240, size=150)
        sel.select([cat], W, H)

        clock.t = 10.0
        assert sel.select([cat], W, H, [True]) is None
        clock.t = 1000.0  # no timer of its own: the mask decides
        assert sel.select([cat], W, H, [True]) is None
        assert sel.bored

    def test_returns_to_the_only_target_once_it_is_no_longer_shunned(self):
        clock = Clock()
        sel = self.selector(clock)
        cat = box(320, 240, size=150)
        sel.select([cat], W, H)

        clock.t = 10.0
        assert sel.select([cat], W, H, [True]) is None
        got = sel.select([cat], W, H, [False])  # its dwell has aged out
        assert got is cat, "comes back when nothing else turns up"
        assert not sel.bored

    def test_bored_at_is_the_lock_time_plus_the_timeout(self):
        clock = Clock()
        sel = self.selector(clock)
        assert sel.bored_at is None, "no lock, nothing to tire of"

        clock.t = 3.0
        sel.select([box(320, 240, size=150)], W, H)
        assert sel.bored_at == 13.0

    def test_bored_at_clears_when_bored_or_lost(self):
        clock = Clock()
        sel = self.selector(clock, max_misses=0)
        cat = box(320, 240, size=150)
        sel.select([cat], W, H)
        clock.t = 10.0
        sel.select([cat], W, H, [True])
        assert sel.bored_at is None

        sel = self.selector(clock, max_misses=0)
        sel.select([cat], W, H)
        sel.select([], W, H)  # lost
        assert sel.bored_at is None

    def test_no_max_lock_is_never_bored(self):
        sel = TargetSelector()
        sel.select([box(320, 240, size=150)], W, H)
        assert sel.bored_at is None

    def test_the_timer_restarts_on_the_new_target(self):
        clock = Clock()
        sel = self.selector(clock)
        here, there = box(100, 240, size=150), box(560, 240, size=150)
        sel.select([here, there], W, H)

        clock.t = 10.0
        sel.select([here, there], W, H, [True, False])  # swings to the other object
        assert not sel.bored, "a fresh lock ends the boredom"
        clock.t = 15.0  # 5s on the new one: not yet bored again
        assert sel.select([here, there], W, H, [True, False]) is there
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
        "held", [(0.0, 0.0), (12.0, 5.0), (-20.0, -9.0), (50.0, 17.0)]
    )
    def test_a_scan_begins_at_the_yaw_the_head_already_has(self, held):
        # Regression: the sine ran off a fixed epoch, so starting a scan threw
        # the head to centre and then out to an arbitrary phase.
        from tracker.main import Tracker

        start = Rotation.from_euler("ZY", held, degrees=True)
        phase = Tracker._scan_phase(start)
        assert self.yaw(Tracker._scan_pose(0.0, phase)) == pytest.approx(
            held[0], abs=1e-6
        )

    def test_the_scan_is_continuous_from_its_first_instant(self):
        from tracker.main import CONTROL_HZ, SCAN_DEGREES, SCAN_HZ, Tracker

        start = Rotation.from_euler("Z", 25.0, degrees=True)
        phase = Tracker._scan_phase(start)
        moved = start.inv() * Tracker._scan_pose(1 / CONTROL_HZ, phase)
        # One tick at the scan's top speed; more than that is a jump, and a
        # fixed bound would only be measuring the amplitude.
        fastest = 2 * math.pi * SCAN_HZ * SCAN_DEGREES / CONTROL_HZ
        assert np.degrees(moved.magnitude()) <= fastest, "no jump into a scan"

    def test_a_scan_heads_outward_not_back_to_centre(self):
        from tracker.main import Tracker

        phase = Tracker._scan_phase(Rotation.from_euler("Z", 20.0, degrees=True))
        assert self.yaw(Tracker._scan_pose(0.5, phase)) > 20.0

    def test_a_scan_from_the_left_heads_further_left_not_back(self):
        # Regression: asin always set off toward +yaw, so a head panned to a
        # negative yaw turned straight back to centre and never swept that side.
        from tracker.main import Tracker

        phase = Tracker._scan_phase(Rotation.from_euler("Z", -20.0, degrees=True))
        assert self.yaw(Tracker._scan_pose(0.5, phase)) < -20.0

    def test_a_scan_continues_the_way_the_head_is_turning(self):
        # Still turning negative as the target is lost: keep going, don't reverse.
        from tracker.main import Tracker

        start = Rotation.from_euler("Z", 0.0, degrees=True)
        phase = Tracker._scan_phase(start, rate=-1.0)
        assert self.yaw(Tracker._scan_pose(0.5, phase)) < 0.0

    def test_an_angle_beyond_the_scan_is_clamped_not_undefined(self):
        from tracker.main import SCAN_DEGREES, Tracker

        beyond = Rotation.from_euler("ZY", [179.0, 40.0], degrees=True)
        phase = Tracker._scan_phase(beyond)
        assert math.isfinite(phase)
        assert self.yaw(Tracker._scan_pose(0.0, phase)) == pytest.approx(SCAN_DEGREES)

    def test_an_axis_of_zero_amplitude_is_not_a_nan(self):
        # Setting SCAN_DEGREES to 0 to turn the scan off must not divide by it
        # and hand the control loop a NaN pose.
        from tracker.main import Tracker

        assert Tracker._phase_at(12.0, 0.0) == 0.0

    def test_the_scan_looks_all_the_way_round_and_stays_level(self):
        from tracker.main import SCAN_DEGREES, SCAN_HZ, Tracker

        # Even starting from a head tilted up or down, the scan only turns.
        phase = Tracker._scan_phase(
            Rotation.from_euler("ZY", [10.0, 15.0], degrees=True)
        )
        seen = [
            self.angles(Tracker._scan_pose(t / 20, phase))
            for t in range(int(20 / SCAN_HZ))
        ]
        yaws, pitches = [y for y, _ in seen], [p for _, p in seen]

        assert max(yaws) == pytest.approx(SCAN_DEGREES, abs=0.5)
        assert min(yaws) == pytest.approx(-SCAN_DEGREES, abs=0.5)
        assert pitches == pytest.approx([0.0] * len(pitches), abs=1e-9)


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

    A target can be acquired well off to one side of where the head is
    pointing, at the far end of a scan or after a hold elsewhere.
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


class TestPriority:
    """Earlier in TRACK_LABELS wins: a cat in view outranks a person in view."""

    PRIORITY = ["cat", "dog", "person"]

    def selector(self, **kw):
        return TargetSelector(self.PRIORITY, **kw)

    def test_acquires_the_preferred_class_over_a_bigger_one(self):
        cat = box(200, 240, size=80, label="cat")
        person = box(450, 240, size=300, label="person")
        assert self.selector().select([person, cat], W, H) is cat

    def test_falls_back_when_the_preferred_class_is_absent(self):
        person = box(450, 240, size=200, label="person")
        assert self.selector().select([person], W, H) is person

    def test_leaves_a_person_for_a_cat(self):
        sel = self.selector(upgrade_after=3)
        person = box(450, 240, size=200, label="person")
        sel.select([person], W, H)
        assert sel.label == "person"

        cat = box(200, 240, size=80, label="cat")
        for _ in range(3):
            sel.select([person, cat], W, H)
        assert sel.label == "cat", "a cat in view must take the lock"

    def test_a_flickering_cat_does_not_bounce_the_lock(self):
        # One frame of a marginal detection must not throw the head across the
        # room and back again.
        sel = self.selector(upgrade_after=3)
        person = box(450, 240, size=200, label="person")
        sel.select([person], W, H)

        cat = box(200, 240, size=80, label="cat")
        for _ in range(10):
            sel.select([person, cat], W, H)
            sel.select([person], W, H)  # the cat drops out again
        assert sel.label == "person", "an intermittent cat must not win"

    def test_it_does_not_leave_a_cat_for_a_person(self):
        sel = self.selector()
        cat = box(320, 240, size=100, label="cat")
        sel.select([cat], W, H)

        person = box(340, 240, size=400, label="person")
        for _ in range(10):
            sel.select([person, cat], W, H)
        assert sel.label == "cat", "the preference does not run backwards"

    def test_an_unlisted_class_ranks_last(self):
        sel = self.selector()
        assert sel._rank("bird") > sel._rank("person")

    def test_a_preferred_speck_does_not_steal_the_lock(self):
        # The area gate still applies to an upgrade, or a stray pixel of cat
        # would take the head off a person standing right there.
        sel = self.selector(upgrade_after=1)
        person = box(450, 240, size=200, label="person")
        sel.select([person], W, H)

        for _ in range(5):
            sel.select([person, box(100, 100, size=4, label="cat")], W, H)
        assert sel.label == "person"

    def test_reset_forgets_the_class_too(self):
        sel = self.selector()
        sel.select([box(320, 240, size=100, label="cat")], W, H)
        sel.reset()
        assert sel.label is None
