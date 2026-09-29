import math
import random

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_gaze.cycle import Cycle, Phase, Stillness, random_pose

HOLD, MOVE_TIMEOUT, IDLE_RESET = 1.5, 5.0, 30.0
POSE = (Rotation.from_euler("Z", 40, degrees=True), np.array([0.01, 0.0, 0.0]))


def facts(enabled=True, arrived=False, aimed=False, bored=False):
    return dict(enabled=enabled, arrived=arrived, aimed=aimed, bored=bored)


@pytest.fixture
def cycle():
    return Cycle(lambda: POSE, HOLD, MOVE_TIMEOUT, IDLE_RESET)


def at(cycle, phase, now=0.0):
    """Drive a fresh cycle to `phase`, entered at `now`."""
    route = {
        Phase.MOVING: [facts()],
        Phase.HOLDING: [facts(), facts(arrived=True)],
        Phase.SCANNING: [facts(), facts(arrived=True), facts()],
        Phase.TRACKING: [facts(), facts(arrived=True), facts(), facts(aimed=True)],
    }[phase]
    t = now - 100.0
    for f in route:
        # holds and timeouts elapse between steps, so each one moves on
        cycle.update(t, **f)
        t += HOLD
    cycle.since = now
    assert cycle.phase is phase
    return cycle


class TestCycle:
    def test_starts_idle_and_moves_once_enabled(self, cycle):
        assert cycle.phase is Phase.IDLE
        assert cycle.update(0.0, **facts())
        assert cycle.phase is Phase.MOVING
        assert cycle.count == 1
        assert cycle.pose is POSE

    def test_holds_once_the_head_arrives(self, cycle):
        at(cycle, Phase.MOVING)
        assert not cycle.update(0.5, **facts())
        cycle.update(0.6, **facts(arrived=True))
        assert cycle.phase is Phase.HOLDING

    def test_a_move_that_never_settles_still_ends(self, cycle):
        at(cycle, Phase.MOVING)
        cycle.update(MOVE_TIMEOUT, **facts())
        assert cycle.phase is Phase.HOLDING

    def test_a_find_while_holding_waits_for_the_scan(self, cycle):
        at(cycle, Phase.HOLDING)
        cycle.update(HOLD - 0.1, **facts(aimed=True))
        assert cycle.phase is Phase.HOLDING, "the hold is not cut short"
        cycle.update(HOLD, **facts())
        assert cycle.phase is Phase.SCANNING

    def test_a_find_while_scanning_starts_tracking(self, cycle):
        at(cycle, Phase.SCANNING)
        cycle.update(3.0, **facts(aimed=True))
        assert cycle.phase is Phase.TRACKING

    def test_a_fruitless_scan_starts_a_new_cycle(self, cycle):
        at(cycle, Phase.SCANNING)
        before = cycle.count
        cycle.update(IDLE_RESET, **facts())
        assert cycle.phase is Phase.MOVING
        assert cycle.count == before + 1

    def test_tracking_continues_while_not_bored(self, cycle):
        at(cycle, Phase.TRACKING)
        assert not cycle.update(1000.0, **facts(aimed=True))

    def test_boredom_starts_a_new_cycle(self, cycle):
        at(cycle, Phase.TRACKING)
        before = cycle.count
        cycle.update(1.0, **facts(aimed=True, bored=True))
        assert cycle.phase is Phase.MOVING
        assert cycle.count == before + 1

    def test_a_lost_target_resumes_scanning(self, cycle):
        at(cycle, Phase.TRACKING)
        cycle.update(1.0, **facts(aimed=False))
        assert cycle.phase is Phase.SCANNING
        assert cycle.count == 1, "losing a target is not a new cycle"

    @pytest.mark.parametrize("phase", list(Phase)[1:])
    def test_disabling_goes_idle_from_anywhere(self, cycle, phase):
        at(cycle, phase)
        cycle.update(1.0, **facts(enabled=False))
        assert cycle.phase is Phase.IDLE
        rot, pos = cycle.pose
        assert rot.magnitude() == 0.0 and not pos.any(), "idle rests at neutral"

    def test_phases_publish_as_plain_strings(self):
        assert [p.value for p in Phase] == [
            "idle",
            "moving",
            "holding",
            "scanning",
            "tracking",
        ]


def toward(yaw_degrees):
    a = math.radians(yaw_degrees)
    return np.array([math.cos(a), math.sin(a), 0.0])


class TestStillness:
    def test_no_target_no_time(self):
        assert Stillness(8.0).still_for(5.0) is None

    def test_jitter_inside_the_threshold_counts_as_still(self):
        s = Stillness(8.0)
        for t, yaw in [(0, 0), (1, 3), (2, -4), (3, 5)]:
            s.update(toward(yaw), t)
        assert s.still_for(3.0) == 3.0

    def test_moving_past_the_threshold_restarts_the_clock(self):
        s = Stillness(8.0)
        s.update(toward(0), 0.0)
        s.update(toward(10), 4.0)
        assert s.still_for(5.0) == 1.0

    def test_steady_motion_never_gets_old(self):
        s = Stillness(8.0)
        for t in range(20):
            s.update(toward(10 * t), float(t))
        assert s.still_for(19.0) == 0.0

    def test_reset_forgets(self):
        s = Stillness(8.0)
        s.update(toward(0), 0.0)
        s.reset()
        assert s.still_for(1.0) is None


def test_random_poses_stay_within_limits():
    rng = random.Random(0)
    for _ in range(200):
        rot, pos = random_pose(rng, yaw=90.0, tilt=15.0, shift=0.01)
        yaw, pitch, roll = rot.as_euler("ZYX", degrees=True)
        assert abs(yaw) <= 90.0 and abs(pitch) <= 15.0 and abs(roll) <= 15.0
        assert np.all(np.abs(pos) <= 0.01)


def test_random_poses_vary():
    rng = random.Random(0)
    yaws = {
        round(random_pose(rng, 90.0, 15.0, 0.01)[0].as_euler("ZYX")[0], 3)
        for _ in range(10)
    }
    assert len(yaws) == 10
