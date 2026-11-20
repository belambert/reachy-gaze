import math

import numpy as np
import pytest

from reachy_gaze.cycle import Cycle, Ignored, Phase, Stillness


def facts(enabled=True, aimed=False, bored=False):
    return dict(enabled=enabled, aimed=aimed, bored=bored)


def at(phase):
    """A cycle driven to `phase`."""
    cycle = Cycle()
    route = {
        Phase.IDLE: [],
        Phase.SCANNING: [facts()],
        Phase.TRACKING: [facts(), facts(aimed=True)],
    }[phase]
    for f in route:
        cycle.update(0.0, **f)
    assert cycle.phase is phase
    return cycle


class TestCycle:
    def test_starts_idle_and_scans_once_enabled(self):
        cycle = Cycle()
        assert cycle.phase is Phase.IDLE
        assert cycle.update(1.0, **facts())
        assert (cycle.phase, cycle.since) == (Phase.SCANNING, 1.0)

    def test_stays_idle_while_disabled(self):
        cycle = Cycle()
        assert not cycle.update(1.0, **facts(enabled=False))

    def test_a_find_while_scanning_starts_tracking(self):
        cycle = at(Phase.SCANNING)
        cycle.update(3.0, **facts(aimed=True))
        assert cycle.phase is Phase.TRACKING

    def test_scanning_goes_on_until_something_turns_up(self):
        assert not at(Phase.SCANNING).update(1000.0, **facts())

    def test_tracking_continues_while_not_bored(self):
        assert not at(Phase.TRACKING).update(1000.0, **facts(aimed=True))

    def test_boredom_goes_back_to_scanning(self):
        cycle = at(Phase.TRACKING)
        cycle.update(1.0, **facts(aimed=True, bored=True))
        assert cycle.phase is Phase.SCANNING

    def test_a_lost_target_resumes_scanning(self):
        cycle = at(Phase.TRACKING)
        cycle.update(1.0, **facts(aimed=False))
        assert cycle.phase is Phase.SCANNING

    @pytest.mark.parametrize("phase", [Phase.SCANNING, Phase.TRACKING])
    def test_disabling_goes_idle_from_anywhere(self, phase):
        cycle = at(phase)
        cycle.update(1.0, **facts(enabled=False))
        assert cycle.phase is Phase.IDLE

    def test_phases_publish_as_plain_strings(self):
        assert [p.value for p in Phase] == ["idle", "scanning", "tracking"]


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


class TestIgnored:
    def test_nothing_is_ignored_at_first(self):
        assert not Ignored(15.0, 60.0).covers(toward(0), 0.0)

    def test_ignores_near_a_bored_target(self):
        ig = Ignored(15.0, 60.0)
        ig.add(toward(30), 0.0)
        assert ig.covers(toward(40), 1.0)
        assert not ig.covers(toward(50), 1.0), "a target off to the side counts"

    def test_forgets_after_a_while(self):
        ig = Ignored(15.0, 60.0)
        ig.add(toward(30), 0.0)
        assert ig.covers(toward(30), 59.0)
        assert not ig.covers(toward(30), 61.0)

    def test_remembers_several(self):
        ig = Ignored(15.0, 60.0)
        ig.add(toward(-40), 0.0)
        ig.add(toward(40), 10.0)
        assert ig.covers(toward(-40), 20.0) and ig.covers(toward(40), 20.0)

    def test_clear_forgets_everything(self):
        ig = Ignored(15.0, 60.0)
        ig.add(toward(0), 0.0)
        ig.clear()
        assert not ig.covers(toward(0), 1.0)
