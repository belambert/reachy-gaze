import threading

import numpy as np
import pytest

from tracker.main import State, Tracker


class FakeMini:
    """Stands in for the robot: records commands, optionally withholds the pose."""

    def __init__(self, pose_ready=True, stop_after=6):
        self.pose_ready = pose_ready
        self.stop_after = stop_after
        self.commands = []
        self.stop_event = threading.Event()

    def get_current_head_pose(self):
        # What the SDK really does before the daemon has published one.
        if not self.pose_ready:
            raise AssertionError("No head pose received yet.")
        return np.eye(4)

    def set_target(self, head=None, antennas=None, body_yaw=None):
        self.commands.append((head, antennas))
        if len(self.commands) >= self.stop_after:
            self.stop_event.set()


@pytest.fixture
def tracker():
    return Tracker()


def drive(tracker, mini, state):
    tracker._drive(mini, state, mini.stop_event)
    return mini.commands


def test_survives_a_pose_that_is_not_published_yet(tracker):
    # Regression: this used to raise straight out of the control loop, which
    # took down the app and left it restarting in a loop.
    mini = FakeMini(pose_ready=False)
    commands = drive(tracker, mini, State())

    assert len(commands) >= 6, "the loop must keep commanding without a pose"
    assert all(head is not None for head, _ in commands)


def test_keeps_the_last_known_pose_when_it_goes_missing(tracker):
    state = State()
    seeded = np.eye(4)
    seeded[0, 3] = 0.123
    state.head_pose = seeded

    drive(tracker, FakeMini(pose_ready=False), state)
    assert state.head_pose[0, 3] == pytest.approx(0.123), "must not clobber it"


def test_caches_the_pose_when_it_is_available(tracker):
    state = State()
    drive(tracker, FakeMini(pose_ready=True), state)
    assert state.head_pose == pytest.approx(np.eye(4))


def test_commands_are_finite_and_well_formed(tracker):
    mini = FakeMini(pose_ready=True)
    for head, antennas in drive(tracker, mini, State()):
        assert head.shape == (4, 4)
        assert np.isfinite(head).all()
        assert len(antennas) == 2
        assert np.isfinite(antennas).all()


def test_tracked_labels_are_real_coco_classes():
    # A typo here is near-silent: the server recognises none of the labels and
    # falls back to detecting everything, so the head chases furniture.
    from tracker.detector import COCO_CLASSES
    from tracker.main import TRACK_LABELS

    assert set(TRACK_LABELS) <= set(COCO_CLASSES)


def test_tracking_starts_on():
    # Regression: a restart used to come back unticked, which reads exactly
    # like a hang — the class list is fetched and then nothing else happens.
    assert State().enabled is True
    assert State().scan is True


class TestHoldWindow:
    """The head keeps its aim for LOST_AFTER, and the panel must say so honestly."""

    def state_at(self, seconds_ago):
        import time

        from scipy.spatial.transform import Rotation

        from tracker.main import State

        state = State()
        state.goal = Rotation.identity()
        state.last_seen = time.monotonic() - seconds_ago
        return state.snapshot()

    def test_no_target_is_not_locked(self):
        from tracker.main import State

        snap = State().snapshot()
        assert snap["locked"] is False
        assert snap["seen_ago"] is None

    def test_a_fresh_sighting_is_locked(self):
        snap = self.state_at(0.1)
        assert snap["locked"] is True
        assert snap["seen_ago"] == pytest.approx(0.1, abs=0.2)

    @pytest.mark.parametrize("ago", [2.0, 5.0, 9.0])
    def test_the_aim_is_held_well_past_the_last_sighting(self, ago):
        from tracker.main import LOST_AFTER

        assert LOST_AFTER >= 10.0, "the hold window is what this guards"
        assert self.state_at(ago)["locked"] is True

    def test_the_lock_is_released_past_the_window(self):
        snap = self.state_at(11.0)
        assert snap["locked"] is False
        assert snap["seen_ago"] is None, "no point reporting staleness once given up"
