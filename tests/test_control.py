import threading

import numpy as np
import pytest

from reachy_gaze.main import ReachyGaze, State


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
def app():
    return ReachyGaze()


def drive(app, mini, state):
    app._drive(mini, state, mini.stop_event)
    return mini.commands


def test_survives_a_pose_that_is_not_published_yet(app):
    # Regression: this used to raise straight out of the control loop, which
    # took down the app and left it restarting in a loop.
    mini = FakeMini(pose_ready=False)
    commands = drive(app, mini, State())

    assert len(commands) >= 6, "the loop must keep commanding without a pose"
    assert all(head is not None for head, _ in commands)


def test_keeps_the_last_known_pose_when_it_goes_missing(app):
    state = State()
    seeded = np.eye(4)
    seeded[0, 3] = 0.123
    state.head_pose = seeded

    drive(app, FakeMini(pose_ready=False), state)
    assert state.head_pose[0, 3] == pytest.approx(0.123), "must not clobber it"


def test_caches_the_pose_when_it_is_available(app):
    state = State()
    drive(app, FakeMini(pose_ready=True), state)
    assert state.head_pose == pytest.approx(np.eye(4))


def test_commands_are_finite_and_well_formed(app):
    mini = FakeMini(pose_ready=True)
    for head, antennas in drive(app, mini, State()):
        assert head.shape == (4, 4)
        assert np.isfinite(head).all()
        assert len(antennas) == 2
        assert np.isfinite(antennas).all()


def test_tracked_labels_are_real_coco_classes():
    # A typo here is near-silent: the server recognises none of the labels and
    # falls back to detecting everything, so the head chases furniture.
    from reachy_gaze.detector import COCO_CLASSES
    from reachy_gaze.main import TRACK_LABELS

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

        from reachy_gaze.main import State

        state = State()
        state.goal = Rotation.identity()
        state.last_seen = time.monotonic() - seconds_ago
        return state.snapshot()

    def test_no_target_is_not_locked(self):
        from reachy_gaze.main import State

        snap = State().snapshot()
        assert snap["locked"] is False
        assert snap["seen_ago"] is None

    def test_a_fresh_sighting_is_locked(self):
        snap = self.state_at(0.1)
        assert snap["locked"] is True
        assert snap["seen_ago"] == pytest.approx(0.1, abs=0.2)

    @pytest.mark.parametrize("ago", [2.0, 5.0, 9.0])
    def test_the_aim_is_held_well_past_the_last_sighting(self, ago):
        from reachy_gaze.main import LOST_AFTER

        assert LOST_AFTER >= 10.0, "the hold window is what this guards"
        assert self.state_at(ago)["locked"] is True

    def test_the_lock_is_released_past_the_window(self):
        snap = self.state_at(11.0)
        assert snap["locked"] is False
        assert snap["seen_ago"] is None, "no point reporting staleness once given up"


class TestOtherTargets:
    """The panel is told about every box in view, not just the one we track."""

    def test_none_in_view_by_default(self):
        assert State().snapshot()["targets"] == []

    def test_other_boxes_are_exposed_to_the_panel(self):
        state = State()
        state.targets = [{"label": "person", "center": [0.5, -0.2]}]
        assert state.snapshot()["targets"] == [
            {"label": "person", "center": [0.5, -0.2]}
        ]


class TestLens:
    """The panel is told the camera intrinsics once a frame has arrived."""

    def test_unknown_until_a_frame(self):
        assert State().snapshot()["lens"] is None


class TestAim:
    """The panel is told where the head is pointing, in yaw/pitch degrees."""

    def test_straight_ahead_by_default(self):
        assert State().snapshot()["aim"] == {"yaw": 0.0, "pitch": 0.0}

    def test_yaw_is_positive_to_the_left(self):
        from scipy.spatial.transform import Rotation

        from reachy_gaze.main import look_yaw_pitch

        # +Y is left in the head frame, so a +30 deg turn about Z reads as left.
        head = np.eye(4)
        head[:3, :3] = Rotation.from_euler("Z", 30, degrees=True).as_matrix()
        yaw, pitch = look_yaw_pitch(head)
        assert yaw == pytest.approx(30.0)
        assert pitch == pytest.approx(0.0, abs=1e-9)

    def test_pitch_is_positive_looking_up(self):
        from scipy.spatial.transform import Rotation

        from reachy_gaze.main import look_yaw_pitch

        # A -20 deg rotation about Y lifts the forward +X axis, i.e. looks up.
        head = np.eye(4)
        head[:3, :3] = Rotation.from_euler("Y", -20, degrees=True).as_matrix()
        yaw, pitch = look_yaw_pitch(head)
        assert pitch == pytest.approx(20.0)
        assert yaw == pytest.approx(0.0, abs=1e-9)


class TestBackendSelection:
    """The detector backend is selectable, and the panel is told the options."""

    def test_the_default_backend_is_reported(self):
        from reachy_gaze.main import DEFAULT_BACKEND

        assert State().snapshot()["backend"] == DEFAULT_BACKEND

    def test_the_options_are_listed_for_the_panel(self):
        listed = {b["key"] for b in State().snapshot()["backends"]}
        assert {"triton", "builtin", "vlm"} <= listed

    def test_config_accepts_a_known_backend(self):
        from reachy_gaze.main import Config

        assert Config(backend="builtin").backend == "builtin"

    def test_config_rejects_an_unknown_backend(self):
        from pydantic import ValidationError

        from reachy_gaze.main import Config

        with pytest.raises(ValidationError):
            Config(backend="nope")


class TestCyclePublishing:
    """The panel, and anything recording, can tell where the head is in the cycle."""

    def test_starts_idle(self):
        snap = State().snapshot()
        assert snap["phase"] == "idle" and snap["cycle"] == 0
        assert snap["bored_in"] is None and snap["still_for"] is None

    def test_the_first_tick_starts_a_cycle(self, app):
        state = State()
        drive(app, FakeMini(), state)
        snap = state.snapshot()
        assert snap["phase"] == "moving" and snap["cycle"] == 1

    def test_a_disabled_app_stays_idle(self, app):
        state = State()
        state.enabled = False
        drive(app, FakeMini(), state)
        assert state.snapshot()["phase"] == "idle"

    def test_boredom_counts_down_while_tracking_a_still_target(self):
        import time

        from reachy_gaze.cycle import Phase

        state = State()
        state.phase = Phase.TRACKING
        state.bored_after = 20.0
        state.stillness.update(np.array([1.0, 0.0, 0.0]), time.monotonic() - 5.0)
        snap = state.snapshot()
        assert snap["still_for"] == pytest.approx(5.0, abs=0.2)
        assert snap["bored_in"] == pytest.approx(15.0, abs=0.2)

    def test_no_countdown_outside_tracking(self):
        import time

        state = State()
        state.stillness.update(np.array([1.0, 0.0, 0.0]), time.monotonic())
        assert state.snapshot()["bored_in"] is None


class TestEnteringAPhase:
    def cycle_in(self, phase):
        from reachy_gaze.cycle import Cycle

        cycle = Cycle(lambda: None, 1.0, 1.0, 1.0)
        cycle.phase, cycle.count = phase, 3
        return cycle

    def tracking_state(self):
        from scipy.spatial.transform import Rotation

        state = State()
        state.goal, state.center, state.label = Rotation.identity(), (0.0, 0.0), "cat"
        state.stillness.update(np.array([1.0, 0.0, 0.0]), 0.0)
        return state

    def test_a_new_cycle_drops_the_old_target(self):
        from reachy_gaze.cycle import Phase

        state = self.tracking_state()
        ReachyGaze._enter_phase(state, self.cycle_in(Phase.MOVING), 7.0)
        assert state.goal is None and state.center is None and state.label == ""
        assert state.stillness.still_for(8.0) is None
        assert (state.phase, state.phase_since, state.cycle) == (Phase.MOVING, 7.0, 3)

    def test_losing_a_target_keeps_the_held_aim_but_not_the_boredom(self):
        from reachy_gaze.cycle import Phase

        state = self.tracking_state()
        ReachyGaze._enter_phase(state, self.cycle_in(Phase.SCANNING), 7.0)
        assert state.goal is not None
        assert state.stillness.still_for(8.0) is None

    def test_starting_to_track_keeps_the_boredom_clock(self):
        from reachy_gaze.cycle import Phase

        state = self.tracking_state()
        ReachyGaze._enter_phase(state, self.cycle_in(Phase.TRACKING), 7.0)
        assert state.stillness.still_for(8.0) == 8.0


class TestBoredomSetting:
    def test_accepted_from_the_panel(self):
        from reachy_gaze.main import Config

        assert Config(bored_after=45.0).bored_after == 45.0

    def test_zero_is_refused(self):
        from pydantic import ValidationError

        from reachy_gaze.main import Config

        with pytest.raises(ValidationError):
            Config(bored_after=0.0)


class TestLockingByPhase:
    """End to end through the vision loop: only a scan or a track takes a lock."""

    class Camera:
        K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
        D = np.zeros(5)

    class Media:
        camera = None

        def get_frame(self):
            return np.zeros((480, 640, 3), np.uint8)

    class Mini:
        def __init__(self, media):
            self.media = media

    class Cat:
        def classes(self):
            return ["cat"]

        def detect(self, frame, labels, conf):
            from reachy_gaze.detector import Detection

            return [Detection("cat", 0.9, (300.0, 220.0, 380.0, 300.0))]

    def run_vision(self, monkeypatch, phase, seconds=0.3):
        import reachy_gaze.main as main

        monkeypatch.setattr(main, "DETECT_HZ", 100.0)
        monkeypatch.setattr(main, "make_detector", lambda backend, url: self.Cat())
        media = self.Media()
        media.camera = self.Camera()
        state, stop = State(), threading.Event()
        state.phase = phase
        vision = threading.Thread(
            target=ReachyGaze()._track_forever,
            args=(self.Mini(media), state, stop),
            daemon=True,
        )
        vision.start()
        stop.wait(seconds)
        stop.set()
        vision.join(timeout=2.0)
        return state

    @pytest.mark.parametrize("phase", ["moving", "holding", "idle"])
    def test_no_lock_while_not_looking(self, monkeypatch, phase):
        from reachy_gaze.cycle import Phase

        state = self.run_vision(monkeypatch, Phase(phase))
        assert state.goal is None
        assert state.stillness.still_for(0.0) is None
        assert state.targets, "what is in view is still shown on the panel"

    def test_a_scan_takes_the_lock_and_starts_the_boredom_clock(self, monkeypatch):
        from reachy_gaze.cycle import Phase

        state = self.run_vision(monkeypatch, Phase.SCANNING)
        assert state.goal is not None and state.label == "cat"
        assert state.stillness.still_for(state.last_seen) is not None


@pytest.fixture(scope="module")
def kin():
    from reachy_mini.kinematics.analytical_kinematics import AnalyticalKinematics

    return AnalyticalKinematics(automatic_body_yaw=True)


class TestReach:
    """Random poses are checked against the kinematics the daemon uses."""

    def test_neutral_is_reachable(self, kin):
        from scipy.spatial.transform import Rotation

        from reachy_gaze.main import head_can_reach

        assert head_can_reach(kin, (Rotation.identity(), np.zeros(3)))

    def test_an_extreme_pose_is_not(self, kin):
        from scipy.spatial.transform import Rotation

        from reachy_gaze.main import head_can_reach

        tilted = Rotation.from_euler("YX", [40, 40], degrees=True)
        assert not head_can_reach(kin, (tilted, np.array([0.0, 0.0, 0.05])))

    def test_the_configured_ranges_do_hit_unreachable_poses(self, kin):
        # Otherwise the check would be dead weight and the ranges could widen.
        import random

        from reachy_gaze.cycle import random_pose
        from reachy_gaze.main import (
            RANDOM_SHIFT,
            RANDOM_TILT,
            RANDOM_YAW,
            head_can_reach,
        )

        rng = random.Random(0)
        draws = [
            random_pose(rng, RANDOM_YAW, RANDOM_TILT, RANDOM_SHIFT) for _ in range(200)
        ]
        assert not all(head_can_reach(kin, p) for p in draws)

    def test_every_chosen_pose_is_reachable(self, kin):
        import random

        from reachy_gaze.cycle import random_pose
        from reachy_gaze.main import (
            RANDOM_SHIFT,
            RANDOM_TILT,
            RANDOM_YAW,
            head_can_reach,
        )

        rng = random.Random(0)
        for _ in range(200):
            pose = random_pose(
                rng,
                RANDOM_YAW,
                RANDOM_TILT,
                RANDOM_SHIFT,
                lambda p: head_can_reach(kin, p),
            )
            assert head_can_reach(kin, pose)
