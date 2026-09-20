"""The survey state machine, driven through the real vision thread.

`_track_forever` is where dwelling, looking around and remembering the target
we left all interact, and those interactions are what have gone wrong. This
drives it against a stubbed camera and detector, with the timings shrunk so a
whole dwell-survey cycle takes a fraction of a second.
"""

import threading

import numpy as np
import pytest

import tracker.main as main
from tracker.detector import COCO_CLASSES, Detection
from tracker.main import State, Tracker

W, H = 640, 480


class FakeCamera:
    K = np.array([[500.0, 0, W / 2], [0, 500.0, H / 2], [0, 0, 1]])
    D = np.zeros(5)


class FakeMedia:
    camera = FakeCamera()

    def get_frame(self):
        return np.zeros((H, W, 3), np.uint8)


class FakeMini:
    media = FakeMedia()


def box(cx, cy, size=160, label="cat"):
    half = size / 2
    return Detection(label, 0.9, (cx - half, cy - half, cx + half, cy + half))


class Script:
    """Stands in for RemoteDetector, and records the state at every cycle."""

    instance = None

    def __init__(self, url):
        self.url = url
        Script.instance = self

    def classes(self):
        return list(COCO_CLASSES)

    def detect(self, frame, labels, conf):
        self.calls += 1
        snap = self.state.snapshot()
        self.trace.append((snap["surveying"], self.state.goal is None))
        if self.calls >= self.limit:
            self.stop.set()
        return self.frames(self.calls)


@pytest.fixture
def run(monkeypatch):
    """Run the vision thread over a scripted detector, returning its trace."""

    def go(frames, cycles=120, dwell=0.25, survey=0.25):
        monkeypatch.setattr(main, "RemoteDetector", Script)
        monkeypatch.setattr(main, "DETECT_HZ", 200.0)
        monkeypatch.setattr(main, "DWELL", dwell)
        monkeypatch.setattr(main, "SURVEY_FOR", survey)

        state, stop = State(), threading.Event()
        Script.calls, Script.limit = 0, cycles
        Script.state, Script.stop, Script.trace = state, stop, []
        Script.frames = staticmethod(frames)

        Tracker()._track_forever(FakeMini(), state, stop)
        return state, Script.instance.trace

    return go


def centred(_):
    """One cat, dead centre, forever."""
    return [box(W / 2, H / 2)]


class TestDwell:
    def test_a_survey_eventually_starts(self, run):
        _, trace = run(centred)
        assert any(surveying for surveying, _ in trace), "it must look away"

    def test_it_watches_before_it_looks_away(self, run):
        _, trace = run(centred)
        assert not trace[0][0], "the first thing it does is watch"


def scans_away(_):
    """The cat, until the head turns away to look around and loses sight of it.

    Nothing here moves the head — the control loop is not running — so the
    frame has to model what a turning head would see.
    """
    return [] if Script.state.surveying else [box(W / 2, H / 2)]


def elsewhere(x):
    """Pixel column far enough off centre to be a genuinely different subject."""
    return W / 2 + FakeCamera.K[0, 0] * np.tan(np.deg2rad(main.AVOID_DEGREES + 5))


class TestFruitlessSurvey:
    """The bug from hardware: it kept snapping back to the same cat."""

    def test_the_lone_target_is_forgotten_not_returned_to(self, run):
        # A survey that finds nobody must drop the aim point. Left in place it
        # is still inside LOST_AFTER, so the control loop drives back to where
        # the cat was instead of carrying on looking.
        _, trace = run(scans_away)

        surveyed = [i for i, (surveying, _) in enumerate(trace) if surveying]
        assert surveyed, "the dwell must expire within the run"
        after = trace[surveyed[-1] + 1 :]
        assert after, "the survey must end within the run"
        assert after[0][1], "the aim point must be dropped when a survey ends"

    def test_a_target_still_in_view_is_picked_up_again(self, run):
        # The other half of it: coming back to the only subject in the room is
        # right, as long as it happens by seeing it rather than by remembering.
        state, _ = run(centred)
        assert state.goal is not None
        assert state.label == "cat"

    def test_the_same_subject_does_not_end_the_survey(self, run):
        # Seeing the cat again is not finding someone else.
        _, trace = run(centred)
        runs, current = [], 0
        for surveying, _ in trace:
            current = current + 1 if surveying else 0
            runs.append(current)
        assert max(runs) > 3, "the avoid gate must keep the survey running"


class TestFindingSomeoneElse:
    def test_a_different_subject_ends_the_survey_early(self, run):
        def and_a_person(_):
            here = [] if Script.state.surveying else [box(W / 2, H / 2)]
            return here + [box(elsewhere(0), H / 2, size=80, label="person")]

        state, _ = run(and_a_person)
        assert state.label == "person", "it must switch to the new subject"

    def test_a_subject_beside_the_old_one_is_not_a_new_one(self, run):
        # Within AVOID_DEGREES it is treated as the target we just left, which
        # is what stops the survey ending on the first frame.
        def alongside(_):
            return [box(W / 2, H / 2), box(W / 2 + 40, H / 2, size=80, label="dog")]

        _, trace = run(alongside)
        assert any(surveying for surveying, _ in trace), "the survey must run"
