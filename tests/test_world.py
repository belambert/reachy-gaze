import base64
import io
import math

import numpy as np
import pytest
from PIL import Image

from tracker.detector import Detection
from tracker.world import THUMB_SIZE, WorldModel, thumbnail, yaw_pitch


def det(label):
    return Detection(label=label, conf=0.9, box=(0.0, 0.0, 10.0, 10.0))


def toward(yaw, pitch=0.0):
    """Unit world direction at `yaw`, `pitch` degrees."""
    y, p = math.radians(yaw), math.radians(pitch)
    return (math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p))


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def world(clock):
    return WorldModel(match_angle=math.radians(15), forget_after=60.0, time_fn=clock)


def test_starts_empty(world):
    assert world.objects() == []


def test_a_sighting_becomes_an_object(world):
    world.observe([det("cat")], [toward(30)])
    [obj] = world.snapshot()
    assert obj["label"] == "cat"
    assert obj["yaw"] == pytest.approx(30.0)
    assert obj["age"] == 0.0


def test_age_counts_up_while_unseen(world, clock):
    world.observe([det("cat")], [toward(30)])
    clock.t += 7.5
    assert world.snapshot()[0]["age"] == 7.5


def test_a_nearby_sighting_updates_the_same_object(world, clock):
    world.observe([det("cat")], [toward(30)])
    clock.t += 5
    world.observe([det("cat")], [toward(38)])
    [obj] = world.snapshot()
    assert obj["yaw"] == pytest.approx(38.0)
    assert obj["age"] == 0.0


def test_a_distant_sighting_is_a_second_object(world):
    world.observe([det("cat")], [toward(30)])
    world.observe([det("cat")], [toward(-40)])
    assert len(world.objects()) == 2


def test_a_different_label_is_a_different_object(world):
    world.observe([det("cat")], [toward(30)])
    world.observe([det("dog")], [toward(30)])
    assert sorted(o.label for o in world.objects()) == ["cat", "dog"]


def test_two_of_a_kind_side_by_side_keep_their_own_entries(world):
    world.observe([det("cat"), det("cat")], [toward(0), toward(10)])
    before = {o.id: yaw_pitch(o.direction)[0] for o in world.objects()}

    # Listed in the other order and both nudged: each still matches its own.
    world.observe([det("cat"), det("cat")], [toward(12), toward(2)])
    after = {o.id: yaw_pitch(o.direction)[0] for o in world.objects()}
    assert after.keys() == before.keys()
    for id, yaw in before.items():
        assert after[id] == pytest.approx(yaw + 2.0)


def test_objects_are_forgotten_after_the_window(world, clock):
    world.observe([det("cat")], [toward(30)])
    clock.t += 60.0
    assert len(world.objects()) == 1
    clock.t += 0.1
    assert world.objects() == []


def test_most_recently_seen_first(world, clock):
    world.observe([det("cat")], [toward(30)])
    clock.t += 1
    world.observe([det("dog")], [toward(-30)])
    assert [o.label for o in world.objects()] == ["dog", "cat"]


def test_directions_are_absolute_not_in_frame(world):
    world.observe([det("person")], [toward(-20, 10)])
    [obj] = world.snapshot()
    assert obj["yaw"] == pytest.approx(-20.0)
    assert obj["pitch"] == pytest.approx(10.0)
    assert math.hypot(*obj["direction"]) == pytest.approx(1.0, abs=1e-3)


def test_same_yaw_different_height_are_distinct_objects(world):
    # One subject above another on the same bearing: a yaw alone would merge them.
    world.observe([det("cat"), det("cat")], [toward(0, -30), toward(0, 30)])
    assert len(world.objects()) == 2


def test_observe_names_each_sighting_consistently(world):
    first = world.observe([det("cat"), det("dog")], [toward(0), toward(40)])
    again = world.observe([det("dog"), det("cat")], [toward(41), toward(1)])
    assert again == first[::-1]


class TestLook:
    """Where the head is aiming: what is in focus, and for how long."""

    def test_never_watched_by_default(self, world):
        [id] = world.observe([det("cat")], [toward(0)])
        assert not world.dwelt_within(id, 60.0)
        [obj] = world.snapshot()
        assert obj["dwelt_ago"] is None and not obj["focused"]

    def test_looking_at_an_object_watches_and_focuses_it(self, world):
        [id] = world.observe([det("cat")], [toward(0)])
        world.look(toward(5))
        assert world.dwelt_within(id, 60.0)
        assert world.snapshot()[0]["focused"]

    def test_only_what_lies_that_way_is_in_focus(self, world):
        cat, dog = world.observe([det("cat"), det("dog")], [toward(0), toward(40)])
        world.look(toward(0))
        assert world.dwelt_within(cat, 60.0)
        assert not world.dwelt_within(dog, 60.0)
        assert {o["label"]: o["focused"] for o in world.snapshot()} == {
            "cat": True,
            "dog": False,
        }

    def test_everything_that_way_is_in_focus_not_just_one(self, world):
        # Bored of a direction means bored of everything over there.
        a, b = world.observe([det("cat"), det("dog")], [toward(0), toward(12)])
        world.look(toward(5))
        assert world.dwelt_within(a, 60.0) and world.dwelt_within(b, 60.0)

    def test_in_focus_counts_as_seen_through_a_miss(self, world, clock):
        world.observe([det("cat")], [toward(0)])
        clock.t += 5.0
        world.observe([], [])  # the detector missed it; the head still aims there
        world.look(toward(0))
        [obj] = world.snapshot()
        assert obj["age"] == 0.0 and obj["dwelt_ago"] == 0.0 and obj["focused"]

    def test_both_ages_count_up_once_the_head_looks_away(self, world, clock):
        world.observe([det("cat")], [toward(0)])
        world.look(toward(0))
        clock.t += 4.0
        world.look(None)  # scanning
        world.observe([det("cat")], [toward(0)])  # glimpsed in passing
        clock.t += 2.0
        [obj] = world.snapshot()
        assert obj["age"] == 2.0 and obj["dwelt_ago"] == 6.0
        assert not obj["focused"]

    def test_watching_is_remembered_for_the_window(self, world, clock):
        [id] = world.observe([det("cat")], [toward(0)])
        world.look(toward(0))
        world.look(None)

        clock.t += 60.0
        assert world.dwelt_within(id, 60.0)
        clock.t += 0.1
        assert not world.dwelt_within(id, 60.0)


class TestFocusedSince:
    """How long the objects in focus have been watched: what boredom counts."""

    def test_nothing_in_focus_while_scanning(self, world):
        world.observe([det("cat")], [toward(0)])
        world.look(None)
        assert world.focused_since is None

    def test_no_object_there_no_clock(self, world):
        world.observe([det("cat")], [toward(0)])
        world.look(toward(60))
        assert world.focused_since is None

    def test_a_stretch_begins_when_an_object_comes_into_focus(self, world, clock):
        world.observe([det("cat")], [toward(0)])
        world.look(toward(0))
        assert world.focused_since == clock.t

    def test_the_head_can_wander_while_the_object_stays_in_focus(self, world, clock):
        start = clock.t
        world.observe([det("cat")], [toward(0)])
        for yaw in (0, 8, -12, 19, 3):
            world.look(toward(yaw))
            clock.t += 5.0
        assert world.focused_since == start

    def test_an_intermittent_target_keeps_its_stretch(self, world, clock):
        # The head holds its aim while the detector keeps missing the cat.
        start = clock.t
        world.observe([det("cat")], [toward(0)])
        world.look(toward(0))
        for _ in range(25):
            clock.t += 1.0
            world.observe([], [])
            world.look(toward(0))
        assert world.focused_since == start

    def test_a_moving_object_keeps_its_stretch_however_far_it_goes(self, world, clock):
        # Followed across 60°: the clock is the object's, not the direction's.
        start = clock.t
        for yaw in range(0, 61, 5):
            world.observe([det("person")], [toward(yaw)])
            world.look(toward(yaw))
            clock.t += 1.0
        assert world.focused_since == start
        assert len(world.objects()) == 1

    def test_looking_away_ends_the_stretch(self, world, clock):
        world.observe([det("cat")], [toward(0)])
        world.look(toward(0))
        clock.t += 10.0
        world.look(None)
        world.look(toward(0))
        assert world.focused_since == clock.t

    def test_the_longest_stretch_counts(self, world, clock):
        start = clock.t
        world.observe([det("cat")], [toward(0)])
        world.look(toward(0))
        clock.t += 10.0
        world.observe([det("dog")], [toward(10)])  # joins it in focus later
        world.look(toward(0))
        assert world.focused_since == start

    def test_the_snapshot_says_how_long_each_has_been_in_focus(self, world, clock):
        world.observe([det("cat"), det("dog")], [toward(0), toward(40)])
        world.look(toward(0))
        clock.t += 7.5
        world.look(toward(0))
        got = {o["label"]: o["focused_for"] for o in world.snapshot()}
        assert got == {"cat": 7.5, "dog": None}

    def test_only_objects_in_focus_count(self, world, clock):
        world.observe([det("cat")], [toward(0)])
        world.look(toward(0))
        clock.t += 10.0
        world.look(toward(40))  # turned away from it
        assert world.focused_since is None
        assert world.snapshot()[0]["focused"] is False


class TestThumbnail:
    def decode(self, uri):
        assert uri.startswith("data:image/jpeg;base64,")
        return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))

    @pytest.mark.parametrize(
        "box", [(10, 10, 110, 60), (200, 0, 240, 470), (600, 400, 900, 900)]
    )
    def test_every_box_shape_gives_the_same_size(self, box):
        frame = np.zeros((480, 640, 3), np.uint8)
        img = self.decode(thumbnail(frame, Detection("cat", 0.9, box)))
        assert img.size == THUMB_SIZE

    def test_crops_the_box_not_the_frame(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[100:200, 300:400] = (0, 0, 255)  # a red square, in BGR
        img = self.decode(thumbnail(frame, Detection("cat", 0.9, (300, 100, 400, 200))))
        r, g, b = img.convert("RGB").getpixel((24, 18))
        assert r > 200 and g < 60 and b < 60, "colour channels come out as RGB"

    def test_a_degenerate_box_does_not_crash(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        img = self.decode(thumbnail(frame, Detection("cat", 0.9, (50, 50, 50, 50))))
        assert img.size == THUMB_SIZE


def test_no_picture_until_photographed(world):
    world.observe([det("cat")], [toward(0)])
    world.look(toward(0))
    assert world.snapshot()[0]["thumb"] is None


def test_the_latest_photograph_is_kept(world):
    [id] = world.observe([det("cat")], [toward(0)])
    world.photograph(id, "data:a")
    world.photograph(id, "data:b")
    assert world.snapshot()[0]["thumb"] == "data:b"


def test_the_picture_outlasts_the_look(world):
    [id] = world.observe([det("cat")], [toward(0)])
    world.photograph(id, "data:a")
    world.look(None)
    world.observe([det("cat")], [toward(1)])
    assert world.snapshot()[0]["thumb"] == "data:a"
