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


class TestDwell:
    def test_never_dwelt_by_default(self, world):
        [id] = world.observe([det("cat")], [toward(0)])
        assert not world.dwelt_within(id, 60.0)
        assert world.snapshot()[0]["dwelt_ago"] is None

    def test_dwelling_is_remembered_for_the_window(self, world, clock):
        [id] = world.observe([det("cat")], [toward(0)])
        world.dwell(id)

        clock.t += 30.0
        assert world.dwelt_within(id, 60.0)
        assert world.snapshot()[0]["dwelt_ago"] == 30.0

        clock.t += 30.1
        assert not world.dwelt_within(id, 60.0)

    def test_the_dwell_follows_the_object_as_it_moves(self, world):
        [id] = world.observe([det("cat")], [toward(0)])
        world.dwell(id)
        [moved] = world.observe([det("cat")], [toward(10)])
        assert moved == id and world.dwelt_within(id, 60.0)

    def test_only_the_object_dwelt_on_is_marked(self, world):
        cat, dog = world.observe([det("cat"), det("dog")], [toward(0), toward(40)])
        world.dwell(cat)
        assert world.dwelt_within(cat, 60.0)
        assert not world.dwelt_within(dog, 60.0)


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


def test_no_picture_until_locked_on(world):
    world.observe([det("cat")], [toward(0)])
    assert world.snapshot()[0]["thumb"] is None


def test_the_picture_comes_from_the_latest_lock(world):
    [id] = world.observe([det("cat")], [toward(0)])
    world.dwell(id, "data:a")
    world.observe([det("cat")], [toward(1)])
    world.dwell(id, "data:b")
    assert world.snapshot()[0]["thumb"] == "data:b"


def test_the_picture_outlasts_the_lock(world):
    # Seen again, but not locked on: the picture from the lock stays.
    [id] = world.observe([det("cat")], [toward(0)])
    world.dwell(id, "data:a")
    world.observe([det("cat")], [toward(1)])
    assert world.snapshot()[0]["thumb"] == "data:a"


def test_dwelling_without_a_picture_keeps_the_old_one(world):
    [id] = world.observe([det("cat")], [toward(0)])
    world.dwell(id, "data:a")
    world.dwell(id)
    assert world.snapshot()[0]["thumb"] == "data:a"
