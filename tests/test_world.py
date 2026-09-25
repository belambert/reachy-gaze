import math

import pytest

from tracker.detector import Detection
from tracker.world import WorldModel, yaw_pitch


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
