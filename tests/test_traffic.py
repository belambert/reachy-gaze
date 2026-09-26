import pytest

from reachy_gaze.server.traffic import Traffic


class Clock:
    """A hand-cranked clock, so summaries are deterministic."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def traffic(clock):
    return Traffic(idle_after=5.0, clock=clock)


def test_first_contact_is_announced_once(traffic):
    assert traffic.record("10.0.0.5", 0.02) == "first contact from 10.0.0.5"
    assert traffic.record("10.0.0.5", 0.02) is None


def test_each_address_is_announced_separately(traffic):
    assert traffic.record("10.0.0.5", 0.01) is not None
    assert traffic.record("10.0.0.9", 0.01) == "first contact from 10.0.0.9"


def test_summary_reports_rate_latency_and_detections(traffic, clock):
    for _ in range(4):
        traffic.record("10.0.0.5", 0.025)
    traffic.note_detections("10.0.0.5", 4)
    traffic.note_detections("10.0.0.5", 2)

    clock.t = 10.0
    (line,) = traffic.summarise()
    assert "10.0.0.5" in line
    assert "4 req in 10s" in line
    assert "0.4/s" in line
    assert "25 ms avg" in line
    assert "1.5 det/req" in line


def test_quiet_clients_are_reported_then_forgotten(traffic, clock):
    traffic.record("10.0.0.5", 0.01)
    clock.t = 10.0
    traffic.summarise()  # drains the window

    clock.t = 20.0
    assert traffic.summarise() == ["10.0.0.5 went quiet after 1 requests"]
    assert traffic.clients == {}
    assert traffic.summarise() == [], "a departure is reported once, not forever"


def test_a_busy_client_is_never_called_quiet(traffic, clock):
    for tick in range(5):
        clock.t = tick * 10.0
        traffic.record("10.0.0.5", 0.01)
        assert not any("quiet" in line for line in traffic.summarise())


def test_returning_client_is_announced_again(traffic, clock):
    traffic.record("10.0.0.5", 0.01)
    clock.t = 20.0
    traffic.summarise()  # reports the window, still within idle_after
    clock.t = 40.0
    traffic.summarise()  # now gone
    assert traffic.record("10.0.0.5", 0.01) == "first contact from 10.0.0.5"


def test_silence_alone_produces_nothing(traffic, clock):
    clock.t = 100.0
    assert traffic.summarise() == []


def test_detections_for_an_unknown_client_are_ignored(traffic):
    traffic.note_detections("10.0.0.99", 5)  # must not raise or invent a client
    assert traffic.clients == {}
