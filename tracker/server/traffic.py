"""Per-client request accounting for the detection server.

A line per request would be ~40k lines an hour at the robot's frame rate, which
amounts to the same thing as no logging at all. This records arrivals,
departures and a periodic summary instead, so the log answers "is the robot
reaching me?" without drowning it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

SUMMARY_EVERY = 10.0  # seconds between traffic summaries
IDLE_AFTER = 5.0  # silence before a client counts as gone


@dataclass
class _Client:
    since: float
    last: float = 0.0
    total: int = 0
    requests: int = 0
    latency: float = 0.0
    detections: int = 0


class Traffic:
    """Counters keyed by client address, summarised on demand.

    Returns log lines rather than emitting them, which keeps the accounting
    testable and the logging at the edge.
    """

    def __init__(
        self,
        idle_after: float = IDLE_AFTER,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Track clients, treating silence longer than `idle_after` as gone."""
        self.idle_after = idle_after
        self.clock = clock
        self.clients: dict[str, _Client] = {}

    def record(self, ip: str, seconds: float) -> str | None:
        """Count one request; returns a line to log if this client is new."""
        now = self.clock()
        new = ip not in self.clients
        if new:
            self.clients[ip] = _Client(since=now)

        client = self.clients[ip]
        client.total += 1
        client.requests += 1
        client.latency += seconds
        client.last = now
        return f"first contact from {ip}" if new else None

    def note_detections(self, ip: str, count: int) -> None:
        """Attribute detections to a client, for the det/req figure."""
        client = self.clients.get(ip)
        if client is not None:
            client.detections += count

    def summarise(self) -> list[str]:
        """Describe activity since the last call, and who has gone quiet since."""
        now = self.clock()
        lines = []

        for ip, client in list(self.clients.items()):
            if client.requests:
                window = max(now - client.since, 1e-9)
                lines.append(
                    f"{ip}: {client.requests} req in {window:.0f}s "
                    f"({client.requests / window:.1f}/s), "
                    f"{1000 * client.latency / client.requests:.0f} ms avg, "
                    f"{client.detections / client.requests:.1f} det/req"
                )
                client.requests = 0
                client.latency = 0.0
                client.detections = 0
                client.since = now
            elif now - client.last > self.idle_after:
                lines.append(f"{ip} went quiet after {client.total} requests")
                del self.clients[ip]

        return lines
