"""Day 2-B: tiny event-level bicycle simulator, standard library only.

Run: python day2_minimal_event_simulator.py
All trips are SYNTHETIC.  No inference about historical unmet demand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import heapq
from collections import deque


@dataclass(frozen=True)
class Trip:
    ride_id: str
    start: str
    end: str
    origin: str
    destination: str


@dataclass
class SimResult:
    initial_total: int
    inventory: dict[str, int]
    capacity: dict[str, int]
    in_transit: set[str] = field(default_factory=set)
    waiting: dict[str, deque[str]] = field(default_factory=dict)
    accepted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    wait_occurrences: list[str] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)


def minutes(s: str) -> int:
    return datetime.strptime(s, '%H:%M').hour * 60 + datetime.strptime(s, '%H:%M').minute


def run_simulation(capacity: dict[str, int], initial: dict[str, int], trips: list[Trip], *, verbose: bool = True) -> SimResult:
    if set(capacity) != set(initial):
        raise ValueError('Capacity and inventory station sets must match')
    if any(not isinstance(k, int) or k <= 0 for k in capacity.values()):
        raise ValueError('Capacities must be positive integers')
    if any(not isinstance(initial[s], int) or not 0 <= initial[s] <= capacity[s] for s in capacity):
        raise ValueError('Initial stocks must be integers within capacities')
    if len({tr.ride_id for tr in trips}) != len(trips):
        raise ValueError('Trip ids must be unique')

    r = SimResult(initial_total=sum(initial.values()), inventory=dict(initial), capacity=dict(capacity),
                  waiting={s: deque() for s in capacity})
    events = []
    seq = 0
    for tr in trips:
        if tr.origin not in capacity or tr.destination not in capacity:
            raise ValueError(f'Unknown station in trip {tr.ride_id}')
        if minutes(tr.end) <= minutes(tr.start):
            raise ValueError(f'Non-positive trip duration in {tr.ride_id}')
        # Priority 1 = start. Arrivals have priority 0 (earlier at identical time).
        heapq.heappush(events, (minutes(tr.start), 1, seq, 'START', tr))
        seq += 1

    def check():
        for s in capacity:
            assert 0 <= r.inventory[s] <= capacity[s], f'Capacity violated: {s}'
        waiting_n = sum(len(q) for q in r.waiting.values())
        assert sum(r.inventory.values()) + len(r.in_transit) + waiting_n == r.initial_total, 'Vehicle conservation violated'
        assert len(r.completed) + len(r.in_transit) + waiting_n == len(r.accepted), 'Accepted trip accounting violated'

    def log(t, event, ride_id, note=''):
        check()
        waiting_n = sum(len(q) for q in r.waiting.values())
        record = {'time': f'{t//60:02d}:{t%60:02d}', 'event': event, 'trip': ride_id,
                  'A': r.inventory.get('A'), 'B': r.inventory.get('B'), 'C': r.inventory.get('C'),
                  'stocks': dict(r.inventory), 'in_transit': len(r.in_transit), 'waiting': waiting_n,
                  'total': sum(r.inventory.values())+len(r.in_transit)+waiting_n, 'note': note}
        r.records.append(record)
        if verbose:
            stocks = ' '.join(f'{s}={r.inventory[s]}' for s in sorted(capacity))
            print(f"{record['time']} {event:<16} {ride_id:<4} | {stocks} | H={record['in_transit']} W={waiting_n} Total={record['total']} {note}")

    def dock_waiting(station: str, t: int):
        while r.waiting[station] and r.inventory[station] < capacity[station]:
            ride_id = r.waiting[station].popleft()
            r.inventory[station] += 1
            r.completed.append(ride_id)
            log(t, 'WAIT_DOCKED', ride_id, f'into {station}')

    check()
    while events:
        t, priority, order, kind, tr = heapq.heappop(events)
        if kind == 'START':
            if r.inventory[tr.origin] == 0:
                r.rejected.append(tr.ride_id)
                log(t, 'REJECTED_START', tr.ride_id, f'empty at {tr.origin}; no arrival scheduled')
                continue
            r.inventory[tr.origin] -= 1
            r.in_transit.add(tr.ride_id)
            r.accepted.append(tr.ride_id)
            log(t, 'STARTED', tr.ride_id, f'{tr.origin} -> {tr.destination}')
            dock_waiting(tr.origin, t)  # freeing a dock lets the earliest waiting rider return
            heapq.heappush(events, (minutes(tr.end), 0, order, 'ARRIVAL', tr))
        else:
            assert tr.ride_id in r.in_transit
            r.in_transit.remove(tr.ride_id)
            if r.inventory[tr.destination] < capacity[tr.destination]:
                r.inventory[tr.destination] += 1
                r.completed.append(tr.ride_id)
                log(t, 'ARRIVED_DOCKED', tr.ride_id, f'at {tr.destination}')
            else:
                r.waiting[tr.destination].append(tr.ride_id)
                r.wait_occurrences.append(tr.ride_id)
                log(t, 'ARRIVED_WAIT', tr.ride_id, f'{tr.destination} full')

    check()
    if verbose:
        print(f'\nSummary: accepted={len(r.accepted)}, rejected={len(r.rejected)}, '
              f'completed={len(r.completed)}, waiting={sum(map(len, r.waiting.values()))}, total={r.initial_total}')
    return r


def test_eight_trips():
    trips = [
        Trip('R1', '08:00', '08:12', 'A', 'B'),
        Trip('R2', '08:01', '08:18', 'A', 'B'),
        Trip('R3', '08:02', '08:20', 'A', 'B'),
        Trip('R4', '08:03', '08:25', 'A', 'B'),
        Trip('R5', '08:05', '08:16', 'B', 'C'),
        Trip('R6', '08:06', '08:22', 'C', 'A'),
        Trip('R7', '08:08', '08:28', 'C', 'A'),
        Trip('R8', '08:10', '08:30', 'B', 'A'),
    ]
    r = run_simulation({'A': 5, 'B': 4, 'C': 3}, {'A': 3, 'B': 1, 'C': 2}, trips)
    assert r.rejected == ['R4', 'R8'], r.rejected
    assert len(r.accepted) == 6 and len(r.completed) == 6
    assert r.inventory == {'A': 2, 'B': 3, 'C': 1}, r.inventory
    assert not r.in_transit and not any(r.waiting.values())
    assert not r.wait_occurrences
    print('PASS: eight-trip scenario; two starts rejected, six completed; conserved every event.')


def test_full_dock():
    # Initial A=2, B=1 (B full).  R1 leaves B, R2 fills its dock, R3 must wait.
    # R4 later departs B and creates space for R3 to finish docking.
    trips = [
        Trip('F1', '08:00', '08:10', 'B', 'A'),
        Trip('F2', '08:01', '08:05', 'A', 'B'),
        Trip('F3', '08:02', '08:06', 'A', 'B'),
        Trip('F4', '08:07', '08:12', 'B', 'A'),
    ]
    r = run_simulation({'A': 3, 'B': 1}, {'A': 2, 'B': 1}, trips)
    assert r.wait_occurrences == ['F3'], r.wait_occurrences
    assert len(r.accepted) == len(r.completed) == 4
    assert r.inventory == {'A': 2, 'B': 1}, r.inventory
    assert not r.in_transit and not any(r.waiting.values())
    print('PASS: full-dock scenario; F3 waited, then docked once space opened.')


if __name__ == '__main__':
    print('=== Test 1: eight hypothetical trips ===')
    test_eight_trips()
    print('\n=== Test 2: full station, wait then dock ===')
    test_full_dock()
    print('\nALL TESTS PASSED')
