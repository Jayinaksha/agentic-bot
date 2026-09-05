#!/usr/bin/env python3
"""
Exploration: drive around the current floor and look at things.

Deliberately simple. A full frontier-exploration implementation would be the
right answer for mapping an unknown building, but this robot is working inside a
house it has already mapped - what it is short of is *semantic* coverage, not
geometry. So exploration here means: visit places on this floor that the camera
has not looked at recently, and look at them.

Targets are chosen from the free space in the costmap, biased away from where
the robot has already looked, and each arrival triggers one VLA frame. The
result is a report of what turned up, which the agent can act on.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

# Minimum separation between successive look-points. Below this the camera sees
# largely the same scene twice and the inference call is wasted.
MIN_TARGET_SEPARATION = 1.5
# How far to cast for a candidate target.
CAST_MIN = 1.5
CAST_MAX = 4.0


def choose_target(pose: Tuple[float, float, float],
                  visited: List[Tuple[float, float]],
                  scan_ranges: Optional[List[float]] = None,
                  angle_min: float = -math.pi,
                  angle_increment: float = 0.0087,
                  rng: Optional[random.Random] = None
                  ) -> Optional[Tuple[float, float, float]]:
    """Pick somewhere on this floor worth looking at next.

    Candidates are directions with enough clear LiDAR range to drive into,
    scored by how far they are from everywhere already visited. Returning None
    means every direction is either blocked or already covered.
    """
    rng = rng or random.Random()
    x, y, yaw = pose

    candidates: List[Tuple[float, float, float, float]] = []
    if scan_ranges:
        angle = angle_min
        step = max(1, len(scan_ranges) // 36)      # ~36 directions is plenty
        for i in range(0, len(scan_ranges), step):
            r = scan_ranges[i]
            a = angle + i * angle_increment
            if not math.isfinite(r) or r < CAST_MIN:
                continue
            # Stop short of whatever the beam hit, so the target is in free space.
            reach = min(r - 0.7, CAST_MAX)
            if reach < CAST_MIN:
                continue
            heading = yaw + a
            candidates.append((x + reach * math.cos(heading),
                               y + reach * math.sin(heading),
                               heading, reach))

    if not candidates:
        return None

    def novelty(candidate) -> float:
        cx, cy = candidate[0], candidate[1]
        if not visited:
            return float('inf')
        return min(math.dist((cx, cy), v) for v in visited)

    scored = [(novelty(c), c) for c in candidates]
    scored = [(n, c) for n, c in scored if n >= MIN_TARGET_SEPARATION]
    if not scored:
        return None

    # Take the best few and pick randomly among them, so a robot that gets stuck
    # in a corner does not deterministically retry the same failing target.
    scored.sort(key=lambda item: -item[0])
    _, best = rng.choice(scored[:3])
    return best[0], best[1], best[2]


async def run_exploration(bridge, duration_s: float = 60.0,
                          seed: Optional[int] = None) -> Dict[str, Any]:
    """Drive and look until the budget runs out."""
    rng = random.Random(seed)
    deadline = time.time() + duration_s
    start_floor = bridge.floor

    visited: List[Tuple[float, float]] = [bridge.pose[:2]]
    looks = 0
    arrivals = 0
    failures = 0
    found: Dict[str, int] = {}

    # Look before moving: the robot may already be somewhere interesting.
    first = await bridge.observe(timeout_s=min(45.0, duration_s))
    if first.get('success'):
        looks += 1
        for obj in first.get('objects', []):
            found[obj['label']] = found.get(obj['label'], 0) + 1

    while time.time() < deadline:
        if bridge.terrain.get('estop'):
            return _report(looks, arrivals, failures, found, visited,
                           stopped_early=('the terrain safety monitor latched: '
                                          f'{bridge.terrain.get("estop_reason")}'))
        if bridge.floor != start_floor:
            return _report(looks, arrivals, failures, found, visited,
                           stopped_early='the robot changed floor mid-explore')

        scan = bridge.scan
        target = choose_target(
            bridge.pose, visited,
            list(scan.ranges) if scan else None,
            scan.angle_min if scan else -math.pi,
            scan.angle_increment if scan else 0.0087,
            rng=rng)

        if target is None:
            return _report(looks, arrivals, failures, found, visited,
                           stopped_early=('nowhere new to look from here: every '
                                          'reachable direction has already been '
                                          'covered or is blocked'))

        remaining = deadline - time.time()
        if remaining < 10.0:
            break

        outcome = await bridge.navigate_to(
            target[0], target[1], target[2],
            timeout_s=min(45.0, remaining))

        if not outcome.get('success'):
            failures += 1
            visited.append((target[0], target[1]))   # do not retry it
            if failures >= 4:
                return _report(looks, arrivals, failures, found, visited,
                               stopped_early=('too many unreachable targets; '
                                              'the robot may be boxed in'))
            continue

        arrivals += 1
        visited.append(bridge.pose[:2])

        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        view = await bridge.observe(timeout_s=min(45.0, remaining))
        if view.get('success'):
            looks += 1
            for obj in view.get('objects', []):
                found[obj['label']] = found.get(obj['label'], 0) + 1

        await asyncio.sleep(0.1)

    return _report(looks, arrivals, failures, found, visited)


def _report(looks: int, arrivals: int, failures: int,
            found: Dict[str, int], visited: List[Tuple[float, float]],
            stopped_early: Optional[str] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        'success': looks > 0,
        'places_visited': arrivals,
        'looks_taken': looks,
        'unreachable_targets': failures,
        'objects_seen': dict(sorted(found.items(),
                                    key=lambda kv: -kv[1])),
        'coverage_points': len(visited),
    }
    if stopped_early:
        result['stopped_early'] = stopped_early
    if looks == 0:
        result['reason'] = (stopped_early or
                            'the vision model did not return anything usable')
    return result
