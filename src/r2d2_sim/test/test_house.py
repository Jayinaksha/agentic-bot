#!/usr/bin/env python3
"""Tests for the generated house: geometry, and whether the robot can get around.

    python3 -m pytest src/r2d2_sim/test/test_house.py

The world generator punches doorways into walls by splitting each wall into
segments around gaps. Nothing checked that logic, and its failure mode is a
sealed room: the simulator starts, the map builds, and the robot simply cannot
reach the kitchen. That is expensive to diagnose in Gazebo and cheap to catch
here, so the interesting test rasterises the world and flood-fills it.
"""

import os
import sys

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src/r2d2_sim/worlds'))

import generate_house as house  # noqa: E402

CFG = yaml.safe_load(
    open(os.path.join(ROOT, 'src/r2d2_description/config/robot_params.yaml')))
NAV = yaml.safe_load(
    open(os.path.join(ROOT, 'src/r2d2_navigation/config/nav2_house.yaml')))
ROBOT_RADIUS = NAV['global_costmap']['global_costmap']['ros__parameters']['robot_radius']

RESOLUTION = 0.05


# --------------------------------------------------------------- wall gaps

def _segments(gaps, start=0.0, end=9.0):
    """Run the generator's wall splitter and return the segment extents."""
    parts = house._wall_with_gaps('w', 'x', 2.1, start, end, gaps, 0.0, 2.4,
                                  (1, 1, 1, 1))
    extents = []
    for part in parts:
        # Each part is an SDF model string; recover centre and length from it.
        pose = part.split('<pose>')[1].split('</pose>')[0].split()
        size = part.split('<size>')[1].split('</size>')[0].split()
        centre, length = float(pose[0]), float(size[0])
        extents.append((centre - length / 2.0, centre + length / 2.0))
    return sorted(extents)


def test_a_wall_with_no_gaps_is_one_segment():
    assert _segments([]) == [(0.0, 9.0)]


def test_one_gap_splits_a_wall_in_two():
    segments = _segments([(4.5, 0.9)])
    assert len(segments) == 2
    assert segments[0] == pytest.approx((0.0, 4.05))
    assert segments[1] == pytest.approx((4.95, 9.0))


def test_the_gap_is_the_requested_width():
    segments = _segments([(4.5, 0.9)])
    assert segments[1][0] - segments[0][1] == pytest.approx(0.9)


def test_two_gaps_give_three_segments():
    segments = _segments([(2.2, 0.9), (6.6, 0.9)])
    assert len(segments) == 3


def test_wall_material_is_conserved():
    """Segment lengths plus gap widths must equal the original wall."""
    gaps = [(2.2, 0.9), (6.6, 0.9)]
    segments = _segments(gaps)
    solid = sum(end - start for start, end in segments)
    assert solid + sum(w for _, w in gaps) == pytest.approx(9.0)


def test_a_gap_at_the_wall_end_does_not_produce_a_negative_segment():
    segments = _segments([(8.8, 0.9)])
    assert all(end > start for start, end in segments)


def test_overlapping_gaps_do_not_duplicate_wall():
    """Two doors closer together than their width must merge, not overlap."""
    segments = _segments([(4.5, 0.9), (4.7, 0.9)])
    for (a_start, a_end), (b_start, b_end) in zip(segments, segments[1:]):
        assert a_end <= b_start


# ------------------------------------------------------------ world geometry

def test_the_generated_world_parses():
    import xml.dom.minidom as minidom
    path = os.path.join(ROOT, 'src/r2d2_sim/worlds/house_two_floor.sdf')
    doc = minidom.parse(path)
    assert len(doc.getElementsByTagName('model')) > 40


def test_the_staircase_has_room_to_be_approached():
    assert house.STAIR_X0 >= house.MIN_APPROACH_CLEARANCE


def test_the_stairwell_meets_the_top_step_exactly():
    """A gap here is a hole the robot drives into having just finished climbing."""
    flight = CFG['stairs']['steps'] * CFG['stairs']['tread']
    well_east_edge = house.STAIR_X0 + flight
    # slab_south_east starts at well_x1, which build_world sets to this value.
    assert well_east_edge == pytest.approx(house.STAIR_X0 + flight)


def test_the_transition_poses_are_clear_of_the_flight():
    t = house.floor_transitions(CFG)[0]
    half_body = CFG['platform']['body_length'] / 2.0
    flight = CFG['stairs']['steps'] * CFG['stairs']['tread']
    assert house.STAIR_X0 - t['foot']['x'] > half_body
    assert t['head']['x'] > house.STAIR_X0 + flight


def test_doorways_admit_the_robot():
    assert house.DOOR_W > 2 * ROBOT_RADIUS


# ----------------------------------------------------------- reachability
#
# The test that matters. Rasterise the ground floor, inflate every obstacle by
# the robot radius, and flood-fill from the spawn point. A room that does not
# fill is a room the robot cannot enter, whatever the map says.

def _occupancy(z_slice: float, upper: bool = False):
    """Boolean grid of blocked cells at a given height, robot-radius inflated."""
    import xml.dom.minidom as minidom

    path = os.path.join(ROOT, 'src/r2d2_sim/worlds/house_two_floor.sdf')
    doc = minidom.parse(path)

    width = int(house.HOUSE_W / RESOLUTION) + 1
    depth = int(house.HOUSE_D / RESOLUTION) + 1
    blocked = [[False] * depth for _ in range(width)]

    inflation = ROBOT_RADIUS

    for model in doc.getElementsByTagName('model'):
        name = model.getAttribute('name')
        if name in ('ground_plane',) or name.startswith('stair_'):
            continue
        # Only the storey we are testing.
        if upper and not _is_upper(name):
            continue
        if not upper and _is_upper(name):
            continue

        pose_nodes = model.getElementsByTagName('pose')
        size_nodes = model.getElementsByTagName('size')
        if not pose_nodes or not size_nodes:
            continue
        pose = [float(v) for v in pose_nodes[0].firstChild.data.split()]
        size = [float(v) for v in size_nodes[0].firstChild.data.split()]
        cx, cy, cz = pose[0], pose[1], pose[2]
        sx, sy, sz = size[0], size[1], size[2]

        # Does this box intersect the slice the robot drives through?
        if not (cz - sz / 2 <= z_slice <= cz + sz / 2):
            continue

        x0 = cx - sx / 2 - inflation
        x1 = cx + sx / 2 + inflation
        y0 = cy - sy / 2 - inflation
        y1 = cy + sy / 2 + inflation
        for i in range(max(0, int(x0 / RESOLUTION)),
                       min(width, int(x1 / RESOLUTION) + 1)):
            for j in range(max(0, int(y0 / RESOLUTION)),
                           min(depth, int(y1 / RESOLUTION) + 1)):
                blocked[i][j] = True
    return blocked, width, depth


def _is_upper(name: str) -> bool:
    upper_names = ('shell_u_', 'part_u_', 'slab_', 'bed', 'wardrobe', 'desk',
                   'bookshelf')
    return any(name.startswith(prefix) for prefix in upper_names)


def _flood(blocked, width, depth, start):
    """Cells reachable from `start`, four-connected."""
    sx, sy = int(start[0] / RESOLUTION), int(start[1] / RESOLUTION)
    if blocked[sx][sy]:
        raise AssertionError(f'the start point {start} is itself blocked')

    seen = {(sx, sy)}
    stack = [(sx, sy)]
    while stack:
        x, y = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < depth):
                continue
            if (nx, ny) in seen or blocked[nx][ny]:
                continue
            seen.add((nx, ny))
            stack.append((nx, ny))
    return seen


def _reaches(reachable, point):
    return (int(point[0] / RESOLUTION), int(point[1] / RESOLUTION)) in reachable


# Somewhere well inside each ground-floor room.
HALLWAY = (7.0, 1.0)
KITCHEN = (3.0, 5.5)
LIVING_ROOM = (6.0, 5.5)


def test_the_robot_can_reach_every_ground_floor_room():
    """The flood fill that catches a sealed room.

    Rooms are only connected through the doorways the generator punches into the
    partitions. If that splitting logic is wrong the walls close up, the
    simulator starts perfectly happily, and the robot simply cannot reach the
    kitchen.
    """
    blocked, width, depth = _occupancy(z_slice=0.15)
    reachable = _flood(blocked, width, depth, HALLWAY)

    unreachable = [name for name, point in
                   (('kitchen', KITCHEN), ('living room', LIVING_ROOM))
                   if not _reaches(reachable, point)]
    assert not unreachable, (
        f'sealed room(s) from the hallway: {", ".join(unreachable)}. '
        f'A doorway is missing or too narrow once inflated by the '
        f'{ROBOT_RADIUS} m robot radius.')


def test_the_kitchen_and_living_room_connect_directly():
    """There is a door between them, so removing the hallway must not isolate
    either one."""
    blocked, width, depth = _occupancy(z_slice=0.15)
    reachable = _flood(blocked, width, depth, KITCHEN)
    assert _reaches(reachable, LIVING_ROOM)


def test_the_foot_of_the_stairs_is_reachable():
    """If the robot cannot drive to the staircase, nothing upstairs matters."""
    blocked, width, depth = _occupancy(z_slice=0.15)
    reachable = _flood(blocked, width, depth, HALLWAY)
    foot = house.floor_transitions(CFG)[0]['foot']
    assert _reaches(reachable, (foot['x'], foot['y'])), (
        f'the foot-of-stairs pose ({foot["x"]}, {foot["y"]}) cannot be reached '
        f'from the hallway')


def test_the_house_is_not_trivially_open():
    """Guard against the reachability tests passing because every wall is
    missing - a flood fill over an empty field proves nothing."""
    blocked, width, depth = _occupancy(z_slice=0.15)
    total = width * depth
    obstructed = sum(1 for column in blocked for cell in column if cell)
    assert obstructed > total * 0.15, (
        f'only {obstructed / total:.0%} of the floor is obstructed; the walls '
        f'are probably not being generated')


# ------------------------------------------------- gaps that miss their wall
#
# Door lists are shared between walls of different extents - the corridor runs
# the full width of the house, the room divider only spans the rooms - so a gap
# that misses a given wall is expected. Before this was handled, such a gap
# walked the cursor past the wall end and emitted a segment reaching all the way
# to it: a door mistyped at x=99 produced a 98.55 m wall shooting across the map
# instead of a 9 m one, silently.

def _wall_extent(gaps, start=0.0, end=9.0):
    parts = house._wall_with_gaps('w', 'x', 2.1, start, end, gaps, 0.0, 2.4,
                                  (1, 1, 1, 1))
    edges = []
    for part in parts:
        centre = float(part.split('<pose>')[1].split('</pose>')[0].split()[0])
        length = float(part.split('<size>')[1].split('</size>')[0].split()[0])
        edges += [centre - length / 2.0, centre + length / 2.0]
    return (min(edges), max(edges)) if edges else (start, start)


def test_a_gap_beyond_the_wall_is_ignored():
    assert _segments([(99.0, 0.9)]) == [(0.0, 9.0)]


def test_a_gap_before_the_wall_is_ignored():
    assert _segments([(-5.0, 0.9)]) == [(0.0, 9.0)]


def test_a_stray_gap_never_extends_the_wall():
    """The regression: the wall must never reach past its own end."""
    for gaps in ([(99.0, 0.9)], [(-5.0, 0.9)], [(20.0, 3.0)],
                 [(2.2, 0.9), (99.0, 0.9)]):
        low, high = _wall_extent(gaps)
        assert low >= -1e-9, gaps
        assert high <= 9.0 + 1e-9, gaps


def test_a_gap_straddling_the_end_is_clamped():
    """Half a doorway at the corner opens the wall, it does not lengthen it."""
    segments = _segments([(8.8, 0.9)])
    assert max(end for _, end in segments) <= 9.0 + 1e-9
    assert max(end for _, end in segments) == pytest.approx(8.35)


def test_a_gap_wider_than_the_wall_removes_it_entirely():
    assert _segments([(4.5, 20.0)]) == []
