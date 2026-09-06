#!/usr/bin/env python3
"""Unit tests for cross-floor route planning.

    python3 -m pytest src/r2d2_navigation/test/test_route_planner.py
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from r2d2_navigation.route_planner import (  # noqa: E402
    LEG_CLIMB, LEG_DOCK, LEG_DRIVE, FloorGraph, RouteError, Transition,
    describe_route, find_floor_path, graph_from_payload, plan_route)


def two_storey() -> FloorGraph:
    """The house world: one staircase between floors 0 and 1."""
    return FloorGraph(
        floors=[0, 1],
        transitions=[Transition(from_floor=0, to_floor=1,
                                foot=(0.35, 0.55), head=(3.60, 0.55),
                                heading=0.0, kind='stairs')],
        mapped={0: True, 1: True},
    )


def three_storey() -> FloorGraph:
    return FloorGraph(
        floors=[0, 1, 2],
        transitions=[
            Transition(0, 1, foot=(0.35, 0.55), head=(3.60, 0.55)),
            Transition(1, 2, foot=(3.60, 5.0), head=(0.35, 5.0)),
        ],
        mapped={0: True, 1: True, 2: True},
    )


# ------------------------------------------------------------- floor search

def test_same_floor_needs_no_transition():
    assert find_floor_path(two_storey(), 0, 0) == []


def test_one_flight_up():
    path = find_floor_path(two_storey(), 0, 1)
    assert len(path) == 1
    assert (path[0].from_floor, path[0].to_floor) == (0, 1)


def test_one_flight_down_reuses_the_same_staircase():
    """Only the upward transition is stored; descending must reverse it."""
    path = find_floor_path(two_storey(), 1, 0)
    assert len(path) == 1
    assert (path[0].from_floor, path[0].to_floor) == (1, 0)
    assert path[0].foot == (3.60, 0.55)     # the head of the up-flight
    assert path[0].head == (0.35, 0.55)
    assert path[0].heading == pytest.approx(math.pi)


def test_two_flights_are_chained():
    path = find_floor_path(three_storey(), 0, 2)
    assert [(t.from_floor, t.to_floor) for t in path] == [(0, 1), (1, 2)]


def test_descending_two_flights():
    path = find_floor_path(three_storey(), 2, 0)
    assert [(t.from_floor, t.to_floor) for t in path] == [(2, 1), (1, 0)]


def test_unreachable_floor_explains_itself():
    graph = FloorGraph(floors=[0, 1], transitions=[], mapped={0: True, 1: True})
    with pytest.raises(RouteError, match='no staircase or ramp'):
        find_floor_path(graph, 0, 1)


def test_unknown_floor_is_rejected():
    with pytest.raises(RouteError, match='unknown destination floor'):
        find_floor_path(two_storey(), 0, 7)


def test_unmapped_destination_is_refused_with_the_fix():
    """Refusing beats serving an empty grid, which a planner reads as free space."""
    graph = two_storey()
    graph.mapped[1] = False
    with pytest.raises(RouteError, match='never been mapped'):
        find_floor_path(graph, 0, 1)


def test_cannot_route_through_an_unmapped_intermediate_floor():
    graph = three_storey()
    graph.mapped[1] = False
    with pytest.raises(RouteError):
        find_floor_path(graph, 0, 2)


def test_bfs_prefers_fewer_transitions():
    """Two routes to floor 2; the direct one wins because every transition
    costs a relocalisation."""
    graph = FloorGraph(
        floors=[0, 1, 2],
        transitions=[
            Transition(0, 1, foot=(0, 0), head=(1, 1)),
            Transition(1, 2, foot=(2, 2), head=(3, 3)),
            Transition(0, 2, foot=(4, 4), head=(5, 5)),
        ],
        mapped={0: True, 1: True, 2: True},
    )
    assert len(find_floor_path(graph, 0, 2)) == 1


# -------------------------------------------------------------- leg building

def test_same_floor_route_is_a_single_drive():
    legs = plan_route(two_storey(), 0, 0, 7.4, 3.2, goal_description='the sofa')
    assert len(legs) == 1
    assert legs[0].kind == LEG_DRIVE
    assert (legs[0].x, legs[0].y) == (7.4, 3.2)


def test_cross_floor_route_interleaves_drive_and_climb():
    legs = plan_route(two_storey(), 0, 1, 2.0, 5.4, goal_description='the bed')
    assert [leg.kind for leg in legs] == [LEG_DRIVE, LEG_CLIMB, LEG_DRIVE]
    # First leg goes to the foot of the stairs, not straight at the bed.
    assert (legs[0].x, legs[0].y) == (0.35, 0.55)
    assert legs[1].to_floor == 1
    assert (legs[2].x, legs[2].y) == (2.0, 5.4)
    assert legs[2].floor == 1


def test_climb_leg_knows_which_way_it_is_going():
    up = plan_route(two_storey(), 0, 1, 2.0, 5.4)
    down = plan_route(two_storey(), 1, 0, 2.0, 4.8)
    assert 'climb' in up[1].description
    assert 'descend' in down[1].description


def test_three_storey_route_has_two_climbs():
    legs = plan_route(three_storey(), 0, 2, 1.0, 1.0)
    assert [leg.kind for leg in legs].count(LEG_CLIMB) == 2
    assert len(legs) == 5


def test_dock_leg_is_appended_last():
    legs = plan_route(two_storey(), 0, 0, 0.5, 6.4,
                      goal_description='the fridge',
                      dock_target='fridge door', dock_bearing=0.1)
    assert legs[-1].kind == LEG_DOCK
    assert legs[-1].dock_target == 'fridge door'
    assert legs[-1].dock_bearing == pytest.approx(0.1)


def test_every_leg_carries_the_floor_it_runs_on():
    """The executor switches maps per leg, so a wrong floor here drives the
    robot using the other storey's costmap."""
    legs = plan_route(three_storey(), 0, 2, 1.0, 1.0)
    assert [leg.floor for leg in legs] == [0, 0, 1, 1, 2]


def test_leg_serialisation_is_complete_enough_to_execute():
    legs = plan_route(two_storey(), 0, 1, 2.0, 5.4)
    drive = legs[0].as_dict()
    assert {'kind', 'floor', 'x', 'y', 'yaw'} <= set(drive)
    climb = legs[1].as_dict()
    assert climb['to_floor'] == 1


def test_describe_route_lists_every_leg():
    text = describe_route(plan_route(two_storey(), 0, 1, 2.0, 5.4))
    assert text.count('\n') == 2
    assert text.startswith('1. ')


def test_describe_empty_route():
    assert describe_route([]) == 'already at the goal'


# ------------------------------------------------------------ payload parsing

def test_graph_round_trips_from_floor_manager_payload():
    payload = {
        'floors': [
            {'index': 0, 'height': 0.0, 'map': '/m/house_f0.yaml', 'mapped': True},
            {'index': 1, 'height': 1.8, 'map': '/m/house_f1.yaml', 'mapped': False},
        ],
        'transitions': [{
            'from_floor': 0, 'to_floor': 1,
            'foot': {'x': 0.35, 'y': 0.55},
            'head': {'x': 3.6, 'y': 0.55},
            'heading': 0.0,
        }],
    }
    graph = graph_from_payload(payload)
    assert graph.floors == [0, 1]
    assert graph.is_mapped(0) and not graph.is_mapped(1)
    with pytest.raises(RouteError, match='never been mapped'):
        find_floor_path(graph, 0, 1)


# ------------------------------------------------------------ descent marking
#
# The locomotion layer refuses to descend unless explicitly enabled, so a route
# has to say which legs go down. Finding out at the top of a flight is worse
# than finding out before setting off.

def test_an_upward_route_does_not_descend():
    from r2d2_navigation.route_planner import route_descends
    assert not route_descends(plan_route(two_storey(), 0, 1, 2.0, 5.4))


def test_a_downward_route_is_marked():
    from r2d2_navigation.route_planner import route_descends
    legs = plan_route(two_storey(), 1, 0, 2.0, 4.8)
    assert route_descends(legs)
    climb = next(leg for leg in legs if leg.kind == LEG_CLIMB)
    assert climb.descending
    assert 'descend' in climb.description


def test_a_same_floor_route_never_descends():
    from r2d2_navigation.route_planner import route_descends
    assert not route_descends(plan_route(two_storey(), 0, 0, 7.4, 3.2))


def test_a_mixed_route_is_marked_if_any_leg_descends():
    """Two flights up then one down still needs descent enabled."""
    from r2d2_navigation.route_planner import route_descends
    assert route_descends(plan_route(three_storey(), 2, 0, 1.0, 1.0))


def test_the_descending_flag_survives_serialisation():
    legs = plan_route(two_storey(), 1, 0, 2.0, 4.8)
    climb = next(leg for leg in legs if leg.kind == LEG_CLIMB)
    assert climb.as_dict()['descending'] is True
