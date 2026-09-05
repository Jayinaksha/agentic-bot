#!/usr/bin/env python3
"""
Cross-floor route planning.

Nav2 plans within one occupancy grid. A two-storey house is two grids joined by
a staircase, so "go to the bedroom" from the kitchen is not one Nav2 goal - it
is a sequence:

    drive to the foot of the stairs   (Nav2, floor 0)
    climb                             (climb_fsm, no Nav2)
    drive to the bedroom              (Nav2, floor 1)

This module turns a (from_floor, to_floor, goal_pose) request into that list of
legs. It is deliberately ROS-free so the sequencing logic can be tested without
a simulator, and so both the MCP tool layer and any ROS node can share it.

The floor graph is small - a house has one or two staircases - so the search is
a plain BFS over floors. There is no need for anything heavier, and BFS gives
the fewest transitions, which is what you want: every stair transition costs the
robot its odometry and forces a relocalisation.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

LEG_DRIVE = 'drive'
LEG_CLIMB = 'climb'
LEG_DOCK = 'dock'


@dataclass
class Leg:
    """One executable step of a route."""

    kind: str                       # LEG_DRIVE | LEG_CLIMB | LEG_DOCK
    floor: int
    description: str
    x: Optional[float] = None
    y: Optional[float] = None
    yaw: float = 0.0
    to_floor: Optional[int] = None  # LEG_CLIMB only
    dock_target: Optional[str] = None
    dock_bearing: Optional[float] = None

    def as_dict(self) -> Dict:
        out = {'kind': self.kind, 'floor': self.floor,
               'description': self.description}
        if self.kind in (LEG_DRIVE,):
            out.update({'x': self.x, 'y': self.y, 'yaw': self.yaw})
        if self.kind == LEG_CLIMB:
            out.update({'x': self.x, 'y': self.y, 'yaw': self.yaw,
                        'to_floor': self.to_floor})
        if self.kind == LEG_DOCK:
            out.update({'target': self.dock_target, 'bearing': self.dock_bearing})
        return out


@dataclass
class Transition:
    """A staircase or ramp joining two floors."""

    from_floor: int
    to_floor: int
    foot: Tuple[float, float]       # approach pose on from_floor
    head: Tuple[float, float]       # arrival pose on to_floor
    heading: float = 0.0            # direction of travel, radians
    kind: str = 'stairs'            # 'stairs' | 'ramp'

    def reversed(self) -> 'Transition':
        import math
        return Transition(
            from_floor=self.to_floor,
            to_floor=self.from_floor,
            foot=self.head,
            head=self.foot,
            heading=self.heading + math.pi,
            kind=self.kind,
        )


@dataclass
class FloorGraph:
    """Floors and the transitions between them."""

    floors: Sequence[int]
    transitions: List[Transition] = field(default_factory=list)
    mapped: Dict[int, bool] = field(default_factory=dict)

    def neighbours(self, floor: int) -> List[Transition]:
        """Every transition leaving `floor`, in either stored direction."""
        out = []
        for t in self.transitions:
            if t.from_floor == floor:
                out.append(t)
            elif t.to_floor == floor:
                out.append(t.reversed())
        return out

    def is_mapped(self, floor: int) -> bool:
        return self.mapped.get(floor, True)


class RouteError(Exception):
    """Raised when no usable route exists. Carries a human-readable reason."""


def find_floor_path(graph: FloorGraph, start: int,
                    goal: int) -> List[Transition]:
    """Fewest-transition path between two floors.

    Fewest, not shortest in metres: every transition costs a relocalisation and
    a window of dead reckoning, which dominates any plausible saving in floor
    distance.
    """
    if start == goal:
        return []
    if start not in graph.floors:
        raise RouteError(f'unknown starting floor {start}')
    if goal not in graph.floors:
        raise RouteError(f'unknown destination floor {goal}')
    if not graph.is_mapped(goal):
        raise RouteError(
            f'floor {goal} has never been mapped; map it before navigating '
            f'there (ros2 launch r2d2_localization slam.launch.py floor:={goal})')

    queue = deque([(start, [])])
    seen = {start}
    while queue:
        floor, path = queue.popleft()
        for t in graph.neighbours(floor):
            nxt = t.to_floor
            if nxt in seen:
                continue
            if not graph.is_mapped(nxt) and nxt != goal:
                continue                     # cannot route *through* a blank floor
            new_path = path + [t]
            if nxt == goal:
                return new_path
            seen.add(nxt)
            queue.append((nxt, new_path))

    raise RouteError(
        f'no staircase or ramp connects floor {start} to floor {goal}; '
        f'known transitions: '
        f'{[(t.from_floor, t.to_floor) for t in graph.transitions] or "none"}')


def plan_route(graph: FloorGraph, current_floor: int, goal_floor: int,
               goal_x: float, goal_y: float, goal_yaw: float = 0.0,
               goal_description: str = 'goal',
               dock_target: Optional[str] = None,
               dock_bearing: float = 0.0) -> List[Leg]:
    """Full leg list from where the robot is to where it should end up.

    A same-floor route is a single drive leg (plus an optional dock). A
    cross-floor route interleaves a drive to each staircase foot with a climb.
    """
    transitions = find_floor_path(graph, current_floor, goal_floor)

    legs: List[Leg] = []
    floor = current_floor
    for t in transitions:
        legs.append(Leg(
            kind=LEG_DRIVE, floor=floor,
            x=t.foot[0], y=t.foot[1], yaw=t.heading,
            description=f'drive to the foot of the {t.kind} on floor {floor}',
        ))
        legs.append(Leg(
            kind=LEG_CLIMB, floor=floor, to_floor=t.to_floor,
            x=t.head[0], y=t.head[1], yaw=t.heading,
            description=(f'{"climb" if t.to_floor > floor else "descend"} the '
                         f'{t.kind} from floor {floor} to floor {t.to_floor}'),
        ))
        floor = t.to_floor

    legs.append(Leg(
        kind=LEG_DRIVE, floor=goal_floor,
        x=goal_x, y=goal_y, yaw=goal_yaw,
        description=f'drive to {goal_description} on floor {goal_floor}',
    ))

    if dock_target is not None:
        legs.append(Leg(
            kind=LEG_DOCK, floor=goal_floor,
            dock_target=dock_target, dock_bearing=dock_bearing,
            description=f'servo onto {dock_target} for a precise stop',
        ))

    return legs


def describe_route(legs: Sequence[Leg]) -> str:
    """One-line-per-leg summary, for logs and for the agent's tool output."""
    if not legs:
        return 'already at the goal'
    return '\n'.join(f'{i + 1}. {leg.description}' for i, leg in enumerate(legs))


def graph_from_payload(payload: Dict) -> FloorGraph:
    """Build a FloorGraph from floor_manager's /floor/graph JSON."""
    floors = [f['index'] for f in payload.get('floors', [])]
    mapped = {f['index']: bool(f.get('mapped', True))
              for f in payload.get('floors', [])}
    transitions = []
    for t in payload.get('transitions', []):
        transitions.append(Transition(
            from_floor=int(t['from_floor']),
            to_floor=int(t['to_floor']),
            foot=(float(t['foot']['x']), float(t['foot']['y'])),
            head=(float(t['head']['x']), float(t['head']['y'])),
            heading=float(t.get('heading', 0.0)),
            kind=t.get('kind', 'stairs'),
        ))
    return FloorGraph(floors=floors, transitions=transitions, mapped=mapped)
