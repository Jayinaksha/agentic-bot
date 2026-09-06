#!/usr/bin/env python3
"""
Check that the robot fits through the house it is asked to navigate.

    python3 scripts/check_clearances.py
    python3 scripts/check_clearances.py --pose-error 0.10

Why this exists
---------------
The staircase in this world was originally built with 0.20 m of approach room
for a 0.36 m robot - it could never have squared up to it, and nothing in the
code said so. That was found by comparing the robot's dimensions against the
world's. The doorways deserve the same arithmetic, and so does the costmap
tuning that decides whether a planner will even consider using them.

Three things are checked, against the same robot_params.yaml, the generated
world constants and the live Nav2 configuration:

  1. Geometric fit      does the robot physically pass, and with what margin
  2. Margin vs pose      is the margin larger than the localisation error the
                         robot actually has when following a path
  3. Costmap inflation   does a zero-cost lane survive through the opening, or
                         does inflation from both jambs meet in the middle

Point 3 is the one that bites quietly. Inflation is not lethal, so a planner
still finds a route, but if the entire doorway is inflated then every route in
the house runs through high-cost cells: the cost-regulated controller slows to
its floor at every threshold and the planner will take absurd detours to avoid a
door when any alternative exists.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

try:
    import yaml
except ImportError:
    sys.exit('PyYAML required: pip install pyyaml')

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Typical AMCL position error in a well-mapped indoor space with a good scan
# match. This is the number the passage margin has to beat - NOT the goal
# tolerance, which governs where the robot stops, not how well it tracks a path.
DEFAULT_POSE_ERROR = 0.06

PASS, WARN, FAIL = 'PASS', 'WARN', 'FAIL'
_status_rank = {PASS: 0, WARN: 1, FAIL: 2}


def _load(path):
    with open(os.path.join(ROOT, path)) as fh:
        return yaml.safe_load(fh)


def report(status, check, detail, advice=None):
    print(f'  [{status}] {check}')
    print(f'         {detail}')
    if advice:
        print(f'         -> {advice}')
    print()
    return _status_rank[status]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pose-error', type=float, default=DEFAULT_POSE_ERROR,
                        help='assumed localisation error while path-following, '
                             f'metres (default {DEFAULT_POSE_ERROR})')
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(ROOT, 'src/r2d2_sim/worlds'))
    import generate_house as house

    platform = _load('src/r2d2_description/config/robot_params.yaml')['platform']
    nav = _load('src/r2d2_navigation/config/nav2_house.yaml')
    global_cm = nav['global_costmap']['global_costmap']['ros__parameters']
    local_cm = nav['local_costmap']['local_costmap']['ros__parameters']

    door = house.DOOR_W
    corridor = 2.0                      # hallway depth in the generated world
    robot_radius = global_cm['robot_radius']
    body_l = platform['body_length']
    body_w = platform['body_width']

    print('clearance check\n')
    print(f'  doorway {door:.2f} m, hallway {corridor:.2f} m, '
          f'robot {body_l:.2f} x {body_w:.2f} m '
          f'(circumscribed radius {robot_radius:.2f} m)\n')

    worst = 0

    # --- 1. does the declared radius actually cover the body? --------------
    half_diagonal = math.hypot(body_l / 2.0, body_w / 2.0)
    if robot_radius < half_diagonal:
        worst = max(worst, report(
            FAIL, 'footprint radius',
            f'robot_radius {robot_radius:.3f} m is smaller than the body '
            f'half-diagonal {half_diagonal:.3f} m, so a rotating robot sweeps '
            f'outside its own costmap footprint',
            f'raise robot_radius to at least {half_diagonal:.3f} m'))
    else:
        worst = max(worst, report(
            PASS, 'footprint radius',
            f'robot_radius {robot_radius:.3f} m covers the body half-diagonal '
            f'{half_diagonal:.3f} m with {robot_radius - half_diagonal:.3f} m '
            f'to spare for the wheel clusters'))

    # --- 2. geometric fit ---------------------------------------------------
    free = door - 2.0 * robot_radius
    margin = free / 2.0
    if free <= 0:
        worst = max(worst, report(
            FAIL, 'doorway fit',
            f'a {robot_radius * 2:.2f} m wide robot cannot pass a {door:.2f} m '
            f'doorway at all'))
    else:
        worst = max(worst, report(
            PASS, 'doorway fit',
            f'{free:.3f} m of free lane for the robot centre, '
            f'{margin:.3f} m of clearance each side'))

    # --- 3. margin against the pose error that actually matters ------------
    ratio = margin / args.pose_error if args.pose_error > 0 else math.inf
    if ratio < 1.5:
        worst = max(worst, report(
            FAIL, 'margin vs localisation',
            f'{margin:.3f} m of clearance against {args.pose_error:.3f} m of '
            f'pose error is a ratio of {ratio:.1f}: the robot will clip door '
            f'frames',
            'use the gap-mode docking servo to cross thresholds, which closes '
            'the loop on the live scan instead of the map pose'))
    elif ratio < 3.0:
        worst = max(worst, report(
            WARN, 'margin vs localisation',
            f'{margin:.3f} m of clearance against {args.pose_error:.3f} m of '
            f'pose error is a ratio of {ratio:.1f} - workable, but thin',
            'dock_precisely(target="gap") exists for exactly this; prefer it '
            'for doorways when the pose estimate is degraded'))
    else:
        worst = max(worst, report(
            PASS, 'margin vs localisation',
            f'{margin:.3f} m of clearance against {args.pose_error:.3f} m of '
            f'pose error, a ratio of {ratio:.1f}'))

    # --- 4. does a zero-cost lane survive the inflation? -------------------
    for label, costmap in (('global', global_cm), ('local', local_cm)):
        inflation = costmap['inflation_layer']['inflation_radius']
        clear = door - 2.0 * inflation
        if clear <= 0:
            suggested = max(robot_radius + 0.04, (door - 0.20) / 2.0)
            worst = max(worst, report(
                WARN, f'{label} costmap inflation',
                f'inflation_radius {inflation:.2f} m from both jambs covers '
                f'{2 * inflation:.2f} m of a {door:.2f} m doorway, so no '
                f'zero-cost lane survives. Routes still exist - inflation is '
                f'not lethal - but every doorway in the house is high-cost, so '
                f'the cost-regulated controller crawls at each threshold and '
                f'the planner detours around doors whenever it can',
                f'reduce inflation_radius to about {suggested:.2f} m, which '
                f'keeps a {door - 2 * suggested:.2f} m free lane while still '
                f'exceeding the {robot_radius:.2f} m robot radius'))
        else:
            worst = max(worst, report(
                PASS, f'{label} costmap inflation',
                f'inflation_radius {inflation:.2f} m leaves a {clear:.2f} m '
                f'zero-cost lane through a {door:.2f} m doorway'))

    # --- 5. can it turn around in the hallway? -----------------------------
    if corridor < 2.0 * robot_radius:
        worst = max(worst, report(
            FAIL, 'hallway turning',
            f'the {corridor:.2f} m hallway is narrower than the robot\'s '
            f'{2 * robot_radius:.2f} m turning circle'))
    else:
        worst = max(worst, report(
            PASS, 'hallway turning',
            f'the {corridor:.2f} m hallway clears the robot\'s '
            f'{2 * robot_radius:.2f} m turning circle'))

    if worst == _status_rank[FAIL]:
        print('a clearance check failed')
        return 1
    if worst == _status_rank[WARN]:
        print('clearances pass with warnings')
        return 0
    print('all clearances pass')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
