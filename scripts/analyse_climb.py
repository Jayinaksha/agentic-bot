#!/usr/bin/env python3
"""
Quasi-static analysis of the tri-star climb.

    python3 scripts/analyse_climb.py
    python3 scripts/analyse_climb.py --riser 0.18 --tread 0.25
    python3 scripts/analyse_climb.py --params other_params.yaml

Why this exists
---------------
Whether the platform climbs is decided by geometry and statics long before it is
decided by a simulator. Gazebo will tell you *that* a climb failed; it is poor
at telling you *why*, because contact-rich tumbling is exactly the regime where
solver settings dominate the answer. So the checks that can be settled on paper
are settled here, on the same robot_params.yaml the URDF and the world generator
read, and re-run whenever those numbers change.

Where the numbers come from
---------------------------
Mass and centre of mass are read from the EXPANDED URDF when xacro is available,
not estimated from robot_params.yaml. That distinction matters: the tipping and
torque checks are only meaningful if they describe the robot that actually
spawns, and a summed-by-hand mass is exactly the kind of second copy that has
drifted before in this repo. Without xacro installed the script falls back to
the parameter estimate and says so, rather than silently analysing a robot that
does not exist.

What is checked
---------------
  1. Reach          can a cluster get a sub-wheel onto the step at all
  2. Tread fit      does a sub-wheel land on the tread or bridge two risers
  3. Gait match     does one 120 deg carrier rotation advance one step pitch
  4. Static tip     does the robot fall over backwards on the flight
  5. Torque         what motor torque the climb actually demands
  6. Speed          how long a flight takes, against the FSM timeout
  7. Ride height    does rolling mode hold a constant chassis height, which the
                    ToF terrain thresholds assume
  8. Mass model     does the URDF agree with robot_params.yaml about how heavy
                    the robot is and where its mass sits
  9. Inertia        are the inertia tensors physically possible

Everything is quasi-static: no impact loads, no wheel slip, no compliance. Real
margins will be worse, which is the right direction for a check whose job is to
catch designs that cannot work.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import yaml
except ImportError:
    sys.exit('PyYAML required: pip install pyyaml')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
DEFAULT_PARAMS = os.path.join(
    ROOT, 'src', 'r2d2_description', 'config', 'robot_params.yaml')

G = 9.81

PASS, WARN, FAIL = 'PASS', 'WARN', 'FAIL'


@dataclass
class Inertials:
    """Mass properties read from the expanded URDF."""

    total_mass: float
    com_z: float
    links: int
    bad_tensors: List[Tuple[str, float, float, float]]
    source: str


def load_inertials(urdf_path: str) -> Optional[Inertials]:
    """Expand the robot description and sum its real mass properties.

    Returns None when xacro is not importable, which is the normal case on a
    machine without ROS. CI always has it.
    """
    try:
        import xacro
    except ImportError:
        return None

    import xml.dom.minidom as minidom
    try:
        doc = xacro.process_file(urdf_path, mappings={'use_sim': 'true'})
    except Exception as exc:                       # noqa: BLE001
        raise SystemExit(f'xacro could not expand {urdf_path}: {exc}')

    robot = minidom.parseString(doc.toxml()).documentElement

    def origin_z(node) -> float:
        found = node.getElementsByTagName('origin')
        if not found:
            return 0.0
        value = found[0].getAttribute('xyz') or '0 0 0'
        return float(value.split()[2])

    # Only joints that are direct children of <robot> are kinematic; the
    # ros2_control block carries its own <joint> elements with no parent or
    # child, and walking those produces nonsense.
    tree = {}
    for joint in robot.childNodes:
        if getattr(joint, 'tagName', None) != 'joint':
            continue
        children = joint.getElementsByTagName('child')
        parents = joint.getElementsByTagName('parent')
        if not children or not parents:
            continue
        tree[children[0].getAttribute('link')] = (
            parents[0].getAttribute('link'), origin_z(joint))

    def world_z(link: str) -> float:
        z, seen = 0.0, set()
        while link in tree and link not in seen:
            seen.add(link)
            parent, offset = tree[link]
            z += offset
            link = parent
        return z

    total = moment = 0.0
    count = 0
    bad: List[Tuple[str, float, float, float]] = []

    for link in robot.childNodes:
        if getattr(link, 'tagName', None) != 'link':
            continue
        inertial = link.getElementsByTagName('inertial')
        if not inertial:
            continue
        name = link.getAttribute('name')
        mass = float(inertial[0].getElementsByTagName('mass')[0]
                     .getAttribute('value'))
        tensor = inertial[0].getElementsByTagName('inertia')[0]
        ixx, iyy, izz = (float(tensor.getAttribute(k))
                         for k in ('ixx', 'iyy', 'izz'))

        # A rigid body's principal moments must satisfy the triangle
        # inequality. A tensor that does not describes no physical object, and
        # a contact solver handed one produces motion that looks like a bug in
        # the controller.
        if not (ixx + iyy >= izz and iyy + izz >= ixx and ixx + izz >= iyy):
            bad.append((name, ixx, iyy, izz))

        total += mass
        moment += mass * (world_z(name) + origin_z(inertial[0]))
        count += 1

    return Inertials(total_mass=total, com_z=moment / total if total else 0.0,
                     links=count, bad_tensors=bad,
                     source=os.path.relpath(urdf_path, ROOT))


@dataclass
class Finding:
    check: str
    status: str
    detail: str
    advice: Optional[str] = None


def analyse(cfg: dict, riser: float, tread: float, steps: int,
            inertials: Optional[Inertials] = None) -> List[Finding]:
    p = cfg['platform']
    r_w = p['sub_wheel_radius']
    r_c = p['cluster_circumradius']
    wheelbase = p['wheelbase']
    axle_z = p['axle_height']
    tail_l = p['tail_length']
    max_cluster_rate = p['max_cluster_rate']
    climb_speed = p['climb_linear_vel']

    estimated_mass = p['body_mass'] + 4 * p['cluster_mass'] + p['tail_mass'] + 0.30
    if inertials is not None:
        mass = inertials.total_mass
        com_height = inertials.com_z
    else:
        mass = estimated_mass
        com_height = axle_z
    weight = mass * G

    findings: List[Finding] = []
    slope = math.atan2(riser, tread)
    step_pitch = math.hypot(riser, tread)

    # --- 1. reach ----------------------------------------------------------
    reach = r_c + r_w
    margin = (reach - riser) / riser if riser > 0 else math.inf
    if reach <= riser:
        findings.append(Finding(
            'reach', FAIL,
            f'cluster reach {reach:.3f} m does not exceed the {riser:.3f} m '
            f'riser: no sub-wheel can get onto the step',
            f'raise cluster_circumradius to at least '
            f'{(riser * 1.15 - r_w):.3f} m for a 15% margin'))
    elif margin < 0.08:
        findings.append(Finding(
            'reach', WARN,
            f'reach {reach:.3f} m clears the {riser:.3f} m riser by only '
            f'{margin * 100:.0f}%; wheel compliance and stair nosings will eat '
            f'that',
            'aim for 15% or more, or restrict the robot to shallower stairs'))
    else:
        findings.append(Finding(
            'reach', PASS,
            f'reach {reach:.3f} m clears the {riser:.3f} m riser by '
            f'{margin * 100:.0f}%'))

    # --- 2. tread fit ------------------------------------------------------
    if tread <= 2 * r_w:
        findings.append(Finding(
            'tread fit', FAIL,
            f'tread {tread:.3f} m is not wider than a sub-wheel '
            f'({2 * r_w:.3f} m): the wheel bridges two risers instead of '
            f'landing on the tread',
            f'sub_wheel_radius must stay below {tread / 2:.3f} m'))
    else:
        findings.append(Finding(
            'tread fit', PASS,
            f'tread {tread:.3f} m gives a sub-wheel '
            f'{tread - 2 * r_w:.3f} m of landing room'))

    # --- 3. gait match -----------------------------------------------------
    # One 120 deg carrier rotation moves the cluster centre by the chord
    # between two sub-wheel axles: 2 * r_c * sin(60 deg) = r_c * sqrt(3).
    advance = r_c * math.sqrt(3.0)
    ratio = advance / step_pitch
    ideal_r_c = step_pitch / math.sqrt(3.0)

    if 0.92 <= ratio <= 1.08:
        findings.append(Finding(
            'gait match', PASS,
            f'one 120 deg tumble advances {advance:.3f} m against a '
            f'{step_pitch:.3f} m step pitch (ratio {ratio:.2f}): the cluster '
            f'walks one step per rotation'))
    else:
        direction = 'short of' if ratio < 1 else 'past'
        findings.append(Finding(
            'gait match', WARN,
            f'one 120 deg tumble advances {advance:.3f} m, {direction} the '
            f'{step_pitch:.3f} m step pitch (ratio {ratio:.2f}). Sub-wheels '
            f'will land at varying points on each tread rather than on the '
            f'nosing, so the climb will be lumpy and slower than the ideal',
            f'a cluster_circumradius of {ideal_r_c:.3f} m would match this '
            f'staircase exactly; the platform still climbs without it, but '
            f'expect the FSM to need the full climb_timeout'))

    # --- 4. static tip -----------------------------------------------------
    # On the flight the body is inclined at `slope`. Backward tipping happens
    # when the centre of mass passes behind the rearmost support.
    rear_support = wheelbase / 2.0
    tip_angle_no_tail = math.atan2(rear_support, com_height)
    tip_angle_tail = math.atan2(rear_support + tail_l, com_height)

    if slope >= tip_angle_no_tail:
        status = FAIL if slope >= tip_angle_tail else WARN
        findings.append(Finding(
            'static tip', status,
            f'the {math.degrees(slope):.1f} deg flight exceeds the '
            f'{math.degrees(tip_angle_no_tail):.1f} deg backward tipping angle '
            f'of the bare chassis; the platform depends entirely on the tail '
            f'(good to {math.degrees(tip_angle_tail):.1f} deg)',
            'lower the centre of mass, lengthen the wheelbase, or treat the '
            'tail as a load-bearing part rather than a safety net'))
    else:
        origin = ('URDF' if inertials is not None
                  else 'axle height, no URDF available')
        findings.append(Finding(
            'static tip', PASS,
            f'{math.degrees(slope):.1f} deg flight against a '
            f'{math.degrees(tip_angle_no_tail):.1f} deg bare-chassis tipping '
            f'angle ({math.degrees(tip_angle_tail):.1f} deg with the tail), '
            f'centre of mass {com_height:.4f} m from the {origin}'))

    # --- 5. torque ---------------------------------------------------------
    # Steady state: gravity along the slope, shared by four clusters.
    steady = weight * math.sin(slope) * r_c / 4.0
    # Worst case: the two front clusters lift the machine over a nosing while
    # the rear ones are on the tread contributing little.
    peak = weight * r_c / 2.0

    findings.append(Finding(
        'torque', WARN if peak > 4.0 else PASS,
        f'{mass:.1f} kg needs {steady:.2f} N.m per cluster in steady climb and '
        f'up to {peak:.2f} N.m when two clusters carry the lift',
        (f'{peak:.1f} N.m is beyond a typical hobby gearmotor (2-4 N.m). Size '
         f'the drive for the peak, not the average, or the robot will stall '
         f'on the first nosing') if peak > 4.0 else None))

    # --- 6. speed ----------------------------------------------------------
    flight_length = steps * step_pitch
    max_speed = max_cluster_rate * r_c
    duration = flight_length / climb_speed if climb_speed > 0 else math.inf

    if climb_speed > max_speed:
        findings.append(Finding(
            'speed', FAIL,
            f'climb_linear_vel {climb_speed:.2f} m/s exceeds what '
            f'max_cluster_rate allows ({max_speed:.2f} m/s)',
            f'lower climb_linear_vel below {max_speed:.2f} m/s or raise '
            f'max_cluster_rate'))
    else:
        findings.append(Finding(
            'speed', PASS,
            f'a {steps}-step flight is {flight_length:.2f} m of travel, '
            f'{duration:.0f} s at {climb_speed:.2f} m/s '
            f'(ceiling {max_speed:.2f} m/s)'))

    # --- 7. ride height ----------------------------------------------------
    # A three-spoke carrier rests either on one sub-wheel (axle at r_c + r_w)
    # or straddling two (axle at r_c*cos(60 deg) + r_w). If the carrier is free
    # to rotate while rolling, the chassis bobs between the two, and every ToF
    # reading moves with it.
    high = r_c + r_w
    low = r_c * math.cos(math.pi / 3) + r_w
    bob = high - low
    step_min = cfg.get('limits', {}).get('step_up_min', 0.035)
    phase_held = bool(p.get('carrier_phase_hold', False))

    if bob < step_min:
        findings.append(Finding(
            'ride height', PASS,
            f'carrier bob {bob * 1000:.0f} mm stays under the '
            f'{step_min * 1000:.0f} mm ToF step threshold even unheld'))
    elif phase_held:
        findings.append(Finding(
            'ride height', PASS,
            f'a free carrier would bob {bob * 1000:.0f} mm between its '
            f'one-wheel ({high:.3f} m) and two-wheel ({low:.3f} m) rest '
            f'heights, well past the {step_min * 1000:.0f} mm ToF step '
            f'threshold - but carrier_phase_hold is on, so the controller '
            f'servos each carrier to the straddle phase and the ride height '
            f'stays at {low:.4f} m'))
    else:
        findings.append(Finding(
            'ride height', FAIL,
            f'an unheld carrier bobs {bob * 1000:.0f} mm between its '
            f'one-wheel ({high:.3f} m) and two-wheel ({low:.3f} m) rest '
            f'heights, which exceeds the {step_min * 1000:.0f} mm ToF step '
            f'threshold: the terrain monitor would see phantom steps on flat '
            f'floor',
            'set platform.carrier_phase_hold and servo the carriers to a '
            'defined phase in rolling mode; a zero-velocity hold parks them '
            'wherever they stopped'))

    # Cross-check the declared axle height against the two real rest positions.
    if not (math.isclose(axle_z, high, abs_tol=0.005)
            or math.isclose(axle_z, low, abs_tol=0.005)):
        findings.append(Finding(
            'ride height', WARN,
            f'axle_height is declared as {axle_z:.3f} m but the carrier can '
            f'only rest at {low:.3f} m (two wheels down) or {high:.3f} m (one '
            f'wheel down)',
            f'set axle_height to {low:.4f} m and park the carrier straddling '
            f'two sub-wheels: it is the lower centre of mass and the stabler '
            f'of the two'))

    # --- 8. mass model ------------------------------------------------------
    if inertials is None:
        findings.append(Finding(
            'mass model', WARN,
            f'xacro is not installed, so mass and centre of mass are estimated '
            f'from robot_params.yaml ({estimated_mass:.2f} kg, centre of mass '
            f'assumed at the axle). The tipping and torque results describe '
            f'that estimate, not the robot that spawns',
            'pip install xacro to analyse the real description'))
    else:
        drift = abs(inertials.total_mass - estimated_mass)
        detail = (f'{inertials.links} links, {inertials.total_mass:.3f} kg, '
                  f'centre of mass at {inertials.com_z:.4f} m '
                  f'({(inertials.com_z - axle_z) * 1000:+.0f} mm relative to '
                  f'the axle)')
        if drift > 0.10 * estimated_mass:
            findings.append(Finding(
                'mass model', FAIL,
                f'the URDF weighs {inertials.total_mass:.2f} kg but '
                f'robot_params.yaml sums to {estimated_mass:.2f} kg, a '
                f'{drift / estimated_mass * 100:.0f}% disagreement. {detail}',
                'the torque figures above are computed from the URDF; the '
                'component masses in robot_params.yaml no longer describe it'))
        else:
            findings.append(Finding(
                'mass model', PASS,
                f'{detail}; robot_params.yaml agrees to within '
                f'{drift / estimated_mass * 100:.1f}%'))

    # --- 9. inertia ---------------------------------------------------------
    if inertials is not None:
        if inertials.bad_tensors:
            names = ', '.join(name for name, *_ in inertials.bad_tensors[:4])
            findings.append(Finding(
                'inertia', FAIL,
                f'{len(inertials.bad_tensors)} link(s) have principal moments '
                f'that violate the triangle inequality and so describe no '
                f'physical object: {names}',
                'a contact solver handed an impossible tensor produces motion '
                'that reads as a controller bug; fix the tensors first'))
        else:
            findings.append(Finding(
                'inertia', PASS,
                f'all {inertials.links} inertia tensors are physically '
                f'realisable'))

    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--params', default=DEFAULT_PARAMS)
    ap.add_argument('--riser', type=float)
    ap.add_argument('--tread', type=float)
    ap.add_argument('--steps', type=int)
    ap.add_argument('--urdf', default=os.path.join(
        ROOT, 'src', 'r2d2_description', 'urdf', 'r2d2_tristar.urdf.xacro'))
    ap.add_argument('--no-urdf', action='store_true',
                    help='skip the URDF and use the robot_params estimate')
    args = ap.parse_args()

    with open(args.params) as fh:
        cfg = yaml.safe_load(fh)

    stairs = cfg.get('stairs', {})
    riser = args.riser if args.riser is not None else stairs.get('riser', 0.15)
    tread = args.tread if args.tread is not None else stairs.get('tread', 0.28)
    steps = args.steps if args.steps is not None else stairs.get('steps', 12)

    inertials = None if args.no_urdf else load_inertials(args.urdf)
    mass_source = 'expanded URDF' if inertials else 'robot_params estimate'

    print('tri-star climb analysis')
    print(f'  platform : {args.params}')
    print(f'  mass     : {mass_source}')
    print(f'  staircase: {steps} x {riser:.3f} m riser / {tread:.3f} m tread '
          f'({math.degrees(math.atan2(riser, tread)):.1f} deg)\n')

    findings = analyse(cfg, riser, tread, steps, inertials)
    width = max(len(f.check) for f in findings)

    for f in findings:
        print(f'  [{f.status}] {f.check.ljust(width)}  {f.detail}')
        if f.advice:
            for line in _wrap(f.advice, 68):
                print(f'         {" " * width}  -> {line}')
        print()

    failures = sum(1 for f in findings if f.status == FAIL)
    warnings = sum(1 for f in findings if f.status == WARN)
    print(f'{len(findings)} checks: {failures} failed, {warnings} warnings')
    return 1 if failures else 0


def _wrap(text: str, width: int) -> List[str]:
    words, lines, current = text.split(), [], ''
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f'{current} {word}'.strip()
    if current:
        lines.append(current)
    return lines


if __name__ == '__main__':
    raise SystemExit(main())
