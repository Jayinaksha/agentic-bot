#!/usr/bin/env python3
"""
Generate the documentation diagrams from the robot's own constants.

    python3 scripts/generate_diagrams.py
    python3 scripts/generate_diagrams.py --out docs/diagrams

Every diagram here is drawn from robot_params.yaml and the locomotion config -
the same files the URDF and the Gazebo world are built from. A diagram that is
drawn by hand starts out right and drifts silently; one that is generated cannot
disagree with the robot without CI noticing, which is the discipline the rest of
this repo already runs on (see check_params.py and the world generator).

Output is plain SVG with no external references, so the pages work offline and
the files diff readably in git.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from typing import Dict, Tuple

try:
    import yaml
except ImportError:
    raise SystemExit('PyYAML required: pip install pyyaml')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
PARAMS = os.path.join(ROOT, 'src/r2d2_description/config/robot_params.yaml')
LOCOMOTION = os.path.join(ROOT, 'src/r2d2_locomotion/config/locomotion.yaml')

# Theme tokens, matching docs/index.html so the diagrams sit inside the page
# rather than on top of it.
GROUND = '#0E1114'
PLATE = '#14181D'
EDGE = '#262E37'
EDGE_HI = '#38424E'
STEEL = '#828E9B'
CHALK = '#E2E7EC'
OXIDE = '#C4552A'
EMBER = '#FF9A52'
SENSE = '#7FA3B8'

# Status palette, used only where a diagram classifies something. Fixed by the
# data-viz reference and never themed; every band is directly labelled, because
# a status colour must never carry meaning on its own.
GOOD = '#0ca30c'
WARNING = '#fab219'
SERIOUS = '#ec835a'
CRITICAL = '#d03b3b'

FONT = ('font-family="Saira Condensed, IBM Plex Sans, system-ui, sans-serif"')
MONO = ('font-family="IBM Plex Mono, ui-monospace, monospace"')


def load() -> Tuple[Dict, Dict]:
    with open(PARAMS) as fh:
        params = yaml.safe_load(fh)
    with open(LOCOMOTION) as fh:
        locomotion = yaml.safe_load(fh)
    return params, locomotion


def svg(width: int, height: int, body: str, title: str, desc: str) -> str:
    """Wrap a body in a themed, accessible SVG root."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"
     width="100%" role="img" aria-labelledby="t d" preserveAspectRatio="xMidYMid meet">
  <title id="t">{title}</title>
  <desc id="d">{desc}</desc>
  <rect width="{width}" height="{height}" fill="{PLATE}"/>
{body}
</svg>
'''


def text(x: float, y: float, s: str, size: int = 13, fill: str = CHALK,
         anchor: str = 'start', mono: bool = False, weight: str = '400',
         opacity: float = 1.0) -> str:
    font = MONO if mono else FONT
    return (f'  <text x="{x:.1f}" y="{y:.1f}" {font} font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" '
            f'opacity="{opacity}">{s}</text>\n')


def line(x1, y1, x2, y2, stroke=EDGE_HI, width=1.5, dash: str = '') -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}"{d}/>\n')


def dim(x1, y1, x2, y2, label: str, offset: float = 0, colour: str = STEEL) -> str:
    """A dimension line with arrow ticks and a centred label."""
    out = line(x1, y1, x2, y2, colour, 1)
    for x, y in ((x1, y1), (x2, y2)):
        out += (f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="{colour}"/>\n')
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    out += text(mx, my - 5 + offset, label, 11, colour, 'middle', mono=True)
    return out


# ---------------------------------------------------------------- diagrams

def diagram_platform(params: Dict, locomotion: Dict) -> str:
    """Side elevation: chassis, one cluster, the ToF beam, key dimensions."""
    p = params['platform']
    tm = locomotion['terrain_monitor']['ros__parameters']

    scale = 900.0          # px per metre
    W, H = 880, 470
    ox, oy = 150, 360      # origin: ground under the rear axle

    r_sub = p['sub_wheel_radius']
    r_clu = p['cluster_circumradius']
    axle = p['axle_height']
    body_l, body_h = p['body_length'], p['body_height']
    wb = p['wheelbase']

    def X(m): return ox + m * scale
    def Y(m): return oy - m * scale

    body = ''
    # Ground line.
    body += line(20, oy, W - 20, oy, EDGE_HI, 2)
    body += text(24, oy + 18, 'ground', 11, STEEL, mono=True)

    # Chassis.
    bx, by = X(-body_l / 2 + wb / 2), Y(axle + body_h / 2)
    body += (f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{body_l*scale:.1f}" '
             f'height="{body_h*scale:.1f}" rx="4" fill="{PLATE}" '
             f'stroke="{CHALK}" stroke-width="2" opacity="0.95"/>\n')
    body += text(X(wb / 2), by - 12, 'chassis', 12, STEEL, 'middle')

    # Two clusters, drawn at the straddle rest phase.
    for cx_m, label in ((0.0, 'rear'), (wb, 'front')):
        cx, cy = X(cx_m), Y(axle)
        body += (f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_clu*scale:.1f}" '
                 f'fill="none" stroke="{EDGE_HI}" stroke-width="1" '
                 f'stroke-dasharray="3 4"/>\n')
        # Straddle phase: sub-wheels at -30, 90, 210 degrees.
        for ang in (-30, 90, 210):
            a = math.radians(ang)
            wx = cx + r_clu * scale * math.cos(a)
            wy = cy - r_clu * scale * math.sin(a)
            body += line(cx, cy, wx, wy, EDGE_HI, 2)
            grounded = ang in (-30, 210)
            body += (f'  <circle cx="{wx:.1f}" cy="{wy:.1f}" '
                     f'r="{r_sub*scale:.1f}" fill="{PLATE}" '
                     f'stroke="{EMBER if grounded else STEEL}" '
                     f'stroke-width="{2.5 if grounded else 1.5}"/>\n')
        body += (f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{CHALK}"/>\n')
        body += text(cx, Y(axle) + 4, '', 10)

    # Axle height dimension.
    body += dim(X(-0.13), oy, X(-0.13), Y(axle),
                f'axle {axle*1000:.1f} mm')
    body += line(X(-0.15), Y(axle), X(0), Y(axle), STEEL, 1, '2 3')

    # Wheelbase.
    body += dim(X(0), oy + 34, X(wb), oy + 34, f'wheelbase {wb*1000:.0f} mm')

    # ToF beam from the front face.
    tof_h = tm['tof_mount_height']
    tilt = tm['tof_tilt']
    tof_x = X(wb + body_l / 2 - 0.0)
    tof_y = Y(tof_h)
    spot = tof_h / math.tan(tilt)
    body += (f'  <circle cx="{tof_x:.1f}" cy="{tof_y:.1f}" r="4" '
             f'fill="{SENSE}"/>\n')
    body += line(tof_x, tof_y, X(wb + body_l / 2 + spot), oy, SENSE, 2)
    body += (f'  <circle cx="{X(wb + body_l/2 + spot):.1f}" cy="{oy:.1f}" '
             f'r="4" fill="none" stroke="{SENSE}" stroke-width="2"/>\n')
    body += text(X(wb + body_l / 2 + spot) + 10, oy - 10,
                 f'beam spot {spot*1000:.0f} mm ahead', 11, SENSE, mono=True)
    body += text(tof_x + 8, tof_y - 8, f'ToF {math.degrees(tilt):.0f}° down',
                 11, SENSE, mono=True)
    body += dim(tof_x + 26, oy, tof_x + 26, tof_y, f'{tof_h*1000:.1f} mm', colour=SENSE)

    # Reach annotation on the front cluster.
    reach = r_clu + r_sub
    body += line(X(wb), Y(axle), X(wb), Y(axle) - reach * scale, EMBER, 1.5, '4 3')
    body += text(X(wb) + 8, Y(axle) - reach * scale - 6,
                 f'reach = r_cluster + r_sub = {reach*1000:.0f} mm',
                 11, EMBER, mono=True)

    body += text(24, 34, 'PLATFORM, SIDE ELEVATION', 16, CHALK, weight='600')
    body += text(24, 54,
                 'Drawn at the straddle rest phase the controller holds in '
                 'rolling mode. Filled wheels are in contact.', 12, STEEL)

    return svg(W, H, body, 'Tri-star platform side elevation',
               f'Chassis {body_l*1000:.0f} by {body_h*1000:.0f} mm on two of '
               f'four tri-star clusters. Cluster circumradius {r_clu*1000:.0f} '
               f'mm, sub-wheel radius {r_sub*1000:.0f} mm, giving a reach of '
               f'{reach*1000:.0f} mm. Axle sits {axle*1000:.1f} mm above ground. '
               f'A Time-of-Flight beam leaves the front face {tof_h*1000:.1f} mm '
               f'up, angled {math.degrees(tilt):.0f} degrees down, meeting the '
               f'floor {spot*1000:.0f} mm ahead of the sensor.')


def diagram_carrier_bob(params: Dict, locomotion: Dict) -> str:
    """The two carrier rest positions and the ride-height band between them."""
    p = params['platform']
    tm = locomotion['terrain_monitor']['ros__parameters']
    r_sub, r_clu = p['sub_wheel_radius'], p['cluster_circumradius']
    step_min = tm['step_up_min']

    high = r_clu + r_sub
    low = r_clu * math.cos(math.pi / 3) + r_sub
    bob = high - low

    # Scale is chosen so the taller of the two rest positions - cluster centre
    # at `high`, plus the carrier radius, plus a sub-wheel - still clears the
    # title band. At 1100 px/m it did not, and the top wheel was cut off.
    W, H = 880, 500
    oy = 372
    headroom = oy - 92
    scale = headroom / (high + r_clu + r_sub)

    def Y(m): return oy - m * scale

    body = ''
    body += text(24, 34, 'WHY THE CARRIER IS HELD AT A PHASE', 16, CHALK, weight='600')
    body += text(24, 54,
                 'A three-spoke carrier has two rest positions. Holding it at '
                 'zero velocity parks it at whichever it reached.', 12, STEEL)

    for i, (cx, phase, height, label, note) in enumerate((
            (240, 'straddle', low, 'two wheels down',
             'stable, lower, and what the controller holds'),
            (600, 'balanced', high, 'one wheel down',
             'unstable, 57 mm higher'))):
        cy = Y(height)
        body += line(cx - 150, oy, cx + 150, oy, EDGE_HI, 2)
        body += (f'  <circle cx="{cx}" cy="{cy:.1f}" r="{r_clu*scale:.1f}" '
                 f'fill="none" stroke="{EDGE_HI}" stroke-width="1" '
                 f'stroke-dasharray="3 4"/>\n')
        angles = (-30, 90, 210) if phase == 'straddle' else (-90, 30, 150)
        for ang in angles:
            a = math.radians(ang)
            wx = cx + r_clu * scale * math.cos(a)
            wy = cy - r_clu * scale * math.sin(a)
            grounded = abs((wy + r_sub * scale) - oy) < 2
            body += line(cx, cy, wx, wy, EDGE_HI, 2)
            body += (f'  <circle cx="{wx:.1f}" cy="{wy:.1f}" '
                     f'r="{r_sub*scale:.1f}" fill="{PLATE}" '
                     f'stroke="{EMBER if grounded else STEEL}" '
                     f'stroke-width="{2.5 if grounded else 1.5}"/>\n')
        body += (f'  <circle cx="{cx}" cy="{cy:.1f}" r="3" fill="{CHALK}"/>\n')
        body += text(cx, oy + 26, label, 13, CHALK, 'middle', weight='600')
        body += text(cx, oy + 44, f'axle {height*1000:.1f} mm', 11,
                     EMBER if phase == 'straddle' else STEEL, 'middle', mono=True)
        body += text(cx, oy + 62, note, 11, STEEL, 'middle')

    # The band between them, against the step threshold.
    bx = 800
    body += line(bx, Y(low), bx, Y(high), OXIDE, 2)
    body += line(bx - 6, Y(low), bx + 6, Y(low), OXIDE, 2)
    body += line(bx - 6, Y(high), bx + 6, Y(high), OXIDE, 2)
    body += text(bx - 12, (Y(low) + Y(high)) / 2, f'{bob*1000:.0f} mm', 12,
                 OXIDE, 'end', mono=True, weight='600')
    body += text(bx - 12, (Y(low) + Y(high)) / 2 + 16,
                 f'vs {step_min*1000:.0f} mm step threshold', 10, OXIDE, 'end')

    body += text(24, H - 40,
                 f'{bob*1000:.0f} mm of ride-height variation against a '
                 f'{step_min*1000:.0f} mm step threshold.', 13, EMBER, weight='600')
    body += text(24, H - 20,
                 'An unheld carrier makes the robot read phantom stairs on flat '
                 'floor, and the climb FSM can then be triggered by nothing.',
                 12, STEEL)

    return svg(W, H, body, 'The two carrier rest positions',
               f'A three-spoke carrier rests either straddling two sub-wheels, '
               f'with its axle {low*1000:.1f} mm above ground, or balanced on '
               f'one, {high*1000:.1f} mm up. The {bob*1000:.0f} mm difference '
               f'exceeds the {step_min*1000:.0f} mm height change the terrain '
               f'monitor treats as a step, so a carrier left at zero velocity '
               f'would make flat floor read as stairs.')


def diagram_terrain_bands(params: Dict, locomotion: Dict) -> str:
    """How a floor-height change is classified, as one labelled axis."""
    tm = locomotion['terrain_monitor']['ros__parameters']
    step_min, step_max = tm['step_up_min'], tm['step_up_max']
    cliff = tm['cliff_drop']

    W, H = 880, 300
    x0, x1 = 70, W - 70
    axis_y = 170
    lo, hi = -0.20, 0.26

    def X(v): return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    bands = [
        (lo, -cliff, CRITICAL, 'cliff', 'a drop: refuse, mark the costmap'),
        (-cliff, step_min, GOOD, 'flat', 'drive over it, sills included'),
        (step_min, step_max, WARNING, 'climbable riser',
         'the route upstairs, deliberately NOT an obstacle'),
        (step_max, hi, SERIOUS, 'blocked', 'taller than the cluster can mount'),
    ]

    body = ''
    body += text(24, 34, 'HOW A HEIGHT CHANGE IS CLASSIFIED', 16, CHALK, weight='600')
    body += text(24, 54,
                 'One axis, four outcomes. Position carries the meaning; colour '
                 'and label reinforce it.', 12, STEEL)

    for a, b, colour, name, note in bands:
        xa, xb = X(a), X(b)
        body += (f'  <rect x="{xa+1:.1f}" y="{axis_y-26}" '
                 f'width="{max(xb-xa-2, 1):.1f}" height="26" rx="3" '
                 f'fill="{colour}" opacity="0.9"/>\n')
        mid = (xa + xb) / 2
        body += text(mid, axis_y - 34, name, 12, CHALK, 'middle', weight='600')
        body += text(mid, axis_y + 22, note, 10, STEEL, 'middle')

    body += line(x0, axis_y, x1, axis_y, EDGE_HI, 1.5)
    for v, label in ((-cliff, f'-{cliff*1000:.0f}'),
                     (0, '0'),
                     (step_min, f'+{step_min*1000:.0f}'),
                     (step_max, f'+{step_max*1000:.0f}')):
        body += line(X(v), axis_y - 4, X(v), axis_y + 5, CHALK, 1.5)
        body += text(X(v), axis_y + 17, label, 10, CHALK, 'middle', mono=True)

    body += text(x1, axis_y + 44, 'floor height change under the beam, mm',
                 11, STEEL, 'end', mono=True)
    body += text(24, H - 24,
                 'A door sill of 18 to 30 mm sits inside "flat" on purpose; a '
                 'stair riser of 150 mm sits inside "climbable riser".',
                 12, STEEL)

    return svg(W, H, body, 'Terrain classification bands',
               f'A single axis of floor-height change, divided into four '
               f'labelled regions. Below minus {cliff*1000:.0f} mm is a cliff. '
               f'From minus {cliff*1000:.0f} to plus {step_min*1000:.0f} mm is '
               f'flat, which is where door sills fall. From {step_min*1000:.0f} '
               f'to {step_max*1000:.0f} mm is a climbable riser, which is '
               f'deliberately not treated as an obstacle. Above '
               f'{step_max*1000:.0f} mm is blocked.')


def diagram_stairs(params: Dict) -> str:
    """The staircase against the cluster's reach and gait."""
    p, st = params['platform'], params['stairs']
    riser, tread, steps = st['riser'], st['tread'], st['steps']
    r_sub, r_clu = p['sub_wheel_radius'], p['cluster_circumradius']
    reach = r_clu + r_sub
    advance = r_clu * math.sqrt(3.0)
    pitch = math.hypot(riser, tread)

    scale = 620.0
    W, H = 880, 430
    ox, oy = 90, 360
    shown = 5

    def X(m): return ox + m * scale
    def Y(m): return oy - m * scale

    body = ''
    body += text(24, 34, 'THE FLIGHT, AGAINST THE CLUSTER', 16, CHALK, weight='600')
    body += text(24, 54,
                 f'First {shown} of {steps} steps. Every dimension below comes '
                 f'from robot_params.yaml.', 12, STEEL)

    # Steps.
    for i in range(shown):
        x = X(i * tread)
        y = Y((i + 1) * riser)
        body += (f'  <rect x="{x:.1f}" y="{y:.1f}" '
                 f'width="{tread*scale:.1f}" height="{(i+1)*riser*scale:.1f}" '
                 f'fill="{GROUND}" stroke="{EDGE_HI}" stroke-width="1.5"/>\n')
    body += line(20, oy, X(0), oy, EDGE_HI, 2)

    # Riser and tread dimensions on the first step.
    body += dim(X(-0.06), oy, X(-0.06), Y(riser), f'riser {riser*1000:.0f}')
    body += dim(X(0), Y(riser) - 16, X(tread), Y(riser) - 16,
                f'tread {tread*1000:.0f}')

    # Reach arc from the nosing.
    nose_x, nose_y = X(0), Y(riser)
    body += (f'  <path d="M {nose_x:.1f} {nose_y - reach*scale:.1f} '
             f'A {reach*scale:.1f} {reach*scale:.1f} 0 0 1 '
             f'{nose_x + reach*scale:.1f} {nose_y:.1f}" fill="none" '
             f'stroke="{EMBER}" stroke-width="1.5" stroke-dasharray="4 3"/>\n')
    body += text(nose_x + 12, nose_y - reach * scale - 8,
                 f'reach {reach*1000:.0f} mm > riser {riser*1000:.0f} mm '
                 f'({(reach-riser)/riser*100:.0f}% margin)',
                 11, EMBER, mono=True)

    # Gait advance against step pitch.
    gy = oy + 44
    body += line(X(0), gy, X(pitch), gy, STEEL, 1)
    body += text(X(pitch) + 8, gy + 4, f'step pitch {pitch*1000:.0f} mm',
                 11, STEEL, mono=True)
    body += line(X(0), gy + 20, X(advance), gy + 20, OXIDE, 2.5)
    body += text(X(advance) + 8, gy + 24,
                 f'one 120° tumble advances {advance*1000:.0f} mm '
                 f'(ratio {advance/pitch:.2f})', 11, OXIDE, mono=True)

    body += text(24, H - 16,
                 'The cluster does not walk one step per rotation. It still '
                 'climbs; the gait is lumpy, and Gazebo has to settle whether '
                 'that matters.', 12, STEEL)

    return svg(W, H, body, 'Staircase against the cluster geometry',
               f'A {riser*1000:.0f} mm riser with a {tread*1000:.0f} mm tread. '
               f'The cluster reach of {reach*1000:.0f} mm clears the riser by '
               f'{(reach-riser)/riser*100:.0f} percent. One 120 degree carrier '
               f'rotation advances {advance*1000:.0f} mm against a step pitch '
               f'of {pitch*1000:.0f} mm, a ratio of {advance/pitch:.2f}, so the '
               f'cluster does not walk exactly one step per rotation.')


def diagram_stack(params: Dict) -> str:
    """The command path, and who is allowed to write to the wheels."""
    W, H = 880, 470
    body = ''
    body += text(24, 34, 'THE COMMAND PATH', 16, CHALK, weight='600')
    body += text(24, 54,
                 'One writer to /cmd_vel, enforced in CI. Nothing can take the '
                 'platform away from the climb FSM mid-staircase.', 12, STEEL)

    boxes = [
        (60, 100, 210, 'Nav2', 'controller_server', SENSE),
        (60, 170, 210, 'velocity_smoother', '/cmd_vel_smoothed', SENSE),
        (60, 240, 210, 'collision_monitor', '/cmd_vel_nav', SENSE),
        (330, 240, 230, 'climb_fsm', 'the only writer', EMBER),
        (620, 240, 210, 'tristar_controller', '/cmd_vel', CHALK),
        (620, 340, 210, '16 joint velocities', 'ros2_control', STEEL),
        (330, 100, 230, 'precise_docking', 'also writes /cmd_vel_nav', SENSE),
        (330, 350, 230, 'terrain_monitor', 'gates via /terrain/estop', OXIDE),
    ]
    for x, y, w, title, sub, colour in boxes:
        body += (f'  <rect x="{x}" y="{y}" width="{w}" height="52" rx="5" '
                 f'fill="{GROUND}" stroke="{colour}" stroke-width="1.6"/>\n')
        body += text(x + 12, y + 22, title, 13, CHALK, weight='600')
        body += text(x + 12, y + 39, sub, 11, colour, mono=True)

    def arrow(x1, y1, x2, y2, colour=EDGE_HI):
        return (line(x1, y1, x2, y2, colour, 1.6) +
                f'  <circle cx="{x2}" cy="{y2}" r="3.5" fill="{colour}"/>\n')

    body += arrow(165, 152, 165, 170)
    body += arrow(165, 222, 165, 240)
    body += arrow(270, 266, 330, 266)
    body += arrow(560, 266, 620, 266)
    body += arrow(725, 292, 725, 340)
    body += arrow(445, 152, 445, 240, SENSE)
    body += arrow(445, 350, 445, 292, OXIDE)

    body += (f'  <rect x="330" y="205" width="230" height="26" rx="3" '
             f'fill="{EMBER}" opacity="0.12"/>\n')
    body += text(445, 222, 'gate: passes through, or substitutes', 11,
                 EMBER, 'middle')

    body += text(24, H - 20,
                 'In IDLE the FSM republishes navigation’s command '
                 'unchanged. In every other state it substitutes its own and '
                 'switches the transmission.', 12, STEEL)

    return svg(W, H, body, 'The velocity command path',
               'Nav2 writes to velocity_smoother, then collision_monitor, then '
               'cmd_vel_nav. The climb FSM is the sole writer of cmd_vel, which '
               'reaches tristar_controller and becomes sixteen joint '
               'velocities. precise_docking also writes cmd_vel_nav, and '
               'terrain_monitor can stop the FSM.')


def diagram_fsm(locomotion: Dict) -> str:
    """The climb state machine, with what moves it between states."""
    cf = locomotion['climb_fsm']['ros__parameters']
    W, H = 880, 400
    body = ''
    body += text(24, 34, 'THE CLIMB STATE MACHINE', 16, CHALK, weight='600')
    body += text(24, 54,
                 'Direction is read from the terrain, not the request: a riser '
                 'ahead means up, a drop means down.', 12, STEEL)

    states = [
        (60, 150, 'IDLE', 'rolling; passes\ncommands through', CHALK),
        (215, 150, 'ALIGN', 'square up on\nToF skew', SENSE),
        (370, 150, 'APPROACH', 'creep until\nthe wheels stall', SENSE),
        (525, 150, 'CLIMB', 'tumbling;\nIMU-integrated rise', EMBER),
        (680, 150, 'SETTLE', 'rolling; hold\nfor relocalisation', CHALK),
        (370, 290, 'ABORT', 'reverse clear,\nlatch the reason', CRITICAL),
    ]
    for x, y, name, sub, colour in states:
        body += (f'  <rect x="{x}" y="{y}" width="140" height="66" rx="6" '
                 f'fill="{GROUND}" stroke="{colour}" stroke-width="1.8"/>\n')
        body += text(x + 70, y + 24, name, 14, colour, 'middle', weight='600')
        for i, part in enumerate(sub.split('\n')):
            body += text(x + 70, y + 41 + i * 13, part, 10, STEEL, 'middle')

    def arrow(x1, y1, x2, y2, label='', colour=EDGE_HI, up=True):
        out = line(x1, y1, x2, y2, colour, 1.6)
        out += f'  <circle cx="{x2}" cy="{y2}" r="3.5" fill="{colour}"/>\n'
        if label:
            out += text((x1 + x2) / 2, (y1 + y2) / 2 - (8 if up else -14),
                        label, 10, colour, 'middle', mono=True)
        return out

    body += arrow(200, 183, 215, 183, 'request', STEEL)
    body += arrow(355, 183, 370, 183, 'square', STEEL)
    body += arrow(510, 183, 525, 183, 'stall', EMBER)
    body += arrow(665, 183, 680, 183, 'level', STEEL)
    body += arrow(750, 150, 750, 120, '', STEEL)
    body += line(750, 120, 130, 120, STEEL, 1.6)
    body += arrow(130, 120, 130, 150, 'done', STEEL)
    body += arrow(440, 216, 440, 290, 'timeout / stall / e-stop', CRITICAL, up=False)
    body += line(370, 323, 130, 323, CRITICAL, 1.6)
    body += arrow(130, 323, 130, 216, '', CRITICAL)

    body += text(24, H - 34,
                 'APPROACH confirms contact from a STALL, not from body '
                 'pitch: with the carriers phase-locked the chassis cannot '
                 'tip, so a pitch threshold would never fire.', 12, EMBER)
    body += text(24, H - 16,
                 f'Contact = both front beams see a riser AND travel falls '
                 f'below {cf["contact_stall_ratio"]*100:.0f}% of commanded, '
                 f'for {cf["contact_confirm_s"]} s.', 12, STEEL)

    return svg(W, H, body, 'The climb state machine',
               'Six states: IDLE, ALIGN, APPROACH, CLIMB, SETTLE and ABORT. '
               'A request moves IDLE to ALIGN. Squaring up moves ALIGN to '
               'APPROACH. A wheel stall moves APPROACH to CLIMB. Level '
               'attitude moves CLIMB to SETTLE, then back to IDLE. A timeout, '
               'a stall in the climb, or an emergency stop moves any active '
               'state to ABORT and back to IDLE.')


DIAGRAMS = {
    'platform-elevation.svg': lambda p, l: diagram_platform(p, l),
    'carrier-rest-positions.svg': lambda p, l: diagram_carrier_bob(p, l),
    'terrain-bands.svg': lambda p, l: diagram_terrain_bands(p, l),
    'staircase-geometry.svg': lambda p, l: diagram_stairs(p),
    'command-path.svg': lambda p, l: diagram_stack(p),
    'climb-fsm.svg': lambda p, l: diagram_fsm(l),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs', 'diagrams'))
    ap.add_argument('--check', action='store_true',
                    help='fail if any committed diagram differs from a fresh render')
    args = ap.parse_args()

    params, locomotion = load()
    os.makedirs(args.out, exist_ok=True)

    stale = []
    rendered = {}
    for name, build in sorted(DIAGRAMS.items()):
        content = build(params, locomotion)
        rendered[name] = content
        path = os.path.join(args.out, name)
        if args.check:
            existing = open(path).read() if os.path.exists(path) else ''
            if existing != content:
                stale.append(name)
            continue
        with open(path, 'w') as fh:
            fh.write(content)
        print(f'  wrote {os.path.relpath(path, ROOT)} ({len(content)} bytes)')

    # docs/learn.html inlines these rather than linking them, so the page needs
    # updating too. Inlined rather than <img src> so the page works offline and
    # the SVG inherits the page's own fonts.
    page = os.path.join(ROOT, 'docs', 'learn.html')
    if os.path.exists(page):
        html = open(page).read()
        updated = html
        for name, content in rendered.items():
            uid = name[:-4].replace('-', '_')
            svg = content.replace('<?xml version="1.0" ?>\n', '')
            svg = svg.replace('aria-labelledby="t d"',
                              f'aria-labelledby="t_{uid} d_{uid}"')
            svg = svg.replace('<title id="t">', f'<title id="t_{uid}">')
            svg = svg.replace('<desc id="d">', f'<desc id="d_{uid}">')
            svg = svg.rstrip('\n')
            pattern = re.compile(
                r'<svg xmlns[^>]*aria-labelledby="t_%s d_%s".*?</svg>' % (uid, uid),
                re.S)
            updated, count = pattern.subn(lambda _m: svg, updated)
            if count != 1 and not args.check:
                print(f'::warning::{name} appears {count} times in learn.html, '
                      f'expected once')
        if args.check:
            if updated != html:
                stale.append('docs/learn.html (inlined copies)')
        elif updated != html:
            open(page, 'w').write(updated)
            print('  updated docs/learn.html with the inlined copies')

    if args.check:
        for name in stale:
            print(f'::error::{name} is out of date with robot_params.yaml. '
                  f'Regenerate: python3 scripts/generate_diagrams.py')
        if stale:
            return 1
        print(f'all {len(DIAGRAMS)} diagrams match the current constants')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
