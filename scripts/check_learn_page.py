#!/usr/bin/env python3
"""
Check the learn page against the constants it claims to use.

    python3 scripts/check_learn_page.py

docs/learn.html carries three browser simulations that re-implement formulas
which already exist in Python: the climb envelope from analyse_climb.py, and
height_delta / classify_delta / RiserContact from r2d2_locomotion/kinematics.py.
A transcription into JavaScript is a second copy, and every bug in this project
came from exactly that shape - two descriptions of one thing with nothing
joining them.

So the page states its constants in one `const P = {...}` block, and this reads
that block back out and compares it against the YAML. It also re-derives the
headline figures the page prints as prose, so a number written into a sentence
cannot quietly stop being true.

It does not execute the JavaScript. What it establishes is that the inputs are
the same; the formulas themselves are short, and the ones that matter are
covered by the Python tests.
"""

from __future__ import annotations

import math
import os
import re
import sys

try:
    import yaml
except ImportError:
    sys.exit('PyYAML required: pip install pyyaml')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
PAGE = os.path.join(ROOT, 'docs/learn.html')
PARAMS = os.path.join(ROOT, 'src/r2d2_description/config/robot_params.yaml')
LOCOMOTION = os.path.join(ROOT, 'src/r2d2_locomotion/config/locomotion.yaml')


def page_constants(html: str) -> dict:
    """Parse the `const P = { ... };` block into a dict."""
    match = re.search(r'const P = \{(.*?)\n\};', html, re.S)
    if not match:
        raise SystemExit('could not find the const P block in learn.html')
    body = match.group(1)
    body = re.sub(r'/\*.*?\*/', '', body, flags=re.S)
    out = {}
    for key, value in re.findall(r'(\w+):\s*([^,\n]+)', body):
        value = value.strip().rstrip(',')
        try:
            # A tiny JS Math shim, so an expression like `Math.PI / 6` parses.
            js_math = type('Math', (), {'PI': math.pi, 'sqrt': math.sqrt,
                                        'cos': math.cos, 'sin': math.sin})
            out[key] = float(eval(value, {'Math': js_math, '__builtins__': {}}))
        except Exception:                          # noqa: BLE001 - non-numeric
            continue
    return out


def main() -> int:
    html = open(PAGE).read()
    params = yaml.safe_load(open(PARAMS))
    locomotion = yaml.safe_load(open(LOCOMOTION))
    p = params['platform']
    tm = locomotion['terrain_monitor']['ros__parameters']
    cf = locomotion['climb_fsm']['ros__parameters']

    js = page_constants(html)
    expected = {
        'trackWidth': p['track_width'],
        'wheelbase': p['wheelbase'],
        'bodyLength': p['body_length'],
        'bodyHeight': p['body_height'],
        'tofMountHeight': tm['tof_mount_height'],
        'tofTilt': tm['tof_tilt'],
        'tofForwardOffset': tm['tof_forward_offset'],
        'stepUpMin': tm['step_up_min'],
        'stepUpMax': tm['step_up_max'],
        'cliffDrop': tm['cliff_drop'],
        'contactStallRatio': cf['contact_stall_ratio'],
        'contactConfirmS': cf['contact_confirm_s'],
        'bodyMass': p['body_mass'],
        'clusterMass': p['cluster_mass'],
        'tailMass': p['tail_mass'],
    }

    problems = 0
    print(f'{len(expected)} constants shared between learn.html and the YAML\n')
    for key, want in sorted(expected.items()):
        got = js.get(key)
        if got is None:
            print(f'[FAIL] learn.html does not define {key}; the page would use '
                  f'undefined in a formula')
            problems += 1
        elif abs(got - want) > 1e-6:
            print(f'[FAIL] {key}: page says {got}, config says {want}. The '
                  f'simulation is showing a robot that does not exist.')
            problems += 1
    if not problems:
        print(f'[ ok ] all {len(expected)} constants match')

    # Headline figures the page states as prose. A sentence is not checked by
    # anything else, and these are the ones a reader would act on.
    reach = p['cluster_circumradius'] + p['sub_wheel_radius']
    advance = p['cluster_circumradius'] * math.sqrt(3)
    pitch = math.hypot(params['stairs']['riser'], params['stairs']['tread'])
    creep = (tm['tof_forward_offset'] + tm['tof_mount_height'] / math.tan(tm['tof_tilt'])
             - p['wheelbase'] / 2 - cf['descent_margin'])
    high = p['cluster_circumradius'] + p['sub_wheel_radius']
    low = p['cluster_circumradius'] * math.cos(math.pi / 3) + p['sub_wheel_radius']

    claims = [
        ('reach in mm', f'{reach*1000:.0f}', ['0.165 m', f'{reach*1000:.0f}&nbsp;mm']),
        ('gait advance in mm', f'{advance*1000:.0f}', [f'{advance*1000:.0f}&nbsp;mm']),
        ('step pitch in mm', f'{pitch*1000:.0f}', [f'{pitch*1000:.0f}&nbsp;mm']),
        ('gait ratio', f'{advance/pitch:.2f}', [f'{advance/pitch:.2f}']),
        ('descent creep', f'{creep:.3f}', [f'{creep:.3f} m']),
        ('carrier bob in mm', f'{(high-low)*1000:.0f}', [f'{(high-low)*1000:.0f}&nbsp;mm']),
        ('straddle height in mm', f'{low*1000:.1f}', [f'{low*1000:.1f}&nbsp;mm']),
        ('flat return', f'{tm["tof_mount_height"]/math.sin(tm["tof_tilt"]):.3f}',
         [f'{tm["tof_mount_height"]/math.sin(tm["tof_tilt"]):.3f} m']),
    ]
    print()
    for name, value, needles in claims:
        if any(n in html for n in needles):
            print(f'[ ok ] page states the current {name} ({value})')
        else:
            print(f'[FAIL] page never states the current {name} ({value}); a '
                  f'figure in the prose has gone stale')
            problems += 1

    # The capture script and the page each name the demos. A slot with no
    # recipe can never be filled; a recipe with no slot records a file the page
    # will never show.
    script = os.path.join(ROOT, 'scripts/record_demo.sh')
    if os.path.exists(script):
        recipes = set(re.findall(r'^"([a-z-]+)\|', open(script).read(), re.M))
        slots = set(re.findall(r'data-media="([a-z-]+)"', html))
        print()
        for name in sorted(slots - recipes):
            print(f'[FAIL] the page has a slot for "{name}" but '
                  f'record_demo.sh has no recipe, so it can never be filled')
            problems += 1
        for name in sorted(recipes - slots):
            print(f'[FAIL] record_demo.sh records "{name}" but the page has no '
                  f'slot, so the file would never be shown')
            problems += 1
        if slots and slots == recipes:
            print(f'[ ok ] all {len(slots)} demo slots have a capture recipe')

    print()
    if problems:
        print(f'{problems} problem(s) found. Regenerate the diagrams and update '
              f'the prose in docs/learn.html.')
    else:
        print('the page and the robot agree')
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
