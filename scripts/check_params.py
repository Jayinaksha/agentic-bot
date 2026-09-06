#!/usr/bin/env python3
"""
Cross-check node parameters against the YAML that is supposed to set them.

    python3 scripts/check_params.py
    python3 scripts/check_params.py --verbose

Why this exists
---------------
A misspelled parameter in ROS 2 fails silently. The node declares
`carrier_hold_gain`, the YAML says `carrier_hold_gian`, and nothing complains:
the node runs happily on its hard-coded default and the value you carefully
tuned is simply ignored. On this robot that means driving with a default slip
factor, or a climb FSM using a 15 s approach timeout you thought you had raised.

There is no runtime error to catch, so it has to be caught statically. This
walks every node's `declare_parameters` block, matches it against the YAML
section named for that node, and reports both directions:

  * a YAML key no node declares  - a typo, or config left behind after a rename
  * a declared parameter with no YAML entry - fine if the default is right,
    listed so the choice is deliberate

It also checks that each YAML section actually corresponds to a node, since a
section named for a node that does not exist is config that will never load.

Finally it checks MIRRORED constants. Several physical dimensions have to appear
in more than one file - robot_params.yaml is the design source of truth, but a
running node reads its own parameter file, and the URDF is built from a third.
Nothing at runtime notices when those copies disagree; the robot simply behaves
as though it has a geometry it does not have. Two bugs in this repo came from
exactly that, so the copies are now declared here and compared.
"""

from __future__ import annotations

import argparse
import ast
import glob
import math
import os
import re
import sys
from typing import Dict, List, Set, Tuple

try:
    import yaml
except ImportError:
    sys.exit('PyYAML required: pip install pyyaml')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))

# YAML sections owned by upstream packages. Their parameters are declared inside
# Nav2, robot_localization, slam_toolbox and ros2_control, not in this repo, so
# there is nothing here to match them against.
EXTERNAL_SECTIONS = {
    'amcl', 'bt_navigator', 'controller_server', 'planner_server',
    'smoother_server', 'behavior_server', 'velocity_smoother',
    'collision_monitor', 'global_costmap', 'local_costmap', 'map_server',
    'ekf_filter_node', 'slam_toolbox', 'laser_scan_matcher',
    'controller_manager', 'joint_state_broadcaster',
    'tristar_velocity_controller', 'lifecycle_manager_navigation',
    'lifecycle_manager_localization',
}


def declared_parameters(path: str) -> Tuple[str, Set[str]]:
    """Node name and the parameters it declares, from a source file.

    Parsed rather than regexed: `declare_parameters` takes a list of tuples and
    a regex over that reliably mangles anything with a nested structure.
    """
    source = open(path).read()
    node_name = None
    match = re.search(r"super\(\)\.__init__\(\s*'([^']+)'", source)
    if match:
        node_name = match.group(1)

    names: Set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return node_name, names

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute)
                and func.attr == 'declare_parameters'):
            continue
        for arg in node.args:
            if not isinstance(arg, (ast.List, ast.Tuple)):
                continue
            for element in arg.elts:
                if isinstance(element, (ast.Tuple, ast.List)) and element.elts:
                    first = element.elts[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        names.add(first.value)
    return node_name, names


def yaml_sections(path: str) -> Dict[str, Set[str]]:
    """Node name -> parameter keys, from a ROS 2 parameter file."""
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        return {}

    out: Dict[str, Set[str]] = {}
    for section, body in data.items():
        if not isinstance(body, dict):
            continue
        params = body.get('ros__parameters')
        if isinstance(params, dict):
            out[section] = set(params)
        else:
            # Nested form, as Nav2 uses for the costmaps.
            for inner, inner_body in body.items():
                if isinstance(inner_body, dict) and isinstance(
                        inner_body.get('ros__parameters'), dict):
                    out[inner] = set(inner_body['ros__parameters'])
    return out


# --------------------------------------------------------------------------
# Mirrored constants.
#
# (label, source path in robot_params.yaml, [copies elsewhere]).
# A copy is (config file, section, key). Everything is compared exactly, since
# these are declared numbers rather than computed ones.
# --------------------------------------------------------------------------
LOCOMOTION = 'src/r2d2_locomotion/config/locomotion.yaml'

MIRRORS = [
    ('sub_wheel_radius', ('platform', 'sub_wheel_radius'),
     [(LOCOMOTION, 'tristar_controller', 'sub_wheel_radius')]),
    ('cluster_circumradius', ('platform', 'cluster_circumradius'),
     [(LOCOMOTION, 'tristar_controller', 'cluster_circumradius')]),
    ('track_width', ('platform', 'track_width'),
     [(LOCOMOTION, 'tristar_controller', 'track_width')]),
    ('wheelbase', ('platform', 'wheelbase'),
     [(LOCOMOTION, 'tristar_controller', 'wheelbase'),
      (LOCOMOTION, 'climb_fsm', 'wheelbase')]),
    ('max_wheel_rate', ('platform', 'max_wheel_rate'),
     [(LOCOMOTION, 'tristar_controller', 'max_wheel_rate')]),
    ('max_cluster_rate', ('platform', 'max_cluster_rate'),
     [(LOCOMOTION, 'tristar_controller', 'max_cluster_rate')]),
    ('step_up_min', ('limits', 'step_up_min'),
     [(LOCOMOTION, 'terrain_monitor', 'step_up_min')]),
    ('cliff_drop', ('limits', 'cliff_drop'),
     [(LOCOMOTION, 'terrain_monitor', 'cliff_drop')]),
    ('max_pitch_flat', ('limits', 'max_pitch_flat'),
     [(LOCOMOTION, 'terrain_monitor', 'max_pitch_flat')]),
    ('max_pitch_climb', ('limits', 'max_pitch_climb'),
     [(LOCOMOTION, 'terrain_monitor', 'max_pitch_climb')]),
    ('max_roll', ('limits', 'max_roll'),
     [(LOCOMOTION, 'terrain_monitor', 'max_roll')]),
]


def _read(path: str):
    with open(os.path.join(ROOT, path)) as fh:
        return yaml.safe_load(fh)


def _param(config, section: str, key: str):
    body = config.get(section, {})
    params = body.get('ros__parameters', {}) if isinstance(body, dict) else {}
    return params.get(key, KeyError)


def check_mirrors() -> int:
    """Compare duplicated constants against robot_params.yaml."""
    problems = 0
    platform = _read('src/r2d2_description/config/robot_params.yaml')
    cache = {}

    print('mirrored constants')
    for label, (group, key), copies in MIRRORS:
        source = platform.get(group, {}).get(key)
        if source is None:
            print(f'[FAIL] robot_params.yaml has no {group}.{key}')
            problems += 1
            continue
        for path, section, target_key in copies:
            if path not in cache:
                cache[path] = _read(path)
            value = _param(cache[path], section, target_key)
            if value is KeyError:
                print(f'[FAIL] {path} {section}.{target_key} is missing; '
                      f'robot_params.yaml sets {label} = {source}')
                problems += 1
            elif value != source:
                print(f'[FAIL] {label} disagrees: robot_params.yaml says '
                      f'{source}, {path} {section}.{target_key} says {value}. '
                      f'Nothing at runtime notices, and the robot behaves as '
                      f'though it has a geometry it does not have.')
                problems += 1
    if not problems:
        total = sum(len(copies) for _, _, copies in MIRRORS)
        print(f'[ ok ] {total} copies of {len(MIRRORS)} constants all agree '
              f'with robot_params.yaml')

    # Derived values: these must FOLLOW from the platform, not be chosen.
    print('\nderived constants')
    p = platform['platform']
    locomotion = cache.get(LOCOMOTION) or _read(LOCOMOTION)
    monitor = locomotion['terrain_monitor']['ros__parameters']
    fsm = locomotion['climb_fsm']['ros__parameters']

    expected_height = p['axle_height'] + p['body_height'] / 2.0
    if abs(monitor['tof_mount_height'] - expected_height) > 1e-6:
        print(f'[FAIL] tof_mount_height is {monitor["tof_mount_height"]} but '
              f'axle_height + body_height/2 = {expected_height:.4f}. The ToF '
              f'thresholds are calibrated against a ride height the robot does '
              f'not have.')
        problems += 1
    else:
        print(f'[ ok ] tof_mount_height {expected_height:.4f} m follows from '
              f'axle_height + body_height/2')

    expected_spot = monitor['tof_mount_height'] / math.tan(monitor['tof_tilt'])
    if abs(fsm['tof_spot_ahead'] - expected_spot) > 0.005:
        print(f'[FAIL] climb_fsm.tof_spot_ahead is {fsm["tof_spot_ahead"]} but '
              f'the mounting geometry gives {expected_spot:.4f}. The descent '
              f'creep distance is computed from this, so the robot would '
              f'commit to a stair edge at the wrong moment.')
        problems += 1
    else:
        print(f'[ ok ] tof_spot_ahead {expected_spot:.4f} m follows from the '
              f'mount height and tilt')

    expected_axle = (p['cluster_circumradius'] * math.cos(math.pi / 3)
                     + p['sub_wheel_radius'])
    if abs(p['axle_height'] - expected_axle) > 1e-4:
        print(f'[FAIL] axle_height is {p["axle_height"]} but a carrier '
              f'straddling two sub-wheels sits at {expected_axle:.4f}')
        problems += 1
    else:
        print(f'[ ok ] axle_height {expected_axle:.4f} m follows from the '
              f'carrier straddle position')

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--verbose', action='store_true',
                        help='also list parameters left on their defaults')
    args = parser.parse_args()

    nodes: Dict[str, Tuple[str, Set[str]]] = {}
    for path in glob.glob(os.path.join(ROOT, 'src/r2d2_*/r2d2_*/*.py')):
        name, params = declared_parameters(path)
        if name and params:
            nodes[name] = (os.path.relpath(path, ROOT), params)

    configs: Dict[str, Dict[str, Set[str]]] = {}
    for path in glob.glob(os.path.join(ROOT, 'src/r2d2_*/config/*.yaml')):
        configs[os.path.relpath(path, ROOT)] = yaml_sections(path)

    print(f'{len(nodes)} nodes declaring parameters, '
          f'{len(configs)} parameter files\n')

    problems = 0
    covered_nodes: Set[str] = set()

    for config_path, sections in sorted(configs.items()):
        for section, keys in sorted(sections.items()):
            if section in EXTERNAL_SECTIONS:
                continue
            if section not in nodes:
                print(f'[FAIL] {config_path}: section "{section}" does not '
                      f'match any node in this repo, so it will never load')
                problems += 1
                continue

            source, declared = nodes[section]
            covered_nodes.add(section)

            # A YAML key the node never declares is silently ignored at runtime.
            unknown = keys - declared - {'use_sim_time'}
            if unknown:
                for key in sorted(unknown):
                    close = _closest(key, declared)
                    hint = f' (did you mean "{close}"?)' if close else ''
                    print(f'[FAIL] {config_path} sets {section}.{key} but '
                          f'{source} never declares it{hint}')
                    problems += 1

            missing = declared - keys
            if args.verbose and missing:
                print(f'[note] {section}: {len(missing)} parameter(s) left on '
                      f'their code defaults: {", ".join(sorted(missing))}')

            if not unknown:
                print(f'[ ok ] {section}: {len(keys & declared)} of '
                      f'{len(declared)} parameters set from {config_path}')

    uncovered = set(nodes) - covered_nodes
    if uncovered and args.verbose:
        print()
        for name in sorted(uncovered):
            print(f'[note] {name} has no parameter file; every parameter uses '
                  f'its code default')

    print()
    problems += check_mirrors()

    print()
    if problems:
        print(f'{problems} problem(s) found')
    else:
        print('every parameter file matches the node it configures')
    return 1 if problems else 0


def _closest(word: str, candidates: Set[str]) -> str:
    """Nearest candidate by edit distance, for typo hints."""
    best, best_distance = '', 3
    for candidate in candidates:
        distance = _levenshtein(word, candidate)
        if distance < best_distance:
            best, best_distance = candidate, distance
    return best


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        current = [i + 1]
        for j, cb in enumerate(b):
            current.append(min(previous[j + 1] + 1, current[j] + 1,
                               previous[j] + (ca != cb)))
        previous = current
    return previous[-1]


if __name__ == '__main__':
    raise SystemExit(main())
