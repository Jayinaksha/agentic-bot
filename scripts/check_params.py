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
"""

from __future__ import annotations

import argparse
import ast
import glob
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
