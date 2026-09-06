#!/usr/bin/env python3
"""
Check the Nav2 plugin names are consistently styled for one ROS distro.

    python3 scripts/check_nav2_plugins.py
    python3 scripts/check_nav2_plugins.py --distro humble

Nav2 standardised plugin naming on "::" in Jazzy (PR #4220). Humble and Iron
use "/". The names are not aliased, so a plugin named in the wrong style does
not warn - it fails to load, and the server comes up without it. A half-finished
migration is the worst case: some plugins load, some do not, and the stack limps
in a way that looks like a tuning problem rather than a configuration error.

This checks every `plugin:` value across the Nav2 and AMCL configuration and
reports anything that disagrees with the declared distro style.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    import yaml
except ImportError:
    sys.exit('PyYAML required: pip install pyyaml')

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

CONFIGS = [
    'src/r2d2_navigation/config/nav2_house.yaml',
    'src/r2d2_localization/config/slam_toolbox.yaml',
]

# Keys whose values name a pluginlib class and therefore follow the style.
PLUGIN_KEYS = {'plugin', 'robot_model_type'}

STYLES = {
    'jazzy': '::',     # Jazzy and newer
    'rolling': '::',
    'humble': '/',     # Humble and Iron
    'iron': '/',
}


def walk(node, path=''):
    if isinstance(node, dict):
        for key, value in node.items():
            here = f'{path}.{key}' if path else key
            if key in PLUGIN_KEYS and isinstance(value, str):
                yield here, value
            else:
                yield from walk(value, here)
    elif isinstance(node, list):
        for item in node:
            yield from walk(item, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--distro', default='jazzy', choices=sorted(STYLES),
                        help='the ROS distro these configs target (default jazzy)')
    args = parser.parse_args()

    expected = STYLES[args.distro]
    other = '/' if expected == '::' else '::'

    print(f'Nav2 plugin naming, targeting {args.distro} (expects "{expected}")\n')

    problems = 0
    checked = 0
    for relative in CONFIGS:
        path = os.path.join(ROOT, relative)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            data = yaml.safe_load(fh)
        for where, plugin in walk(data):
            # slam_toolbox's solver is not a Nav2 plugin and has always used "::".
            if plugin.startswith('solver_plugins'):
                continue
            checked += 1
            if expected not in plugin:
                problems += 1
                fixed = plugin.replace(other, expected)
                print(f'[FAIL] {relative}\n'
                      f'       {where} = "{plugin}"\n'
                      f'       is {"Humble/Iron" if other == "/" else "Jazzy"} '
                      f'style; {args.distro} needs "{fixed}"')

    print()
    if problems:
        print(f'{problems} of {checked} plugin names are in the wrong style. '
              f'They will not warn - they will simply fail to load.')
        return 1
    print(f'all {checked} plugin names use "{expected}", consistent with '
          f'{args.distro}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
