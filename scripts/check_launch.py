#!/usr/bin/env python3
"""
Check that every file a launch file or xacro asks for will actually be installed.

    python3 scripts/check_launch.py

Why this exists
---------------
ROS packages reference their data through the install share directory, not the
source tree. A config file can therefore exist perfectly well in `src/`, be
referenced correctly by a launch file, and still be missing at run time because
its `setup.py` never globbed it into `data_files`. Nothing catches that: the
build succeeds, the launch file resolves the path, and the node dies on a
missing file the developer can see sitting right there in their editor.

Three references are followed:

  Node(package=, executable=)     against the package's console_scripts
  get_package_share_directory()   against what setup.py / CMakeLists install
  $(find pkg)/path in xacro       likewise

The "is it installed" half is the point. Checking only that the source file
exists would pass on exactly the bug this is looking for.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from typing import Dict, Set

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def entry_points() -> Dict[str, Set[str]]:
    """Executables each package installs, from either build system.

    Python packages declare them as console_scripts; C++ packages build them
    with add_executable and install them with install(TARGETS ...). Reading only
    the Python side is how a wrong C++ executable name survives - which is
    exactly what happened here: the launch file asked for
    `laser_scan_matcher_node` while the CMakeLists installs `laser_scan_matcher`.
    """
    out: Dict[str, Set[str]] = {}
    for setup in glob.glob(os.path.join(ROOT, 'src/*/setup.py')):
        package = os.path.basename(os.path.dirname(setup))
        out[package] = set(re.findall(r"'(\w+) = [\w.]+:main'", open(setup).read()))

    for cmake in glob.glob(os.path.join(ROOT, 'src/*/CMakeLists.txt')):
        package = os.path.basename(os.path.dirname(cmake))
        text = open(cmake).read()
        targets = set(re.findall(r'add_executable\(\s*(\w+)', text))
        # Only targets that are actually installed can be launched.
        installed_targets = set()
        for match in re.finditer(r'install\(\s*TARGETS\s+([^\)]+?)\s+DESTINATION',
                                 text, re.S):
            installed_targets.update(match.group(1).split())
        out.setdefault(package, set())
        out[package] |= (targets & installed_targets) if installed_targets else targets
    return out


def installed_dirs(package: str) -> Set[str]:
    """Top-level directories a package installs into its share directory."""
    directory = os.path.join(ROOT, 'src', package)
    installed: Set[str] = set()

    setup = os.path.join(directory, 'setup.py')
    if os.path.exists(setup):
        text = open(setup).read()
        # ('share/<pkg>/<sub>', glob('<sub>/*.ext'))
        for match in re.finditer(r"glob\(\s*(?:os\.path\.join\(\s*)?'([^']+)'", text):
            pattern = match.group(1)
            installed.add(pattern.split('/')[0])
        for match in re.finditer(r"os\.path\.join\('share', package_name, '(\w+)'\)", text):
            installed.add(match.group(1))

    cmake = os.path.join(directory, 'CMakeLists.txt')
    if os.path.exists(cmake):
        for match in re.finditer(r'install\(\s*DIRECTORY\s+([^\)]+?)\s*DESTINATION',
                                 open(cmake).read(), re.S):
            installed.update(match.group(1).split())

    return installed


def main() -> int:
    packages = entry_points()
    installs = {package: installed_dirs(package) for package in packages}

    print('launch and xacro resource references\n')
    problems = 0
    checked = 0

    # --- Node(package=, executable=) ------------------------------------
    for launch in sorted(glob.glob(os.path.join(ROOT, 'src/*/launch/*.py'))):
        relative = os.path.relpath(launch, ROOT)
        text = open(launch).read()
        for match in re.finditer(
                r"package\s*=\s*'(\w+)'\s*,\s*\n?\s*executable\s*=\s*'(\w+)'", text):
            package, executable = match.groups()
            if package not in packages:
                continue                      # an upstream package
            checked += 1
            if executable not in packages[package]:
                problems += 1
                print(f'[FAIL] {relative}\n'
                      f'       runs {package}/{executable}, which that package '
                      f'does not install\n'
                      f'       available: {sorted(packages[package]) or "none"}')

    # --- share-directory lookups ----------------------------------------
    for launch in sorted(glob.glob(os.path.join(ROOT, 'src/*/launch/*.py'))):
        relative = os.path.relpath(launch, ROOT)
        text = open(launch).read()
        for match in re.finditer(
                r"get_package_share_directory\('(\w+)'\)[^\n]*?\n?[^\n]*?"
                r"'(\w+)'\s*,\s*'([\w.\-]+)'", text):
            package, subdir, filename = match.groups()
            if package not in packages:
                continue
            checked += 1
            source = os.path.join(ROOT, 'src', package, subdir, filename)
            if not os.path.exists(source):
                problems += 1
                print(f'[FAIL] {relative}\n'
                      f'       wants {package}/{subdir}/{filename}, which does '
                      f'not exist in the source tree')
            elif subdir not in installs.get(package, set()):
                problems += 1
                print(f'[FAIL] {relative}\n'
                      f'       wants {package}/{subdir}/{filename}. The file '
                      f'exists, but {package} never installs the "{subdir}" '
                      f'directory, so it will be absent after colcon build.\n'
                      f'       installs: {sorted(installs.get(package, set())) or "nothing"}')

    # --- $(find pkg)/path inside xacro ----------------------------------
    for xacro in sorted(glob.glob(os.path.join(ROOT, 'src/*/urdf/*.xacro'))):
        relative = os.path.relpath(xacro, ROOT)
        for match in re.finditer(r"\$\(find (\w+)\)/([\w/\-]+\.\w+)",
                                 open(xacro).read()):
            package, path = match.groups()
            if package not in packages:
                continue
            checked += 1
            source = os.path.join(ROOT, 'src', package, path)
            subdir = path.split('/')[0]
            if not os.path.exists(source):
                problems += 1
                print(f'[FAIL] {relative}\n'
                      f'       references $(find {package})/{path}, which does '
                      f'not exist')
            elif subdir not in installs.get(package, set()):
                problems += 1
                print(f'[FAIL] {relative}\n'
                      f'       references $(find {package})/{path}. The file '
                      f'exists, but {package} never installs "{subdir}".')

    print()
    if problems:
        print(f'{problems} of {checked} references will not resolve at run time')
        return 1
    print(f'all {checked} references resolve, and every referenced directory is '
          f'installed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
