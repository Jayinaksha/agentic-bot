#!/usr/bin/env python3
"""
Run the CI checks locally, exactly as the GitHub runner does.

    python3 scripts/check_ci.py
    python3 scripts/check_ci.py --job design-checks
    python3 scripts/check_ci.py --list

Reads .github/workflows/tests.yml and executes each step's `run:` block through
bash, in order. The point is to find a broken CI step here rather than after a
push: a workflow that only ever runs on GitHub has a slow, public feedback loop,
and a step with a shell typo looks exactly like a real failure until you read
the log.

Dependency installs are not run - this should not reach out to the network or
mutate the machine it is checking - but they are not simply ignored either.
Their package names are collected, and a step that then fails on one of those
missing imports is reported as SKIPPED rather than FAILED. Reporting it as a
failure would be worse than useless: it trains you to ignore a red result.
Install lines embedded inside a larger step are stripped so the rest of that
step still runs.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from typing import List, Set

try:
    import yaml
except ImportError:
    sys.exit('PyYAML required: pip install pyyaml')

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'tests.yml')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--job', action='append',
                        help='only run these jobs (repeatable)')
    parser.add_argument('--list', action='store_true',
                        help='list the jobs and steps without running them')
    parser.add_argument('--workflow', default=WORKFLOW)
    args = parser.parse_args()

    with open(args.workflow) as fh:
        workflow = yaml.safe_load(fh)

    jobs = workflow.get('jobs', {})
    if args.job:
        unknown = set(args.job) - set(jobs)
        if unknown:
            print(f'no such job: {", ".join(sorted(unknown))}', file=sys.stderr)
            print(f'available: {", ".join(jobs)}', file=sys.stderr)
            return 2
        jobs = {name: job for name, job in jobs.items() if name in args.job}

    if args.list:
        for name, job in jobs.items():
            print(f'{name}:')
            for step in job.get('steps', []):
                marker = 'run ' if step.get('run') else 'uses'
                print(f'  [{marker}] {step.get("name", step.get("uses", "?"))}')
        return 0

    failures: List[str] = []
    skipped: List[str] = []
    install_only = 0

    for job_name, job in jobs.items():
        print(f'\n=== {job_name} ===')
        deps: Set[str] = set()

        for step in job.get('steps', []):
            script = step.get('run')
            name = step.get('name', '(unnamed)')
            if not script:
                continue                       # an action, not a shell step

            deps |= _pip_packages(script)
            runnable = _strip_installs(script)
            if not runnable.strip():
                install_only += 1
                continue

            # Honour the step's own env block. Without this a step that sets
            # PYTHONPATH fails on an import that would resolve fine on the
            # runner, and the tool reports a problem that does not exist.
            environment = dict(os.environ)
            environment.update({k: str(v) for k, v in
                                (job.get('env') or {}).items()})
            environment.update({k: str(v) for k, v in
                                (step.get('env') or {}).items()})

            result = subprocess.run(['bash', '-c', runnable], cwd=ROOT,
                                    capture_output=True, text=True,
                                    env=environment)
            if result.returncode == 0:
                print(f'  [ ok ] {name}')
                continue

            missing = _missing_module(result.stdout + result.stderr, deps)
            if missing:
                print(f'  [skip] {name}')
                print(f'         needs "{missing}", which CI installs and this '
                      f'run does not')
                skipped.append(f'{job_name}: {name} (needs {missing})')
                continue

            print(f'  [FAIL] {name}')
            failures.append(f'{job_name}: {name}')
            for stream in (result.stdout, result.stderr):
                for line in stream.strip().splitlines()[-15:]:
                    print(f'         {line}')

    print()
    if install_only:
        print(f'({install_only} dependency-install step(s) not run)')
    if skipped:
        print(f'{len(skipped)} step(s) skipped for missing dependencies:')
        for entry in skipped:
            print(f'  {entry}')
    if failures:
        print(f'{len(failures)} step(s) failed:')
        for failure in failures:
            print(f'  {failure}')
        return 1
    print('every runnable CI step passes')
    return 0


def _pip_packages(script: str) -> Set[str]:
    """Package names a script installs, so a later import error can be
    attributed to a dependency CI would have provided."""
    found: Set[str] = set()
    for line in script.splitlines():
        if 'pip install' not in line:
            continue
        for token in line.split():
            if token.startswith('-') or token in ('pip', 'install', 'python',
                                                  '-m', 'python3'):
                continue
            # Strip quoting and any version specifier.
            name = token.strip('\'"').split('>')[0].split('<')[0].split('=')[0]
            name = name.split('[')[0]
            if name and name.replace('-', '').replace('_', '').isalnum():
                found.add(name)
    found.discard('pip')
    return found


def _strip_installs(script: str) -> str:
    """The script without its dependency-install lines."""
    return '\n'.join(line for line in script.splitlines()
                     if 'pip install' not in line)


def _missing_module(output: str, deps: Set[str]) -> str:
    """The dependency a failure is attributable to, if any.

    Two spellings, because Python reports the same condition differently
    depending on how the module was reached:

        import pyflakes        ModuleNotFoundError: No module named 'pyflakes'
        python -m pyflakes     /usr/bin/python: No module named pyflakes

    Matching only the quoted form meant every `python -m <tool>` step was
    reported as a genuine failure on a machine without that tool - which is the
    crying-wolf behaviour this function exists to prevent, arriving through a
    different door.
    """
    match = re.search(r"No module named '([\w.]+)'", output)
    if not match:
        match = re.search(r'No module named ([\w.]+)', output)
    if not match:
        return ''
    module = match.group(1).split('.')[0]
    for dep in deps:
        if module.lower().replace('_', '-') == dep.lower().replace('_', '-'):
            return dep
    return ''


if __name__ == '__main__':
    raise SystemExit(main())
