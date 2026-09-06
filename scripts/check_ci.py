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

Steps that only install dependencies are skipped, on the assumption that a
machine running this already has pytest and PyYAML.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List

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
    skipped = 0

    for job_name, job in jobs.items():
        print(f'\n=== {job_name} ===')
        for step in job.get('steps', []):
            script = step.get('run')
            name = step.get('name', '(unnamed)')
            if not script:
                continue                       # an action, not a shell step
            if 'pip install' in script:
                skipped += 1
                continue

            result = subprocess.run(['bash', '-c', script], cwd=ROOT,
                                    capture_output=True, text=True)
            if result.returncode == 0:
                print(f'  [ ok ] {name}')
            else:
                print(f'  [FAIL] {name}')
                failures.append(f'{job_name}: {name}')
                for stream in (result.stdout, result.stderr):
                    for line in stream.strip().splitlines()[-15:]:
                        print(f'         {line}')

    print()
    if skipped:
        print(f'({skipped} dependency-install step(s) skipped)')
    if failures:
        print(f'{len(failures)} step(s) failed:')
        for failure in failures:
            print(f'  {failure}')
        return 1
    print('every CI step passes')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
