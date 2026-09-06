#!/usr/bin/env python3
"""
Check the hand-written SQL in store.py against the schema it queries.

    python3 scripts/check_sql.py
    pip install pglast     # needs a real PostgreSQL grammar

Why this exists
---------------
The memory layer's queries are written by hand and reference about sixty columns
across seven tables. A misspelled column does not fail at import, at build, or
at connection - it fails on the first query that touches it, in a robot that has
been driving happily for ten minutes. The schema is right there in the same
package, so the mismatch is checkable without a database.

Two things are checked:

  parse     every embedded statement is valid PostgreSQL, using pglast (the real
            server grammar, not a regex approximation)
  columns   every column referenced exists somewhere in schema.sql

Runs without Postgres. If pglast is not installed it says so and skips rather
than silently passing, because a check that quietly does nothing is worse than
no check.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from typing import Dict, List, Set, Tuple

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SCHEMA = 'src/r2d2_memory/r2d2_memory/schema.sql'
SOURCES = ['src/r2d2_memory/r2d2_memory/store.py']

# `excluded` is PostgreSQL's pseudo-table inside ON CONFLICT DO UPDATE; it is not
# a schema column and never will be.
PSEUDO_COLUMNS = {'excluded'}

# A string only counts as SQL if it has the shape of a statement. Without this a
# docstring beginning "Update an object with a new detection." is read as an
# UPDATE and reported as a syntax error.
SQL_SHAPE = re.compile(
    r'^\s*(SELECT|INSERT|UPDATE|DELETE|TRUNCATE)\b.*\b(FROM|INTO|SET|WHERE|VALUES)\b',
    re.I | re.S)


def schema_columns(path: str) -> Tuple[Dict[str, Set[str]], Set[str]]:
    import pglast
    from pglast import ast as pgast

    relations: Dict[str, Set[str]] = {}
    for statement in pglast.parse_sql(open(path).read()):
        node = statement.stmt
        if isinstance(node, pgast.CreateStmt):
            columns = {element.colname
                       for element in (node.tableElts or [])
                       if isinstance(element, pgast.ColumnDef)}
            relations[node.relation.relname] = columns
        elif isinstance(node, pgast.ViewStmt):
            relations[node.view.relname] = set()

    every = set()
    for columns in relations.values():
        every |= columns
    return relations, every


def embedded_sql(path: str) -> List[Tuple[int, str]]:
    """String literals in a Python file that are actually SQL statements."""
    found = []
    for node in ast.walk(ast.parse(open(path).read())):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if SQL_SHAPE.match(node.value):
                found.append((node.lineno, node.value))
    return found


def referenced_columns(statement) -> Set[str]:
    from pglast import ast as pgast

    used: Set[str] = set()

    def walk(node):
        if isinstance(node, pgast.ColumnRef):
            for field in (node.fields or []):
                if isinstance(field, pgast.String):
                    used.add(field.sval)
        for slot in getattr(node, '__slots__', ()) or ():
            value = getattr(node, slot, None)
            if isinstance(value, pgast.Node):
                walk(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, pgast.Node):
                        walk(item)

    walk(statement)
    return used


def main() -> int:
    try:
        import pglast
    except ImportError:
        print('pglast is not installed, so the SQL cannot be checked against a '
              'real PostgreSQL grammar.\n'
              '  pip install pglast\n'
              'Skipping rather than passing silently.', file=sys.stderr)
        return 0

    relations, every_column = schema_columns(os.path.join(ROOT, SCHEMA))
    print(f'schema: {len(relations)} relations, {len(every_column)} columns\n')

    problems = 0
    checked = 0

    for relative in SOURCES:
        path = os.path.join(ROOT, relative)
        for lineno, sql in embedded_sql(path):
            checked += 1
            try:
                parsed = pglast.parse_sql(sql)
            except Exception as exc:            # noqa: BLE001 - parser errors vary
                problems += 1
                print(f'[FAIL] {relative}:{lineno} is not valid PostgreSQL\n'
                      f'       {str(exc)[:120]}')
                continue

            # Aliases introduced by the query itself are legitimate names.
            aliases = set(re.findall(r'\bAS\s+(\w+)', sql, re.I))
            aliases |= set(re.findall(r'\bFROM\s+(\w+)\s+(\w+)\b', sql, re.I) and
                           [m[1] for m in re.findall(r'\bFROM\s+(\w+)\s+(\w+)\b',
                                                     sql, re.I)])
            aliases |= {m[1] for m in re.findall(r'\bJOIN\s+(\w+)\s+(\w+)\b',
                                                 sql, re.I)}
            known = every_column | set(relations) | aliases | PSEUDO_COLUMNS

            for statement in parsed:
                unknown = referenced_columns(statement.stmt) - known
                if unknown:
                    problems += 1
                    print(f'[FAIL] {relative}:{lineno} references column(s) not '
                          f'in the schema: {sorted(unknown)}\n'
                          f'       {sql.strip().splitlines()[0][:80]}')

    print()
    if problems:
        print(f'{problems} problem(s) across {checked} statements')
        return 1
    print(f'all {checked} statements parse and reference only schema columns')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
