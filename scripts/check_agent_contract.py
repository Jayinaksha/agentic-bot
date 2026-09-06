#!/usr/bin/env python3
"""
Check that everything naming a tool by string agrees with the tools that exist.

    python3 scripts/check_agent_contract.py
    python3 scripts/check_agent_contract.py --verbose

Why this exists
---------------
The MCP server is the only place a tool is defined, but three other places name
tools as bare strings, and none of them fails loudly when a name goes stale:

  * the agent's system prompt tells the model how to use specific tools. Rename
    one and the model is briefed on a tool that does not exist, and told nothing
    about the one that replaced it.
  * `_READ_ONLY_TOOLS` in agent.py decides what a --dry-run may still execute.
    A movement tool missing from that set is merely conservative; a renamed
    *read* tool silently stops being allowed, and worse, if a movement tool were
    ever added under a name already in the set, a dry run would move the robot.
  * tool docstrings cross-reference each other ("use explore() to drive around"),
    which is how the model discovers what to reach for next.

All three are prose or literals that no import resolves, so nothing catches the
drift. This does, by reading the tool names straight out of the decorators.

Tool names are taken from the AST rather than by importing the server, so this
runs without the MCP SDK, without ROS, and without any of the optional backends.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from typing import Dict, List, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))

SERVER = os.path.join(ROOT, 'src/r2d2_mcp/r2d2_mcp/server.py')
AGENT = os.path.join(ROOT, 'src/r2d2_mcp/r2d2_mcp/agent.py')
WORKFLOW = os.path.join(ROOT, '.github/workflows/tests.yml')

# Words that look like a call but are not tools: Python builtins and methods
# that appear in tool prose.
NOT_TOOLS = {
    'print', 'len', 'range', 'str', 'int', 'float', 'dict', 'list', 'set',
    'get', 'append', 'format', 'json', 'round', 'min', 'max', 'abs',
}

# snake_case words that legitimately appear in the system prompt as prose or as
# field names rather than as tool names. Anything not listed here and not a tool
# is treated as a stale reference, so this list stays deliberately short.
PROMPT_VOCABULARY: Set[str] = set()


def registered_tools(path: str) -> Tuple[Set[str], Dict[str, str]]:
    """Tool names and their model-facing text, from the @mcp.tool() decorators.

    The text is not just the docstring. A tool's return values carry prose the
    model reads too - "Try explore() to drive around and look" is returned by
    find_object when it finds nothing - and a stale tool name there misleads the
    model exactly as much as one in the description. So every string literal in
    the function body is collected.
    """
    tree = ast.parse(open(path).read())
    names: Set[str] = set()
    docs: Dict[str, str] = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            attribute = call.func if call else decorator
            if isinstance(attribute, ast.Attribute) and attribute.attr == 'tool':
                # An explicit name= overrides the function name.
                explicit = None
                if call:
                    for keyword in call.keywords:
                        if keyword.arg == 'name' and isinstance(
                                keyword.value, ast.Constant):
                            explicit = keyword.value.value
                name = explicit or node.name
                names.add(name)
                literals = [n.value for n in ast.walk(node)
                            if isinstance(n, ast.Constant)
                            and isinstance(n.value, str)]
                docs[name] = '\n'.join(literals)
    return names, docs


def string_constant(path: str, variable: str) -> str:
    """A module-level string assignment, without importing the module."""
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Name) and target.id == variable
                        and isinstance(node.value, ast.Constant)):
                    return node.value.value
    return ''


def name_set(path: str, variable: str) -> Set[str]:
    """A module-level frozenset/set of string literals."""
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Name) and target.id == variable):
                continue
            value = node.value
            if isinstance(value, ast.Call) and value.args:
                value = value.args[0]
            if isinstance(value, (ast.Set, ast.List, ast.Tuple)):
                return {e.value for e in value.elts
                        if isinstance(e, ast.Constant)}
    return set()


def called_names(text: str) -> Set[str]:
    """Names written as a call: `name()`. Unambiguously a tool reference.

    Backticked names are deliberately NOT treated as tool references. Tool
    descriptions backtick their own parameters and result fields too - `floor`,
    `standoff`, `grounding_quality` - and flagging those as missing tools is
    noise that would train a reader to ignore this check.
    """
    return {m.group(1) for m in
            re.finditer(r'\b([a-z_][a-z0-9_]{2,})\(\)', text)} - NOT_TOOLS


def mentioned_tools(text: str, known: Set[str]) -> Set[str]:
    """Which of the known tools this text talks about, in any form."""
    return {name for name in known
            if re.search(rf'\b{re.escape(name)}\b', text)}


def tool_shaped_names(text: str) -> Set[str]:
    """snake_case identifiers, which is the shape every tool name here has.

    The system prompt names tools bare - "find_object searches everything" -
    with no parentheses and no backticks, so the call-syntax scan alone misses a
    rename in exactly the place it matters most. Requiring an underscore keeps
    ordinary prose out: the words in this prompt that carry one are precisely
    the multi-word tool names.
    """
    return {m.group(0) for m in
            re.finditer(r'\b[a-z]+(?:_[a-z]+)+\b', text)} - NOT_TOOLS


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    tools, docs = registered_tools(SERVER)
    if not tools:
        print('::error::no @mcp.tool() decorators found in server.py; has the '
              'decorator been renamed?')
        return 1

    print(f'{len(tools)} tools registered in '
          f'{os.path.relpath(SERVER, ROOT)}\n')
    problems = 0

    # --- the system prompt --------------------------------------------------
    prompt = string_constant(AGENT, 'SYSTEM_PROMPT')
    if not prompt:
        print('::error::could not read SYSTEM_PROMPT from agent.py')
        problems += 1
    else:
        mentioned = mentioned_tools(prompt, tools)
        stale = sorted((called_names(prompt) | tool_shaped_names(prompt))
                       - tools - PROMPT_VOCABULARY)
        for name in stale:
            print(f'[FAIL] the system prompt briefs the model on "{name}", '
                  f'which is not a registered tool')
            problems += 1
        if not stale:
            covered = sorted(mentioned & tools)
            print(f'[ ok ] system prompt names {len(covered)} tools, all real')
            if args.verbose:
                print(f'       {", ".join(covered)}')
                unmentioned = sorted(tools - mentioned)
                if unmentioned:
                    print(f'[note] not mentioned in the prompt (discovered from '
                          f'their descriptions instead): '
                          f'{", ".join(unmentioned)}')

    # --- dry-run safety list ------------------------------------------------
    read_only = name_set(AGENT, '_READ_ONLY_TOOLS')
    if not read_only:
        print('[FAIL] _READ_ONLY_TOOLS is empty or unreadable; a dry run would '
              'refuse every tool, including the harmless ones')
        problems += 1
    else:
        unknown = sorted(read_only - tools)
        for name in unknown:
            print(f'[FAIL] _READ_ONLY_TOOLS allows "{name}" during a dry run, '
                  f'but no such tool exists. If it was renamed, the tool that '
                  f'replaced it is now blocked; if a movement tool ever takes '
                  f'this name, a dry run would move the robot')
            problems += 1
        if not unknown:
            print(f'[ ok ] all {len(read_only)} dry-run-allowed tools exist')

        # Anything that moves the robot must not be in the set.
        moving = {t for t in tools if any(
            verb in t for verb in ('navigate', 'climb', 'dock', 'explore',
                                   'say', 'stop', 'reset'))}
        leaked = sorted(read_only & moving)
        for name in leaked:
            print(f'[FAIL] "{name}" acts on the robot but is listed as '
                  f'read-only, so --dry-run would execute it')
            problems += 1
        if not leaked:
            print(f'[ ok ] no acting tool is marked read-only')

    # --- cross-references between tool descriptions -------------------------
    dangling: List[Tuple[str, str]] = []
    for name, doc in docs.items():
        for other in called_names(doc) - tools:
            if other != name:
                dangling.append((name, other))
    for owner, target in dangling:
        print(f'[FAIL] the description of "{owner}" points the model at '
              f'"{target}", which does not exist')
        problems += 1
    if not dangling:
        print(f'[ ok ] tool descriptions cross-reference only real tools')

    # --- the CI step's own exemption list -----------------------------------
    if os.path.exists(WORKFLOW):
        text = open(WORKFLOW).read()
        match = re.search(r"NO_ARGS = \{([^}]*)\}", text)
        if match:
            no_args = {v.strip().strip("'\"")
                       for v in match.group(1).split(',') if v.strip()}
            unknown = sorted(no_args - tools)
            for name in unknown:
                print(f'[FAIL] the CI schema check exempts "{name}", which is '
                      f'not a tool; a real tool losing its schema could hide '
                      f'behind a stale exemption')
                problems += 1
            if not unknown:
                print(f'[ ok ] CI schema exemptions all name real tools')

    print()
    if problems:
        print(f'{problems} problem(s) found')
    else:
        print('every string naming a tool agrees with the tools that exist')
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
