#!/usr/bin/env python3
"""
The agent: an MCP client driving the robot through tool calls.

    python3 -m r2d2_mcp.agent                       # interactive
    python3 -m r2d2_mcp.agent "go to the kitchen and tell me what is on the table"
    python3 -m r2d2_mcp.agent --dry-run "..."       # plan without moving

This is the replacement for gpt_oss.py. The difference is not the model, it is
the shape of the loop:

    before   prompt -> one JSON blob containing an entire plan -> execute it
             open-loop, hoping the world matched the plan
    now      prompt -> tool call -> real outcome -> next tool call -> ...

which means the model finds out that the study door is shut, and can do
something about it, instead of driving into it and reporting success.

Everything the agent does is written to the ledger as an episode, including
failures, so the next run can recall what happened.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from r2d2_mcp.config import LlmConfig, RobotConfig
from r2d2_mcp.llm import LlmClient, LlmError, Turn, mcp_tools_to_openai

log = logging.getLogger('r2d2.agent')

SYSTEM_PROMPT = """You control a small indoor robot in a two-storey house.

The robot is a four-cluster tri-star rover: it drives like a differential-drive
robot and can climb an ordinary staircase by switching its wheel clusters into a
tumbling mode. It carries a 2D LiDAR, an IMU, four downward distance sensors and
one forward camera. It has no depth camera, so it knows the direction of things
it sees far better than their exact distance.

How to work:

- Start by checking the robot's state. Do not assume where it is.
- Prefer memory over looking. find_object searches everything the robot has ever
  seen and costs nothing; look() runs a vision model and costs seconds. Look when
  you arrive somewhere new or when memory does not have what you need.
- Navigation gets within about 20 cm. That is fine for "go to the kitchen". When
  you need to be square in front of something to read or inspect it, follow with
  dock_precisely.
- Tools tell you the truth about what happened. When one fails, read the reason
  and adapt: a blocked path may need a different route, a missing object may need
  exploring, poor localisation may need driving around before precision work.
- Stairs are slow and cost the robot its position estimate. Cross floors when the
  task needs it, not speculatively. navigate_to_coordinates and
  navigate_to_object handle the whole route for you when the target is upstairs.
- If the terrain safety monitor has stopped the robot, find out why before
  clearing it. It latches when the robot has tipped too far.

Answer the person in plain language when you are done, saying what you actually
observed. If you could not finish, say what stopped you rather than describing
what you intended. Use say() only to talk to someone in the room with the robot.
"""


class Agent:
    """MCP client plus tool-calling loop."""

    def __init__(self, llm_config: Optional[LlmConfig] = None,
                 robot_config: Optional[RobotConfig] = None,
                 server_command: Optional[List[str]] = None,
                 dry_run: bool = False):
        self.llm = LlmClient(llm_config or LlmConfig())
        self.robot_config = robot_config or RobotConfig()
        self.server_command = server_command or [sys.executable, '-m', 'r2d2_mcp.server']
        self.dry_run = dry_run

        self._session = None
        self._stack: Optional[AsyncExitStack] = None
        self._tools: List[Dict[str, Any]] = []
        self._tool_names: List[str] = []

    # ------------------------------------------------------------ lifecycle

    async def connect(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        params = StdioServerParameters(
            command=self.server_command[0],
            args=self.server_command[1:],
            env=os.environ.copy(),
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(
            ClientSession(read, write))
        await self._session.initialize()

        listing = await self._session.list_tools()
        self._tools = mcp_tools_to_openai(listing.tools)
        self._tool_names = [t.name for t in listing.tools]
        log.info('connected to the robot: %d tools (%s)',
                 len(self._tool_names), ', '.join(self._tool_names))

    async def close(self) -> None:
        await self.llm.close()
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None

    # ----------------------------------------------------------------- loop

    async def run(self, instruction: str) -> Dict[str, Any]:
        """Carry out one instruction. Returns the outcome and the trace."""
        started = time.time()
        messages: List[Dict[str, Any]] = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': instruction},
        ]
        trace: List[Dict[str, Any]] = []

        for round_index in range(self.llm.config.max_rounds):
            try:
                turn = await self.llm.chat(messages, self._tools)
            except LlmError as exc:
                return self._finish(instruction, 'failure', started, trace,
                                    failure_reason=str(exc))
            except Exception as exc:              # noqa: BLE001 - network etc.
                return self._finish(instruction, 'failure', started, trace,
                                    failure_reason=f'model unreachable: {exc}')

            if not turn.wants_tools:
                return self._finish(instruction, 'success', started, trace,
                                    summary=turn.text)

            messages.append(_assistant_message(turn))

            for call in turn.tool_calls:
                result = await self._invoke(call.name, call.arguments)
                trace.append({'tool': call.name, 'args': call.arguments,
                              'result': _truncate(result)})
                messages.append({
                    'role': 'tool',
                    'tool_call_id': call.id,
                    'content': json.dumps(result)[:4000],
                })

        # Out of rounds. This is a real outcome, not an error to hide: report it
        # with the trace so the loop can be diagnosed.
        return self._finish(
            instruction, 'partial', started, trace,
            failure_reason=(f'gave up after {self.llm.config.max_rounds} tool '
                            f'rounds without reaching a conclusion'))

    async def _invoke(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call one tool, converting every failure into a readable result.

        A tool that raises must come back as text the model can reason about,
        not as an exception that ends the run. An unknown tool name, a bad
        argument, a robot that is not there - all of those are things the model
        can recover from if it is told.
        """
        if name not in self._tool_names:
            return {'error': f'there is no tool called "{name}". '
                             f'Available: {", ".join(self._tool_names)}'}

        if self.dry_run and name not in _READ_ONLY_TOOLS:
            return {'dry_run': True,
                    'note': f'{name} would have been called with {arguments}, '
                            f'but this is a dry run and the robot did not move'}

        log.info('-> %s(%s)', name, _compact(arguments))
        try:
            response = await self._session.call_tool(name, arguments)
        except Exception as exc:                  # noqa: BLE001
            log.warning('   %s failed: %s', name, exc)
            return {'error': f'{name} failed: {exc}'}

        payload = _unpack(response)
        log.info('   %s', _compact(payload))
        return payload

    def _finish(self, instruction: str, outcome: str, started: float,
                trace: List[Dict[str, Any]],
                summary: Optional[str] = None,
                failure_reason: Optional[str] = None) -> Dict[str, Any]:
        return {
            'instruction': instruction,
            'outcome': outcome,
            'summary': summary,
            'failure_reason': failure_reason,
            'tool_calls': trace,
            'rounds': len(trace),
            'duration_s': round(time.time() - started, 1),
        }


# Tools that only read. A dry run may call these, since they neither move the
# robot nor spend inference budget in a way that matters.
_READ_ONLY_TOOLS = frozenset({
    'get_state', 'find_object', 'list_known_objects', 'recall',
})


def _assistant_message(turn: Turn) -> Dict[str, Any]:
    """Rebuild the assistant turn for the next request.

    The raw message is reused when the endpoint gave one, so a model's own
    formatting survives the round trip; otherwise it is reconstructed from the
    parsed calls, which is what the content-fallback path needs.
    """
    if turn.raw_message.get('tool_calls'):
        return turn.raw_message
    return {
        'role': 'assistant',
        'content': turn.text or None,
        'tool_calls': [{
            'id': call.id,
            'type': 'function',
            'function': {'name': call.name,
                         'arguments': json.dumps(call.arguments)},
        } for call in turn.tool_calls],
    }


def _unpack(response) -> Dict[str, Any]:
    """MCP call result to a plain dict.

    The SDK renamed `structuredContent` to `structured_content` in 2.0. Reading
    only the old name silently falls through to re-parsing the text content,
    which mostly works and occasionally does not - so both are tried.
    """
    for attribute in ('structured_content', 'structuredContent'):
        structured = getattr(response, attribute, None)
        if isinstance(structured, dict):
            return structured

    parts = []
    for item in getattr(response, 'content', []) or []:
        text = getattr(item, 'text', None)
        if text:
            parts.append(text)
    joined = '\n'.join(parts)
    if not joined:
        return {'ok': True}
    try:
        value = json.loads(joined)
    except json.JSONDecodeError:
        return {'text': joined}
    return value if isinstance(value, dict) else {'value': value}


def _compact(value: Any, limit: int = 220) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + '...'


def _truncate(value: Dict[str, Any], limit: int = 800) -> Dict[str, Any]:
    text = json.dumps(value, default=str)
    if len(text) <= limit:
        return value
    return {'truncated': text[:limit]}


async def _record_episode(result: Dict[str, Any],
                          config: RobotConfig) -> None:
    """Write the episode to memory so later runs can recall it."""
    if not config.memory_enabled:
        return
    try:
        from r2d2_memory.ledger import make_ledger
        ledger = make_ledger(enabled=True, servers=config.nats_url)
        await ledger.connect()
        await ledger.append('episode', {
            'instruction': result['instruction'],
            'outcome': result['outcome'],
            'summary': result.get('summary'),
            'failure_reason': result.get('failure_reason'),
            'tool_calls': result.get('tool_calls', []),
            'duration_s': result.get('duration_s'),
        })
        await ledger.close()
    except Exception as exc:                      # noqa: BLE001
        log.warning('episode not recorded (%s); the run itself was unaffected', exc)


async def _main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format='%(levelname)s %(name)s: %(message)s')

    llm_config = LlmConfig()
    if args.model:
        llm_config.model = args.model
    if args.base_url:
        llm_config.base_url = args.base_url.rstrip('/')

    log.info('planner: %s', llm_config.describe())
    if args.dry_run:
        log.info('dry run: the robot will not move')

    robot_config = RobotConfig()
    agent = Agent(llm_config, robot_config, dry_run=args.dry_run)

    try:
        await agent.connect()
    except Exception as exc:                      # noqa: BLE001
        print(f'could not start the robot MCP server: {exc}', file=sys.stderr)
        print('Check that the navigation stack is running and that '
              '"pip install mcp" has been done.', file=sys.stderr)
        return 1

    try:
        if args.instruction:
            result = await agent.run(' '.join(args.instruction))
            _print_result(result)
            await _record_episode(result, robot_config)
            return 0 if result['outcome'] == 'success' else 2

        print('R2D2 agent. Type an instruction, or "quit".\n')
        while True:
            try:
                line = input('> ').strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.lower() in ('quit', 'exit'):
                break
            result = await agent.run(line)
            _print_result(result)
            await _record_episode(result, robot_config)
        return 0
    finally:
        await agent.close()


def _print_result(result: Dict[str, Any]) -> None:
    print()
    if result['outcome'] == 'success':
        print(result.get('summary') or '(done, but the model said nothing)')
    else:
        print(f'[{result["outcome"]}] {result.get("failure_reason") or ""}')
        if result.get('summary'):
            print(result['summary'])
    print(f'\n({result["rounds"]} tool calls, {result["duration_s"]} s)')


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Drive the robot with a natural-language instruction.')
    parser.add_argument('instruction', nargs='*',
                        help='what to do; omit for an interactive session')
    parser.add_argument('--model', help='override R2D2_LLM_MODEL')
    parser.add_argument('--base-url', help='override R2D2_LLM_BASE_URL')
    parser.add_argument('--dry-run', action='store_true',
                        help='plan and query, but never move the robot')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == '__main__':
    raise SystemExit(main())
