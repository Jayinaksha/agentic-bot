#!/usr/bin/env python3
"""Unit tests for the agent loop, with a stub model and a stub robot.

    python3 -m pytest src/r2d2_mcp/test/test_agent.py

These exercise the control flow that matters: that a failed tool result reaches
the model rather than ending the run, that an unknown tool name is reported
instead of raising, and that running out of rounds is an honest "partial"
outcome rather than a silent success.
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from r2d2_mcp.agent import Agent, _READ_ONLY_TOOLS, _assistant_message, _unpack  # noqa: E402
from r2d2_mcp.config import LlmConfig  # noqa: E402
from r2d2_mcp.llm import ToolCall, Turn  # noqa: E402


class StubLlm:
    """Replays a scripted list of turns."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.config = LlmConfig(max_rounds=6)
        self.seen = []

    async def chat(self, messages, tools=None):
        self.seen.append(messages[-1])
        return self.turns.pop(0) if self.turns else Turn(text='done')

    async def close(self):
        return None


class StubSession:
    """Stands in for an MCP client session."""

    def __init__(self, responses=None, raises=None):
        self.responses = responses or {}
        self.raises = raises or set()
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name in self.raises:
            raise RuntimeError('the robot bridge is not running')
        payload = self.responses.get(name, {'success': True})

        class _Result:
            structuredContent = payload
            content = []
        return _Result()


def build_agent(turns, session, tool_names=None, dry_run=False):
    agent = Agent(dry_run=dry_run)
    agent.llm = StubLlm(turns)
    agent._session = session
    agent._tool_names = tool_names or ['get_state', 'navigate_to_room', 'look', 'say']
    agent._tools = []
    return agent


def call(name, arguments=None, call_id='c1'):
    return Turn(tool_calls=[ToolCall(id=call_id, name=name,
                                     arguments=arguments or {})])


# ---------------------------------------------------------------- happy path

def test_a_text_only_reply_finishes_immediately():
    agent = build_agent([Turn(text='The table is clear.')], StubSession())
    result = asyncio.run(agent.run('what is on the table?'))
    assert result['outcome'] == 'success'
    assert result['summary'] == 'The table is clear.'
    assert result['rounds'] == 0


def test_a_tool_call_is_executed_and_the_loop_continues():
    session = StubSession({'get_state': {'floor': 0}})
    agent = build_agent([call('get_state'), Turn(text='We are downstairs.')],
                        session)
    result = asyncio.run(agent.run('which floor?'))
    assert session.calls == [('get_state', {})]
    assert result['outcome'] == 'success'
    assert result['tool_calls'][0]['result'] == {'floor': 0}


def test_several_calls_in_one_turn_all_run():
    session = StubSession()
    turn = Turn(tool_calls=[
        ToolCall(id='a', name='get_state', arguments={}),
        ToolCall(id='b', name='look', arguments={}),
    ])
    agent = build_agent([turn, Turn(text='done')], session)
    asyncio.run(agent.run('have a look'))
    assert [c[0] for c in session.calls] == ['get_state', 'look']


def test_tool_results_are_fed_back_to_the_model():
    """The whole point of the redesign: the model sees the real outcome."""
    session = StubSession({'navigate_to_room': {'success': False,
                                                'reason': 'the door is shut'}})
    agent = build_agent([call('navigate_to_room', {'name': 'study'}),
                         Turn(text='The study door is shut.')], session)
    asyncio.run(agent.run('go to the study'))
    last_seen = agent.llm.seen[-1]
    assert last_seen['role'] == 'tool'
    assert 'the door is shut' in last_seen['content']


# -------------------------------------------------------------- failure paths

def test_a_failing_tool_does_not_end_the_run():
    session = StubSession(raises={'navigate_to_room'})
    agent = build_agent([call('navigate_to_room', {'name': 'kitchen'}),
                         Turn(text='I could not move.')], session)
    result = asyncio.run(agent.run('go to the kitchen'))
    assert result['outcome'] == 'success'
    assert 'error' in result['tool_calls'][0]['result']


def test_an_unknown_tool_is_reported_not_raised():
    """A hallucinated tool name must come back as text the model can act on."""
    session = StubSession()
    agent = build_agent([call('teleport', {'to': 'kitchen'}),
                         Turn(text='I cannot do that.')], session)
    result = asyncio.run(agent.run('teleport to the kitchen'))
    assert session.calls == []
    error = result['tool_calls'][0]['result']['error']
    assert 'no tool called "teleport"' in error
    assert 'get_state' in error          # the available ones are listed


def test_running_out_of_rounds_is_partial_not_success():
    session = StubSession()
    agent = build_agent([call('get_state') for _ in range(20)], session)
    result = asyncio.run(agent.run('loop forever'))
    assert result['outcome'] == 'partial'
    assert 'gave up after' in result['failure_reason']
    assert result['rounds'] == agent.llm.config.max_rounds


def test_a_model_error_is_a_failure_with_a_reason():
    class BrokenLlm(StubLlm):
        async def chat(self, messages, tools=None):
            raise RuntimeError('connection refused')

    agent = build_agent([], StubSession())
    agent.llm = BrokenLlm([])
    result = asyncio.run(agent.run('go to the kitchen'))
    assert result['outcome'] == 'failure'
    assert 'connection refused' in result['failure_reason']


# ------------------------------------------------------------------- dry run

def test_dry_run_blocks_movement_tools():
    session = StubSession()
    agent = build_agent([call('navigate_to_room', {'name': 'kitchen'}),
                         Turn(text='That is the plan.')], session, dry_run=True)
    result = asyncio.run(agent.run('go to the kitchen'))
    assert session.calls == []
    assert result['tool_calls'][0]['result']['dry_run'] is True


def test_dry_run_still_allows_reads():
    session = StubSession({'get_state': {'floor': 1}})
    agent = build_agent([call('get_state'), Turn(text='Upstairs.')],
                        session, dry_run=True)
    asyncio.run(agent.run('which floor?'))
    assert session.calls == [('get_state', {})]


def test_read_only_set_covers_the_query_tools():
    assert {'get_state', 'find_object', 'recall'} <= _READ_ONLY_TOOLS
    assert 'navigate_to_room' not in _READ_ONLY_TOOLS
    assert 'climb_stairs' not in _READ_ONLY_TOOLS


# ------------------------------------------------------------------ plumbing

def test_assistant_message_is_reused_when_the_api_gave_one():
    raw = {'role': 'assistant', 'content': None,
           'tool_calls': [{'id': 'x', 'type': 'function',
                           'function': {'name': 'stop', 'arguments': '{}'}}]}
    assert _assistant_message(Turn(raw_message=raw)) is raw


def test_assistant_message_is_rebuilt_for_recovered_calls():
    """The content-fallback path has no raw tool_calls to reuse."""
    turn = Turn(text='calling now',
                tool_calls=[ToolCall(id='c', name='look', arguments={'q': 'x'})])
    message = _assistant_message(turn)
    assert message['tool_calls'][0]['function']['name'] == 'look'
    assert json.loads(
        message['tool_calls'][0]['function']['arguments']) == {'q': 'x'}


def test_unpack_prefers_structured_content_sdk_1():
    class _R:
        structuredContent = {'success': True, 'floor': 2}
        content = []
    assert _unpack(_R()) == {'success': True, 'floor': 2}


def test_unpack_prefers_structured_content_sdk_2():
    """SDK 2.0 renamed structuredContent to structured_content. Reading only the
    old name falls through to re-parsing the text content, which mostly works
    and occasionally does not."""
    class _R:
        structured_content = {'success': True, 'floor': 2}
        content = []
    assert _unpack(_R()) == {'success': True, 'floor': 2}


def test_unpack_parses_json_from_text_content():
    class _Item:
        text = '{"success": true}'

    class _R:
        structuredContent = None
        content = [_Item()]
    assert _unpack(_R()) == {'success': True}


def test_unpack_wraps_plain_text():
    class _Item:
        text = 'the kitchen is clear'

    class _R:
        structuredContent = None
        content = [_Item()]
    assert _unpack(_R()) == {'text': 'the kitchen is clear'}


def test_unpack_handles_an_empty_response():
    class _R:
        structuredContent = None
        content = []
    assert _unpack(_R()) == {'ok': True}
