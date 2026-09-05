#!/usr/bin/env python3
"""Unit tests for tool-call parsing and the agent's result handling.

    python3 -m pytest src/r2d2_mcp/test/test_llm.py

This covers the layer that replaced the old repair_json/extract_json machinery.
The point of moving to tool calling is that a malformed response is *detected*
rather than repaired into something that might mean the wrong thing, so most of
these tests are about refusing to invent a call that was never made.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from r2d2_mcp.llm import (  # noqa: E402
    Turn, _json_objects, _parse_arguments, _parse_tool_calls,
    mcp_tools_to_openai)


def _message(**kwargs):
    return {'role': 'assistant', **kwargs}


# ------------------------------------------------------------ argument shapes

def test_dict_arguments_pass_through():
    assert _parse_arguments({'x': 1.0}) == {'x': 1.0}


def test_json_string_arguments_are_parsed():
    """The common case: the API delivers arguments as a JSON string."""
    assert _parse_arguments('{"x": 1.5, "name": "chair"}') == {'x': 1.5,
                                                               'name': 'chair'}


def test_empty_arguments_become_an_empty_dict():
    assert _parse_arguments('') == {}
    assert _parse_arguments('{}') == {}
    assert _parse_arguments(None) == {}


def test_malformed_arguments_do_not_raise():
    assert _parse_arguments('{"x": ') == {}


def test_non_object_arguments_are_rejected():
    """A bare array would be silently mismapped onto keyword parameters."""
    assert _parse_arguments('[1, 2, 3]') == {}


# ------------------------------------------------------------- tool calls

def test_standard_tool_call():
    calls = _parse_tool_calls(_message(tool_calls=[{
        'id': 'call_1',
        'function': {'name': 'navigate_to_room',
                     'arguments': '{"name": "kitchen"}'}}]))
    assert len(calls) == 1
    assert calls[0].name == 'navigate_to_room'
    assert calls[0].arguments == {'name': 'kitchen'}
    assert calls[0].id == 'call_1'


def test_multiple_tool_calls_are_all_returned():
    calls = _parse_tool_calls(_message(tool_calls=[
        {'id': 'a', 'function': {'name': 'get_state', 'arguments': '{}'}},
        {'id': 'b', 'function': {'name': 'look', 'arguments': '{}'}}]))
    assert [c.name for c in calls] == ['get_state', 'look']


def test_missing_ids_are_synthesised():
    """The chat API requires a tool_call_id on the reply; a missing one would
    make the next request invalid."""
    calls = _parse_tool_calls(_message(tool_calls=[
        {'function': {'name': 'get_state', 'arguments': '{}'}}]))
    assert calls[0].id


def test_nameless_calls_are_dropped():
    calls = _parse_tool_calls(_message(tool_calls=[
        {'id': 'a', 'function': {'arguments': '{}'}},
        {'id': 'b', 'function': {'name': 'stop', 'arguments': '{}'}}]))
    assert [c.name for c in calls] == ['stop']


def test_plain_text_is_not_a_tool_call():
    assert _parse_tool_calls(_message(
        content='The kitchen table has a mug and a laptop on it.')) == []


def test_prose_containing_braces_is_not_a_tool_call():
    """The failure mode the old regex-based parser had: any braces looked like
    a plan."""
    assert _parse_tool_calls(_message(
        content='I considered {going upstairs} but the door is shut.')) == []


def test_a_json_object_without_a_name_is_not_a_tool_call():
    assert _parse_tool_calls(_message(content='{"x": 1, "y": 2}')) == []


def test_a_tool_call_in_the_content_is_recovered():
    """Some open models emit the call in the message body under load."""
    calls = _parse_tool_calls(_message(
        content='I will check first. {"name": "get_state", "arguments": {}}'))
    assert [c.name for c in calls] == ['get_state']


def test_the_tool_calls_field_wins_over_the_content():
    calls = _parse_tool_calls(_message(
        content='{"name": "stop", "arguments": {}}',
        tool_calls=[{'id': '1', 'function': {'name': 'look',
                                             'arguments': '{}'}}]))
    assert [c.name for c in calls] == ['look']


def test_content_recovery_handles_a_stringified_arguments_field():
    calls = _parse_tool_calls(_message(
        content='{"name": "say", "arguments": "{\\"text\\": \\"hello\\"}"}'))
    assert calls[0].arguments == {'text': 'hello'}


# ------------------------------------------------------- json object scanning

def test_finds_several_objects():
    assert _json_objects('a {"a": 1} b {"b": 2}') == [{'a': 1}, {'b': 2}]


def test_handles_nesting():
    assert _json_objects('{"a": {"b": {"c": 1}}}') == [{'a': {'b': {'c': 1}}}]


def test_braces_inside_strings_do_not_confuse_the_scanner():
    """A brace inside a quoted string is text, not structure."""
    assert _json_objects('{"note": "use {this} form"}') == [
        {'note': 'use {this} form'}]


def test_escaped_quotes_inside_strings_are_handled():
    assert _json_objects(r'{"note": "he said \"hi\" {ok}"}') == [
        {'note': 'he said "hi" {ok}'}]


def test_unbalanced_braces_yield_nothing():
    assert _json_objects('{"a": 1') == []


def test_stray_closing_brace_does_not_break_later_parsing():
    assert _json_objects('} {"a": 1}') == [{'a': 1}]


# ------------------------------------------------------------ tool conversion

class _FakeTool:
    def __init__(self, name, description, schema):
        self.name = name
        self.description = description
        self.inputSchema = schema


def test_mcp_tools_convert_to_openai_functions():
    schema = {'type': 'object',
              'properties': {'name': {'type': 'string'}},
              'required': ['name']}
    converted = mcp_tools_to_openai([
        _FakeTool('navigate_to_room', '  Drive to a named room.  ', schema)])
    assert converted[0]['type'] == 'function'
    assert converted[0]['function']['name'] == 'navigate_to_room'
    assert converted[0]['function']['description'] == 'Drive to a named room.'
    # The schema passes through untouched: this correspondence is the reason
    # MCP is worth using, rather than hand-writing the same schema twice.
    assert converted[0]['function']['parameters'] is schema


def test_a_tool_with_no_schema_still_converts():
    converted = mcp_tools_to_openai([_FakeTool('stop', 'Stop.', None)])
    assert converted[0]['function']['parameters'] == {'type': 'object',
                                                      'properties': {}}


# ------------------------------------------------------------------- turns

def test_a_turn_with_calls_wants_tools():
    turn = Turn(text='', tool_calls=_parse_tool_calls(_message(tool_calls=[
        {'id': '1', 'function': {'name': 'stop', 'arguments': '{}'}}])))
    assert turn.wants_tools


def test_a_text_only_turn_does_not():
    assert not Turn(text='All done.').wants_tools
