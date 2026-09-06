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

from r2d2_mcp.config import LlmConfig, redact_url  # noqa: E402
from r2d2_mcp.llm import (  # noqa: E402
    Turn, _json_objects, _parse_arguments, _parse_tool_calls,
    mcp_tools_to_openai, tool_input_schema)


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
    """An SDK 1.x tool, which spelled the schema field in camelCase."""

    def __init__(self, name, description, schema):
        self.name = name
        self.description = description
        self.inputSchema = schema


class _FakeToolV2:
    """An SDK 2.x tool. The field was renamed to input_schema in 2.0.

    Reading only the old name does not raise - it returns None, and the tool is
    then advertised to the model as taking no arguments. Every call arrives with
    an empty argument object, the robot is asked to navigate to nowhere, and
    nothing in any log says why.
    """

    def __init__(self, name, description, schema):
        self.name = name
        self.description = description
        self.input_schema = schema


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


# ------------------------------------------------- MCP SDK version skew
#
# The MCP Python SDK renamed FastMCP to MCPServer and moved several fields to
# snake_case in 2.0. A bare `pip install mcp` now resolves to 2.x, so code
# written against 1.x fails - loudly at import, and silently at the schema.

SCHEMA = {'type': 'object',
          'properties': {'name': {'type': 'string'}},
          'required': ['name']}


def test_schema_is_read_from_an_sdk_1_tool():
    assert tool_input_schema(_FakeTool('t', 'd', SCHEMA)) is SCHEMA


def test_schema_is_read_from_an_sdk_2_tool():
    assert tool_input_schema(_FakeToolV2('t', 'd', SCHEMA)) is SCHEMA


def test_a_tool_with_neither_field_yields_none():
    class Bare:
        name = 't'
        description = 'd'
    assert tool_input_schema(Bare()) is None


def test_sdk_2_tools_convert_with_their_parameters_intact():
    """The regression. Reading only `inputSchema` gave every SDK 2.x tool an
    empty schema, so the model was told navigate_to_room takes no arguments."""
    converted = mcp_tools_to_openai([
        _FakeToolV2('navigate_to_room', 'Drive to a named room.', SCHEMA)])
    assert converted[0]['function']['parameters'] is SCHEMA
    assert 'name' in converted[0]['function']['parameters']['properties']


def test_both_sdk_generations_convert_identically():
    v1 = mcp_tools_to_openai([_FakeTool('go', 'Go.', SCHEMA)])
    v2 = mcp_tools_to_openai([_FakeToolV2('go', 'Go.', SCHEMA)])
    assert v1 == v2


def test_a_mixed_listing_converts_completely():
    converted = mcp_tools_to_openai([
        _FakeTool('old', 'Old.', SCHEMA),
        _FakeToolV2('new', 'New.', SCHEMA),
    ])
    assert all(c['function']['parameters']['properties'] for c in converted)


# ------------------------------------------------------ credential redaction
#
# Credentials reach logs by two routes. A self-hosted endpoint is quite
# reasonably configured as https://user:password@host/v1, and httpx's
# raise_for_status() embeds the full request URL in its exception message. The
# agent then puts LlmError text into its episode record and its user-facing
# output, so an unredacted message would be persisted and shown.

def test_userinfo_is_stripped_from_a_url():
    assert redact_url('https://admin:HUNTER2@nim.internal:8000/v1') == \
        'https://***@nim.internal:8000/v1'


def test_a_url_inside_a_sentence_is_redacted():
    message = "Client error '401 Unauthorized' for url 'https://a:b@h/v1'"
    assert 'b@h' not in redact_url(message)
    assert '***@h' in redact_url(message)


def test_a_url_without_credentials_is_untouched():
    url = 'https://integrate.api.nvidia.com/v1'
    assert redact_url(url) == url


def test_a_postgres_dsn_is_redacted_too():
    assert redact_url('postgresql://r2d2:pw@db:5432/r2d2') == \
        'postgresql://***@db:5432/r2d2'


def test_the_host_survives_redaction():
    """The host is diagnostic and not secret; losing it would make the log
    useless."""
    assert 'nim.internal' in redact_url('https://u:p@nim.internal/v1')


def test_several_urls_in_one_message():
    text = 'tried https://a:b@one/v1 then https://c:d@two/v1'
    out = redact_url(text)
    assert 'b@one' not in out and 'd@two' not in out
    assert out.count('***@') == 2


def test_plain_text_is_unchanged():
    assert redact_url('connection refused') == 'connection refused'


def test_describe_redacts_the_configured_endpoint():
    """This string is logged at startup."""
    config = LlmConfig(base_url='https://admin:HUNTER2@nim.internal/v1',
                       model='m', api_key='k')
    assert 'HUNTER2' not in config.describe()
    assert 'nim.internal' in config.describe()


def test_describe_does_not_print_the_api_key():
    config = LlmConfig(base_url='https://h/v1', model='m',
                       api_key='nvapi-SECRET')
    assert 'SECRET' not in config.describe()
    assert 'with API key' in config.describe()


def test_headers_carry_the_key_but_describe_does_not():
    """The key must reach the wire and nowhere else."""
    config = LlmConfig(base_url='https://h/v1', model='m', api_key='nvapi-S3CRET')
    assert config.headers()['Authorization'] == 'Bearer nvapi-S3CRET'
    assert 'S3CRET' not in config.describe()
