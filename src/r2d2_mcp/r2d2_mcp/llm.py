#!/usr/bin/env python3
"""
OpenAI-compatible chat client with tool calling, pointed at NVIDIA models.

One adapter covers every deployment this project cares about, because they all
speak the same wire format:

    NVIDIA API catalogue    https://integrate.api.nvidia.com/v1
    self-hosted NIM         http://<host>:8000/v1
    vLLM serving a Nemotron http://<host>:8000/v1

The only awkward part is that open models vary in how strictly they honour the
tool-calling schema. Two accommodations are made here, both narrow:

  * a model that emits a tool call as a JSON object in the message content
    instead of in `tool_calls` is still understood, since several open models
    do this under load
  * arguments arriving as a JSON string rather than an object are parsed

Neither invents a call that was not made. If nothing parses, the turn is treated
as plain text, which is the safe reading.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from r2d2_mcp.config import LlmConfig

log = logging.getLogger('r2d2.llm')


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class Turn:
    """One assistant response."""

    text: str = ''
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = 'stop'
    raw_message: Dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LlmClient:
    """Minimal async chat-completions client."""

    def __init__(self, config: Optional[LlmConfig] = None):
        self.config = config or LlmConfig()
        self._client = None

    async def _ensure_client(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=self.config.headers(),
                timeout=self.config.timeout_s,
            )
        return self._client

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat(self, messages: List[Dict[str, Any]],
                   tools: Optional[List[Dict[str, Any]]] = None) -> Turn:
        client = await self._ensure_client()

        body: Dict[str, Any] = {
            'model': self.config.model,
            'messages': messages,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
        }
        if tools:
            body['tools'] = tools
            body['tool_choice'] = 'auto'

        response = await client.post('/chat/completions', json=body)
        if response.status_code >= 400:
            raise LlmError(
                f'{self.config.model} returned {response.status_code}: '
                f'{response.text[:400]}')

        payload = response.json()
        choice = payload['choices'][0]
        message = choice.get('message', {})

        return Turn(
            text=(message.get('content') or '').strip(),
            tool_calls=_parse_tool_calls(message),
            finish_reason=choice.get('finish_reason', 'stop'),
            raw_message=message,
        )


class LlmError(RuntimeError):
    """The model endpoint refused or failed."""


def _parse_tool_calls(message: Dict[str, Any]) -> List[ToolCall]:
    """Extract tool calls, tolerating the shapes open models actually emit."""
    calls: List[ToolCall] = []

    for index, raw in enumerate(message.get('tool_calls') or []):
        function = raw.get('function', {})
        name = function.get('name')
        if not name:
            continue
        calls.append(ToolCall(
            id=raw.get('id') or f'call_{index}',
            name=name,
            arguments=_parse_arguments(function.get('arguments')),
        ))

    if calls:
        return calls

    # Fallback: some open models put the call in the content instead. Only a
    # well-formed object with both a name and arguments is accepted, so ordinary
    # prose that happens to contain braces is not misread as a call.
    content = message.get('content') or ''
    for candidate in _json_objects(content):
        name = candidate.get('name') or candidate.get('tool')
        arguments = candidate.get('arguments', candidate.get('parameters'))
        if isinstance(name, str) and isinstance(arguments, (dict, str)):
            log.debug('recovered a tool call from message content: %s', name)
            calls.append(ToolCall(id=f'content_{len(calls)}', name=name,
                                  arguments=_parse_arguments(arguments)))
    return calls


def _parse_arguments(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _json_objects(text: str) -> List[Dict[str, Any]]:
    """Every balanced top-level JSON object in a string."""
    out: List[Dict[str, Any]] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for i, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    value = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(value, dict):
                        out.append(value)
                start = -1
            elif depth < 0:
                depth = 0
    return out


def mcp_tools_to_openai(tools: List[Any]) -> List[Dict[str, Any]]:
    """Convert an MCP tool listing into OpenAI function-tool definitions.

    MCP already carries a JSON Schema for each tool, which is exactly what the
    chat API wants, so this is a rename rather than a translation. That
    correspondence is the practical reason MCP is worth using here: the robot
    describes its own capabilities once and every model sees the same contract.
    """
    out = []
    for tool in tools:
        schema = getattr(tool, 'inputSchema', None) or {'type': 'object',
                                                        'properties': {}}
        out.append({
            'type': 'function',
            'function': {
                'name': tool.name,
                'description': (tool.description or '').strip(),
                'parameters': schema,
            },
        })
    return out
