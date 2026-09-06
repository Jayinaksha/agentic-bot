#!/usr/bin/env python3
"""
Configuration for the MCP layer.

Everything is environment-driven so the same code runs against NVIDIA's hosted
catalogue, your own NIM container on a cloud GPU, or a local vLLM, without a
code change.

    # NVIDIA-hosted (easiest to start with)
    export R2D2_NVIDIA_API_KEY=nvapi-...
    export R2D2_LLM_BASE_URL=https://integrate.api.nvidia.com/v1
    export R2D2_LLM_MODEL=nvidia/nemotron-nano-12b-v2-vl

    # Self-hosted NIM / vLLM on a cloud GPU (Nebius, GCP, ...)
    export R2D2_LLM_BASE_URL=http://10.0.0.5:8000/v1
    export R2D2_LLM_MODEL=nvidia/cosmos-reason2-8b

Both are OpenAI-compatible, which is the whole reason this is one adapter and
not three. The agent runs on the machine next to Gazebo and only makes outbound
HTTPS calls, so nothing has to be exposed inbound.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

# NVIDIA-hosted API catalogue. Self-hosted NIM containers expose the same paths
# on their own host and port.
NVIDIA_CATALOGUE = 'https://integrate.api.nvidia.com/v1'

# Defaults chosen for this robot:
#   planner  a Nemotron VL model - it has to reason over tool results and, when
#            asked, look at a frame, so a VLM avoids a second model round trip.
#   vla      Cosmos Reason 2, which returns object localisation with its
#            reasoning, which is what the grounding step needs.
DEFAULT_PLANNER_MODEL = 'nvidia/nemotron-nano-12b-v2-vl'
DEFAULT_VLA_MODEL = 'nvidia/cosmos-reason2-8b'


@dataclass
class LlmConfig:
    base_url: str = field(
        default_factory=lambda: os.environ.get('R2D2_LLM_BASE_URL', NVIDIA_CATALOGUE))
    model: str = field(
        default_factory=lambda: os.environ.get('R2D2_LLM_MODEL', DEFAULT_PLANNER_MODEL))
    api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get('R2D2_NVIDIA_API_KEY') or None)

    temperature: float = 0.2
    max_tokens: int = 1536
    timeout_s: float = 90.0

    # How many tool-calling rounds one instruction may take before the agent
    # gives up. A house errand is a handful of calls; anything past this is a
    # loop, and letting it run costs tokens and wall-clock for no progress.
    max_rounds: int = 14

    def headers(self) -> dict:
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def describe(self) -> str:
        """A log-safe one-liner. The base URL is redacted because a self-hosted
        endpoint is quite reasonably written as https://user:password@host/v1,
        and this string is logged at startup."""
        auth = 'with API key' if self.api_key else 'no API key (self-hosted?)'
        return f'{self.model} at {redact_url(self.base_url)} ({auth})'


def redact_url(text: str) -> str:
    """Strip userinfo from any URL in a string.

    Credentials reach logs by two routes: a URL configured with them inline, and
    httpx's raise_for_status(), which embeds the full request URL in its message.
    Anything that prints a URL, or an exception that might contain one, goes
    through here first.
    """
    return re.sub(r'(\w+://)[^/@\s]+@', r'\1***@', text)


@dataclass
class RobotConfig:
    """Where the MCP server finds the robot and its memory."""

    nats_url: str = field(
        default_factory=lambda: os.environ.get('R2D2_NATS_URL',
                                               'nats://127.0.0.1:4222'))
    pg_dsn: str = field(
        default_factory=lambda: os.environ.get(
            'R2D2_PG_DSN', 'postgresql://r2d2:r2d2@127.0.0.1:5432/r2d2'))
    memory_enabled: bool = field(
        default_factory=lambda: os.environ.get('R2D2_MEMORY', 'on') != 'off')

    # Timeouts for robot actions, seconds. Generous, because this platform
    # legitimately crawls: it creeps over sills and takes over a minute on a
    # flight of stairs.
    nav_timeout_s: float = 180.0
    climb_timeout_s: float = 120.0
    dock_timeout_s: float = 30.0
    observe_timeout_s: float = 60.0


@dataclass
class ServerConfig:
    """MCP transport."""

    # stdio is the default because the agent runs beside the server on the same
    # machine; http is there for when you want the agent elsewhere.
    transport: str = field(
        default_factory=lambda: os.environ.get('R2D2_MCP_TRANSPORT', 'stdio'))
    host: str = field(
        default_factory=lambda: os.environ.get('R2D2_MCP_HOST', '127.0.0.1'))
    port: int = field(
        default_factory=lambda: int(os.environ.get('R2D2_MCP_PORT', '8931')))
