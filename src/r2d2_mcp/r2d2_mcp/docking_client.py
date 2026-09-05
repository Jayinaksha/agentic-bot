#!/usr/bin/env python3
"""Client for the precise docking servo.

Publishes a request on /docking/request and waits for the matching result.
Requests carry an id so a late result from an abandoned dock cannot be mistaken
for the answer to the current one.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Dict, Optional

from std_msgs.msg import String


async def run_docking(bridge, bearing_rad: float, target: str,
                      standoff: float, timeout_s: float = 30.0
                      ) -> Dict[str, Any]:
    """Ask the servo to dock, and wait for it to report back."""
    request_id = uuid.uuid4().hex[:12]
    result: Dict[str, Any] = {}
    done = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_result(msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if payload.get('id') != request_id:
            # A result from an earlier, abandoned request.
            return
        result.update(payload)
        loop.call_soon_threadsafe(done.set)

    subscription = bridge.node.create_subscription(
        String, '/docking/result', _on_result, 10)
    publisher = bridge.node.create_publisher(String, '/docking/request', 10)

    try:
        # Two reasons to wait before publishing: the servo node subscribes on
        # demand, so the first request of a session would otherwise be dropped
        # by discovery; and Nav2's velocity chain keeps writing /cmd_vel_nav for
        # up to velocity_smoother.velocity_timeout (1.0 s) after its goal is
        # cancelled, which would fight the servo for the wheels.
        await asyncio.sleep(1.2)
        message = String()
        message.data = json.dumps({
            'id': request_id,
            'bearing': bearing_rad,
            'target': target,
            'standoff': standoff,
        })
        publisher.publish(message)

        try:
            await asyncio.wait_for(done.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return {
                'success': False,
                'reason': (f'the docking servo did not respond within '
                           f'{timeout_s:.0f} s. Is precise_docking running '
                           f'(r2d2_navigation navigation.launch.py)?'),
            }

        out: Dict[str, Any] = {
            'success': bool(result.get('success')),
            'message': result.get('message'),
        }
        if result.get('residual_m') is not None:
            out['final_error_cm'] = round(result['residual_m'] * 100, 1)
        if result.get('yaw_error_rad') is not None:
            out['final_yaw_error_deg'] = round(
                result['yaw_error_rad'] * 57.2958, 1)
        if not out['success']:
            out['reason'] = result.get('message', 'docking failed')
        return out
    finally:
        bridge.node.destroy_subscription(subscription)
        bridge.node.destroy_publisher(publisher)
