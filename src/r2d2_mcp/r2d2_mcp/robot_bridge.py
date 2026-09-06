#!/usr/bin/env python3
"""
The ROS side of the MCP server.

Holds one rclpy node in a background executor and exposes the robot as a set of
awaitable coroutines. The MCP tool layer is a thin wrapper over this: keeping
ROS out of the tool definitions means the tool contract can be read, tested and
changed without a ROS installation.

The important difference from the previous design
-------------------------------------------------
gpt_oss.py posted a navigation goal over HTTP and immediately reported success:
the planner learned whether its command was *accepted*, never whether it
*worked*. Every method here waits for the real outcome and returns it, including
the failure reasons - blocked, timed out, aborted by the terrain monitor. That
is the entire point of moving to tools: a tool call that returns "failed: path
blocked by an unmountable step" lets the model try something else, and a
fire-and-forget POST does not.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from typing import Any, Dict, Optional, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Int32, String


class RobotBridge:
    """Async facade over the ROS graph."""

    def __init__(self, node_name: str = 'r2d2_mcp_bridge'):
        if not rclpy.ok():
            rclpy.init()
        self.node = Node(node_name)
        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        # --- cached state -------------------------------------------------
        self.pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.floor: int = 0
        self.floor_graph: Dict[str, Any] = {}
        self.terrain: Dict[str, Any] = {}
        self.climb: Dict[str, Any] = {}
        self.health: Dict[str, Any] = {}
        self.scene: Dict[str, Any] = {}
        self.detections: Dict[str, Any] = {}
        self.scan: Optional[LaserScan] = None
        self._detection_seq = 0

        sensor_qos = QoSProfile(depth=5,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)

        self.node.create_subscription(Odometry, '/odometry/filtered',
                                      self._on_odom, sensor_qos)
        self.node.create_subscription(Int32, '/floor/current', self._on_floor, latched)
        self.node.create_subscription(String, '/floor/graph', self._on_graph, latched)
        self.node.create_subscription(String, '/terrain/state', self._on_terrain, 10)
        self.node.create_subscription(String, '/climb/state', self._on_climb, 10)
        self.node.create_subscription(String, '/odometry/health', self._on_health, 10)
        self.node.create_subscription(String, '/perception/scene', self._on_scene, 10)
        self.node.create_subscription(String, '/perception/detections',
                                      self._on_detections, 10)
        self.node.create_subscription(LaserScan, '/scan', self._on_scan, sensor_qos)

        self.climb_request_pub = self.node.create_publisher(Bool, '/climb/request', 10)
        self.observe_pub = self.node.create_publisher(String, '/perception/observe', 10)
        self.speech_pub = self.node.create_publisher(String, '/tts_text', 10)
        self.floor_set_pub = self.node.create_publisher(Int32, '/floor/set', 10)
        self.estop_reset_pub = self.node.create_publisher(Bool, '/terrain/estop_reset', 10)
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel_nav', 10)

        self.nav_client = ActionClient(self.node, NavigateToPose, 'navigate_to_pose')
        self._nav_goal_handle = None

    def shutdown(self):
        self._executor.shutdown()
        self.node.destroy_node()

    # ------------------------------------------------------------ callbacks

    def _on_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def _on_floor(self, msg: Int32):
        self.floor = msg.data

    def _on_graph(self, msg: String):
        self.floor_graph = _load(msg.data, {})

    def _on_terrain(self, msg: String):
        self.terrain = _load(msg.data, self.terrain)

    def _on_climb(self, msg: String):
        self.climb = _load(msg.data, self.climb)

    def _on_health(self, msg: String):
        self.health = _load(msg.data, self.health)

    def _on_scene(self, msg: String):
        self.scene = _load(msg.data, self.scene)

    def _on_detections(self, msg: String):
        self.detections = _load(msg.data, self.detections)
        self._detection_seq += 1

    def _on_scan(self, msg: LaserScan):
        self.scan = msg

    # ------------------------------------------------------------ navigation

    async def navigate_to(self, x: float, y: float, yaw: float = 0.0,
                          timeout_s: float = 180.0) -> Dict[str, Any]:
        """Send a Nav2 goal and wait for the real outcome.

        Returns a dict with `success` and, when it fails, a reason a language
        model can act on rather than a bare boolean.
        """
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            return {'success': False,
                    'reason': 'Nav2 is not running (no navigate_to_pose action '
                              'server). Start r2d2_navigation.'}

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.z = math.sin(float(yaw) / 2.0)
        goal.pose.pose.orientation.w = math.cos(float(yaw) / 2.0)

        started = time.time()
        start_pose = self.pose

        send_future = self.nav_client.send_goal_async(goal)
        handle = await _await_ros_future(send_future, timeout_s=10.0)
        if handle is None or not handle.accepted:
            return {'success': False,
                    'reason': 'Nav2 rejected the goal, which usually means it '
                              'is outside the current floor map or inside an '
                              'obstacle.'}

        self._nav_goal_handle = handle
        result_future = handle.get_result_async()
        result = await _await_ros_future(result_future, timeout_s=timeout_s)
        self._nav_goal_handle = None

        elapsed = time.time() - started
        distance_to_goal = math.dist(self.pose[:2], (x, y))

        if result is None:
            await self.cancel_navigation()
            return {'success': False,
                    'reason': f'navigation timed out after {elapsed:.0f} s, '
                              f'still {distance_to_goal:.2f} m from the goal',
                    'elapsed_s': round(elapsed, 1),
                    'distance_remaining_m': round(distance_to_goal, 2)}

        # Nav2 reports status 4 for SUCCEEDED.
        status = getattr(result, 'status', None)
        succeeded = status == 4 or distance_to_goal < 0.35

        out = {
            'success': bool(succeeded),
            'elapsed_s': round(elapsed, 1),
            'distance_remaining_m': round(distance_to_goal, 2),
            'travelled_m': round(math.dist(start_pose[:2], self.pose[:2]), 2),
            'pose': {'x': round(self.pose[0], 2), 'y': round(self.pose[1], 2),
                     'yaw': round(self.pose[2], 3)},
        }
        if not succeeded:
            out['reason'] = self._explain_nav_failure(distance_to_goal)
        return out

    def _explain_nav_failure(self, distance_to_goal: float) -> str:
        """Turn a Nav2 abort into something the model can reason about.

        The terrain monitor and the odometry supervisor usually know why, and
        their answer is far more useful than "goal aborted".
        """
        if self.terrain.get('estop'):
            return (f'stopped by the terrain safety monitor: '
                    f'{self.terrain.get("estop_reason", "attitude limit")}. '
                    f'Clear it with reset_safety_stop once the robot is safe.')
        if self.terrain.get('cliff_ahead'):
            return ('a drop was detected ahead - probably the top of a '
                    'staircase or a landing edge - so the approach was refused')
        if self.terrain.get('blocked_ahead'):
            return ('a step too tall for the wheel clusters to mount is in the '
                    'way; the route needs to avoid it')
        if self.health.get('quality') == 'dead_reckoning':
            return ('the robot does not currently know where it is well enough '
                    'to navigate: it is dead reckoning after a stair '
                    'transition and needs to relocalise')
        return (f'Nav2 gave up {distance_to_goal:.2f} m short. The path is '
                f'likely blocked by something not in the map, such as a closed '
                f'door or a moved chair.')

    async def cancel_navigation(self) -> Dict[str, Any]:
        handle = self._nav_goal_handle
        if handle is None:
            return {'cancelled': False, 'reason': 'nothing is running'}
        await _await_ros_future(handle.cancel_goal_async(), timeout_s=5.0)
        self._nav_goal_handle = None
        self.cmd_pub.publish(Twist())
        return {'cancelled': True}

    # ----------------------------------------------------------------- climb

    async def climb(self, timeout_s: float = 120.0) -> Dict[str, Any]:
        """Request a stair transition and wait for the FSM to finish.

        The robot loses wheel odometry for the duration and the pose estimate
        drifts, so the result reports that explicitly - the model should not
        follow a climb with a precision task before relocalising.
        """
        start_floor = self.floor
        request = Bool()
        request.data = True
        self.climb_request_pub.publish(request)

        started = time.time()
        entered = False
        while time.time() - started < timeout_s:
            state = self.climb.get('state', 'idle')
            if state in ('align', 'approach', 'climb', 'settle'):
                entered = True
            elif entered and state == 'idle':
                break
            await asyncio.sleep(0.2)
        else:
            cancel = Bool()
            cancel.data = False
            self.climb_request_pub.publish(cancel)
            return {'success': False,
                    'reason': f'the climb did not finish within {timeout_s:.0f} s'}

        if not entered:
            # The FSM records why it declined; pass that through rather than
            # guessing, since "no staircase here" and "descent is disabled" call
            # for completely different responses.
            declined = self.climb.get('last_result', '')
            if declined.startswith('refused'):
                return {'success': False, 'reason': declined}
            return {'success': False,
                    'reason': ('the climb never started: the terrain monitor '
                               'saw neither a mountable riser nor a drop ahead. '
                               'Drive to the foot of the stairs and face them '
                               'squarely first.')}

        result = self.climb.get('last_result', '')
        rise = self.climb.get('rise', 0.0)
        success = not result.startswith('abort')

        # Give floor_manager a moment to swap the map and re-seed AMCL.
        await asyncio.sleep(2.0)

        return {
            'success': success,
            'reason': result if not success else None,
            'rise_m': round(float(rise), 2),
            'floor_before': start_floor,
            'floor_now': self.floor,
            'note': ('The pose estimate drifted during the climb and has been '
                     're-seeded at the head of the flight. Confirm the '
                     'surroundings before attempting anything precise.')
            if success else None,
        }

    # --------------------------------------------------------------- observe

    async def observe(self, question: Optional[str] = None,
                      timeout_s: float = 60.0) -> Dict[str, Any]:
        """Ask the VLA for a fresh look and wait for the result."""
        before = self._detection_seq
        msg = String()
        msg.data = question or ''
        self.observe_pub.publish(msg)

        started = time.time()
        while time.time() - started < timeout_s:
            if self._detection_seq > before:
                return {
                    'success': True,
                    'scene': self.scene.get('scene'),
                    'room': self.scene.get('room'),
                    'floor': self.detections.get('floor', self.floor),
                    'objects': self.detections.get('detections', []),
                    'latency_s': self.detections.get('latency_s'),
                }
            await asyncio.sleep(0.25)

        return {'success': False,
                'reason': (f'the vision model did not answer within '
                           f'{timeout_s:.0f} s. It may be rate limited or '
                           f'unreachable; navigation is unaffected.')}

    # ---------------------------------------------------------------- simple

    def speak(self, text: str) -> Dict[str, Any]:
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)
        return {'spoken': text}

    def set_floor(self, floor: int) -> Dict[str, Any]:
        msg = Int32()
        msg.data = int(floor)
        self.floor_set_pub.publish(msg)
        return {'floor_set_to': int(floor)}

    def reset_safety_stop(self) -> Dict[str, Any]:
        msg = Bool()
        msg.data = True
        self.estop_reset_pub.publish(msg)
        return {'reset_requested': True,
                'was': self.terrain.get('estop_reason', 'not latched')}

    def emergency_stop(self) -> Dict[str, Any]:
        self.cmd_pub.publish(Twist())
        cancel = Bool()
        cancel.data = False
        self.climb_request_pub.publish(cancel)
        return {'stopped': True}

    def state(self) -> Dict[str, Any]:
        """Everything the model needs to decide what to do next."""
        return {
            'pose': {'x': round(self.pose[0], 2), 'y': round(self.pose[1], 2),
                     'yaw': round(self.pose[2], 3)},
            'floor': self.floor,
            'localisation': {
                'quality': self.health.get('quality', 'unknown'),
                'active_sources': self.health.get('active_sources', []),
                'precise_goals_allowed': self.health.get('precise_goals_allowed', True),
                'drift_estimate_m': self.health.get('dead_reckoning_drift_estimate'),
            },
            'terrain': {
                'on_slope': self.terrain.get('on_slope'),
                'riser_ahead': self.terrain.get('riser_ahead'),
                'riser_height_m': self.terrain.get('riser_height'),
                'cliff_ahead': self.terrain.get('cliff_ahead'),
                'blocked_ahead': self.terrain.get('blocked_ahead'),
                'safety_stop': self.terrain.get('estop'),
                'safety_stop_reason': self.terrain.get('estop_reason'),
            },
            'locomotion': {
                'climb_state': self.climb.get('state', 'idle'),
                'climb_direction': self.climb.get('direction'),
                'descent_enabled': self.climb.get('descent_enabled', False),
                'last_climb_result': self.climb.get('last_result'),
            },
            'last_view': {
                'room': self.scene.get('room'),
                'scene': self.scene.get('scene'),
                'objects_seen': self.scene.get('object_count', 0),
            },
        }


async def _await_ros_future(future, timeout_s: float):
    """Await an rclpy future from asyncio without blocking the event loop.

    rclpy futures are not awaitable and are completed by the executor thread, so
    they are polled here rather than awaited. Polling at 50 ms costs nothing
    against actions that take tens of seconds.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if future.done():
            try:
                return future.result()
            except Exception:                     # noqa: BLE001 - surfaced by caller
                return None
        await asyncio.sleep(0.05)
    future.cancel()
    return None


def _load(raw: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return fallback
    return value if isinstance(value, dict) else fallback
