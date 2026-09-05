#!/usr/bin/env python3
"""
ROS bridge into the event ledger.

Subscribes to the topics worth remembering and appends them to JetStream. It is
the only ROS node that writes to the ledger from the perception side; the agent
writes its own decisions and episodes directly from the MCP layer, because it is
already async and already holds the context those records need.

What is recorded, and what is not:

    /perception/detections   every grounded VLA detection            -> percept
    /odometry/filtered       pose, throttled to `pose_period`        -> telemetry
    /terrain/state           only on a state change, not at 20 Hz    -> telemetry
    /climb/state             transitions only                        -> telemetry
    /odometry/health         only when quality changes               -> telemetry

Throttling is the whole design here. A robot running for a day at full rate
would write tens of millions of near-identical pose records, which costs disk,
slows every replay, and buries the events that actually matter. Telemetry is
sampled; perception is not, because a detection is exactly the thing you cannot
reconstruct later.

Runs its own asyncio loop in a background thread: rclpy's executor is
synchronous, and blocking a ROS callback on a NATS publish would stall
perception whenever the network hiccups.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, String

from r2d2_memory.ledger import make_ledger


class MemoryNode(Node):

    def __init__(self):
        super().__init__('memory_node')

        self.declare_parameters('', [
            ('enabled', True),
            ('nats_url', 'nats://127.0.0.1:4222'),
            ('pose_period', 2.0),          # s between pose records
            ('pose_min_travel', 0.25),     # m; a stationary robot writes nothing
            ('queue_limit', 2000),
        ])
        g = self.get_parameter
        self.enabled = g('enabled').value
        self.pose_period = g('pose_period').value
        self.pose_min_travel = g('pose_min_travel').value
        queue_limit = g('queue_limit').value

        self._ledger = make_ledger(enabled=self.enabled,
                                   servers=g('nats_url').value)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()
        self._queue: Optional[asyncio.Queue] = None
        self._queue_limit = queue_limit
        self._dropped = 0

        self._floor = 0
        self._pose = (0.0, 0.0, 0.0)
        self._last_pose_record = (0.0, 0.0)
        self._last_pose_time = 0.0
        self._last_terrain_digest: Optional[str] = None
        self._last_climb_state: Optional[str] = None
        self._last_health_quality: Optional[str] = None

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10.0)

        sensor_qos = QoSProfile(depth=10,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(String, '/perception/detections',
                                 self._on_detections, 10)
        self.create_subscription(Odometry, '/odometry/filtered',
                                 self._on_odom, sensor_qos)
        self.create_subscription(String, '/terrain/state', self._on_terrain, 10)
        self.create_subscription(String, '/climb/state', self._on_climb, 10)
        self.create_subscription(String, '/odometry/health', self._on_health, 10)
        self.create_subscription(Int32, '/floor/current', self._on_floor, 10)

        self.create_timer(30.0, self._report)
        self.get_logger().info(
            f'memory node up (ledger {"enabled" if self.enabled else "disabled"})')

    # ------------------------------------------------------- async plumbing

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._queue = asyncio.Queue(maxsize=self._queue_limit)
        self._loop.run_until_complete(self._drain())

    async def _drain(self):
        try:
            await self._ledger.connect()
        except Exception as exc:                  # noqa: BLE001 - keep driving
            # Losing memory must not stop the robot. Log once, loudly, and keep
            # the node alive so nothing downstream blocks on it.
            self.get_logger().error(
                f'ledger unavailable ({exc}); continuing without memory. '
                f'Navigation is unaffected; recall is not.')
        self._ready.set()

        while True:
            kind, payload, floor = await self._queue.get()
            try:
                await self._ledger.append(kind, payload, floor=floor)
            except Exception as exc:              # noqa: BLE001
                self.get_logger().warn(f'ledger append failed: {exc}')
            finally:
                self._queue.task_done()

    def _submit(self, kind: str, payload: Dict[str, Any],
                floor: Optional[int] = None):
        """Hand an event to the async loop without blocking the ROS callback."""
        if self._loop is None or self._queue is None:
            return
        try:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, (kind, payload, floor))
        except asyncio.QueueFull:
            # Backpressure: drop rather than grow without bound. Counted so the
            # loss is visible instead of silent.
            self._dropped += 1
        except RuntimeError:
            pass                                  # loop shutting down

    # ---------------------------------------------------------------- inputs

    def _on_floor(self, msg: Int32):
        self._floor = msg.data

    def _on_detections(self, msg: String):
        """Grounded VLA detections. Never throttled - these are the memory."""
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('malformed /perception/detections payload')
            return

        for detection in payload.get('detections', []):
            if 'world_x' not in detection or 'world_y' not in detection:
                continue                          # ungrounded; nothing to store
            record = dict(detection)
            record.setdefault('floor', payload.get('floor', self._floor))
            record.setdefault('robot_pose', payload.get('robot_pose', {
                'x': self._pose[0], 'y': self._pose[1], 'yaw': self._pose[2]}))
            record.setdefault('source', payload.get('source', 'vla'))
            self._submit('percept', record, floor=record['floor'])

    def _on_odom(self, msg: Odometry):
        now = self.get_clock().now().nanoseconds * 1e-9
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

        travelled = math.dist(self._last_pose_record, self._pose[:2])
        if now - self._last_pose_time < self.pose_period:
            return
        if travelled < self.pose_min_travel:
            # Parked: writing the same pose every two seconds is pure noise.
            self._last_pose_time = now
            return

        self._last_pose_time = now
        self._last_pose_record = self._pose[:2]
        self._submit('telemetry', {
            'kind': 'pose',
            'x': round(self._pose[0], 3),
            'y': round(self._pose[1], 3),
            'yaw': round(self._pose[2], 4),
            'floor': self._floor,
        }, floor=self._floor)

    def _on_terrain(self, msg: String):
        try:
            state = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        # Record a change of situation, not a 20 Hz stream of the same one.
        digest = (f"{state.get('riser_ahead')}|{state.get('cliff_ahead')}|"
                  f"{state.get('blocked_ahead')}|{state.get('on_slope')}|"
                  f"{state.get('estop')}")
        if digest == self._last_terrain_digest:
            return
        self._last_terrain_digest = digest
        self._submit('telemetry', {
            'kind': 'terrain',
            'riser_ahead': state.get('riser_ahead'),
            'riser_height': state.get('riser_height'),
            'cliff_ahead': state.get('cliff_ahead'),
            'blocked_ahead': state.get('blocked_ahead'),
            'on_slope': state.get('on_slope'),
            'estop': state.get('estop'),
            'estop_reason': state.get('estop_reason'),
            'pose': {'x': self._pose[0], 'y': self._pose[1]},
        }, floor=self._floor)

    def _on_climb(self, msg: String):
        try:
            state = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        current = state.get('state')
        if current == self._last_climb_state:
            return
        self._last_climb_state = current
        self._submit('telemetry', {
            'kind': 'climb',
            'state': current,
            'rise': state.get('rise'),
            'last_result': state.get('last_result'),
            'floor': self._floor,
        }, floor=self._floor)

    def _on_health(self, msg: String):
        try:
            health = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        quality = health.get('quality')
        if quality == self._last_health_quality:
            return
        self._last_health_quality = quality
        self._submit('telemetry', {
            'kind': 'odometry_health',
            'quality': quality,
            'regime': health.get('regime'),
            'active_sources': health.get('active_sources'),
            'drift_estimate': health.get('dead_reckoning_drift_estimate'),
        }, floor=self._floor)

    def _report(self):
        if self._dropped:
            self.get_logger().warn(
                f'{self._dropped} ledger events dropped under backpressure; '
                f'the ledger is not keeping up with the robot')
            self._dropped = 0


def main(args=None):
    rclpy.init(args=args)
    node = MemoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
