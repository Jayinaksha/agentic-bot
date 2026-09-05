#!/usr/bin/env python3
"""
Stair-climb state machine.

Owns the transmission mode. Nothing else in the stack may publish
/locomotion/mode; the controller only executes what this node decides.

    IDLE ──(request or confirmed riser)──> ALIGN ──> APPROACH ──> CLIMB
      ^                                      |          |           |
      └──────────── ABORT <──────────────────┴──────────┴───────────┤
      └──────────── SETTLE <─────────────────────────────────────────┘

IDLE      Rolling mode. /cmd_vel passes through untouched.
ALIGN     Square the platform to the flight. The two front ToF beams give a
          skew signal; yaw until it is near zero. Climbing a staircase crooked
          is the classic failure mode for tri-wheel platforms - one side mounts,
          the other does not, and the robot ends up wedged diagonally.
APPROACH  Creep forward in rolling mode until the front clusters are against
          the riser (detected as the riser distance closing plus a drop in
          forward progress).
CLIMB     Tumbling mode. Constant slow forward command; the clusters walk the
          risers. Yaw is servoed off the skew signal continuously, because a
          tri-star flight drifts. Exits when the IMU reports level attitude for
          exit_confirm_cycles and no riser remains ahead.
SETTLE    Back to rolling, hold still while odom_supervisor relocalises. This
          state exists because wheel odometry was invalid for the whole climb
          and the EKF needs a clean restart before Nav2 is handed control again.
ABORT     Reverse away from the flight and latch a failure. Triggered by
          timeout, stall, or a roll excursion.

Progress is measured from IMU-integrated vertical rise, not from wheels, since
wheel odometry is meaningless while tumbling.
"""

import json
import math
import time
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, String


class State(Enum):
    IDLE = 'idle'
    ALIGN = 'align'
    APPROACH = 'approach'
    CLIMB = 'climb'
    SETTLE = 'settle'
    ABORT = 'abort'


MODE_ROLLING = 'rolling'
MODE_TUMBLING = 'tumbling'
MODE_STOPPED = 'stopped'


class ClimbFsm(Node):

    def __init__(self):
        super().__init__('climb_fsm')

        self.declare_parameters('', [
            ('rate', 20.0),
            ('entry_confirm_cycles', 6),
            ('exit_confirm_cycles', 10),
            ('approach_speed', 0.10),
            ('climb_speed', 0.12),
            ('align_gain', 1.4),
            ('max_align_yaw', 0.35),
            ('align_tolerance', 0.012),
            ('climb_timeout', 90.0),
            ('stall_travel_epsilon', 0.02),
            ('stall_window', 2.0),
            ('settle_duration', 1.5),
            ('abort_backoff_distance', 0.35),
            ('autonomous_entry', False),
        ])
        g = self.get_parameter
        self.rate = g('rate').value
        self.entry_cycles = g('entry_confirm_cycles').value
        self.exit_cycles = g('exit_confirm_cycles').value
        self.approach_speed = g('approach_speed').value
        self.climb_speed = g('climb_speed').value
        self.align_gain = g('align_gain').value
        self.max_align_yaw = g('max_align_yaw').value
        self.align_tol = g('align_tolerance').value
        self.timeout = g('climb_timeout').value
        self.stall_eps = g('stall_travel_epsilon').value
        self.stall_window = g('stall_window').value
        self.settle_duration = g('settle_duration').value
        self.backoff_distance = g('abort_backoff_distance').value
        # Off by default: a climb costs the robot its wheel odometry, so it is
        # normally requested explicitly by the navigation layer rather than
        # triggered by whatever the ToF beams happen to see.
        self.autonomous_entry = g('autonomous_entry').value

        self.state = State.IDLE
        self.terrain = {}
        self._riser_streak = 0
        self._level_streak = 0
        self._state_entered = time.time()
        self._climb_started = 0.0
        self._requested = False
        self._last_result = 'none'

        # IMU-integrated vertical rise, the only trustworthy progress signal
        # while tumbling.
        self._rise = 0.0
        self._vz = 0.0
        self._last_imu_t = None
        self._rise_at_window_start = 0.0
        self._window_start = time.time()

        self._passthrough = Twist()

        sensor_qos = QoSProfile(depth=5,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)

        self.mode_pub = self.create_publisher(String, '/locomotion/mode', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/climb/state', 10)

        self.create_subscription(String, '/terrain/state', self._on_terrain, 10)
        self.create_subscription(Imu, '/imu/data', self._on_imu, sensor_qos)
        self.create_subscription(Bool, '/terrain/estop', self._on_estop, 10)
        # Navigation writes here; we gate it and republish on /cmd_vel.
        self.create_subscription(Twist, '/cmd_vel_nav', self._on_nav_cmd, 10)
        self.create_subscription(Bool, '/climb/request', self._on_request, 10)

        self.create_timer(1.0 / self.rate, self._tick)
        self.get_logger().info('climb FSM up, autonomous entry: '
                               f'{"on" if self.autonomous_entry else "off"}')

    # ---------------------------------------------------------------- inputs

    def _on_terrain(self, msg: String):
        try:
            self.terrain = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('malformed /terrain/state payload')

    def _on_imu(self, msg: Imu):
        now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_imu_t is None:
            self._last_imu_t = now
            return
        dt = now - self._last_imu_t
        self._last_imu_t = now
        if dt <= 0.0 or dt > 0.25:
            return
        if self.state not in (State.CLIMB, State.APPROACH):
            self._vz = 0.0
            return
        # Gravity-compensated vertical acceleration in the body frame, projected
        # onto world z using the current pitch. Crude, but it only has to
        # distinguish "rising" from "stuck" over a couple of seconds.
        pitch = self.terrain.get('pitch', 0.0)
        az_world = (msg.linear_acceleration.z * math.cos(pitch)
                    + msg.linear_acceleration.x * math.sin(pitch)) - 9.81
        self._vz += az_world * dt
        self._vz *= 0.98          # leak, to bound integrator drift
        self._rise += self._vz * dt

    def _on_estop(self, msg: Bool):
        if msg.data and self.state in (State.ALIGN, State.APPROACH, State.CLIMB):
            self._abort('terrain e-stop')

    def _on_nav_cmd(self, msg: Twist):
        self._passthrough = msg

    def _on_request(self, msg: Bool):
        if msg.data and self.state == State.IDLE:
            self.get_logger().info('climb requested by navigation layer')
            self._requested = True
        elif not msg.data and self.state in (State.ALIGN, State.APPROACH, State.CLIMB):
            self._abort('climb cancelled by navigation layer')

    # ------------------------------------------------------------ transitions

    def _enter(self, state: State):
        if state == self.state:
            return
        self.get_logger().info(f'climb FSM {self.state.value} -> {state.value}')
        self.state = state
        self._state_entered = time.time()
        if state == State.CLIMB:
            self._climb_started = time.time()
            self._rise = 0.0
            self._vz = 0.0
            self._rise_at_window_start = 0.0
            self._window_start = time.time()

    def _abort(self, reason: str):
        self.get_logger().error(f'climb aborted: {reason}')
        self._last_result = f'abort: {reason}'
        self._requested = False
        self._enter(State.ABORT)

    def _elapsed(self) -> float:
        return time.time() - self._state_entered

    # -------------------------------------------------------------- main loop

    def _tick(self):
        mode, cmd = self._step()

        m = String()
        m.data = mode
        self.mode_pub.publish(m)
        self.cmd_pub.publish(cmd)

        s = String()
        s.data = json.dumps({
            'state': self.state.value,
            'requested': self._requested,
            'rise': round(self._rise, 3),
            'elapsed': round(self._elapsed(), 2),
            'last_result': self._last_result,
        })
        self.state_pub.publish(s)

    def _step(self):
        """Return (transmission_mode, cmd_vel) for this cycle."""
        t = self.terrain
        riser_ahead = bool(t.get('riser_ahead'))
        skew = t.get('skew')
        pitch = abs(t.get('pitch', 0.0))

        if self.state == State.IDLE:
            self._riser_streak = self._riser_streak + 1 if riser_ahead else 0
            triggered = self._requested or (
                self.autonomous_entry
                and self._riser_streak >= self.entry_cycles
                and self._passthrough.linear.x > 0.02)
            if triggered:
                self._enter(State.ALIGN)
                return MODE_ROLLING, Twist()
            # Normal driving: hand navigation's command straight through.
            return MODE_ROLLING, self._passthrough

        if self.state == State.ALIGN:
            if self._elapsed() > 12.0:
                self._abort('could not square up to the flight')
                return MODE_STOPPED, Twist()
            cmd = Twist()
            if skew is None:
                # Beams disagree about whether there is a floor at all; creep
                # forward to get a cleaner look rather than yawing blind.
                cmd.linear.x = self.approach_speed * 0.5
                return MODE_ROLLING, cmd
            if abs(skew) <= self.align_tol:
                self._enter(State.APPROACH)
                return MODE_ROLLING, Twist()
            # Left beam reading higher than right means the left cluster is
            # nearer the riser, so yaw left to even them out.
            cmd.angular.z = _clamp(self.align_gain * skew, self.max_align_yaw)
            return MODE_ROLLING, cmd

        if self.state == State.APPROACH:
            if self._elapsed() > 15.0:
                self._abort('never reached the first riser')
                return MODE_STOPPED, Twist()
            if not riser_ahead and self._elapsed() > 2.0:
                # The riser disappeared: it was a chair leg or a passing person.
                self._last_result = 'no riser found'
                self._requested = False
                self._enter(State.IDLE)
                return MODE_ROLLING, Twist()
            if pitch > 0.08:
                # The front clusters have started to lift: we are on the step.
                self._enter(State.CLIMB)
                return MODE_TUMBLING, Twist()
            cmd = Twist()
            cmd.linear.x = self.approach_speed
            if skew is not None:
                cmd.angular.z = _clamp(self.align_gain * skew, self.max_align_yaw)
            return MODE_ROLLING, cmd

        if self.state == State.CLIMB:
            if time.time() - self._climb_started > self.timeout:
                self._abort('climb timeout')
                return MODE_STOPPED, Twist()
            if self._stalled():
                self._abort('no vertical progress, wedged on the flight')
                return MODE_STOPPED, Twist()

            level = pitch < 0.06
            self._level_streak = self._level_streak + 1 if level else 0
            if self._level_streak >= self.exit_cycles and not riser_ahead:
                self._last_result = f'reached landing after {self._rise:+.2f} m rise'
                self._requested = False
                self._level_streak = 0
                self._enter(State.SETTLE)
                return MODE_ROLLING, Twist()

            cmd = Twist()
            cmd.linear.x = self.climb_speed
            if skew is not None:
                cmd.angular.z = _clamp(self.align_gain * 0.6 * skew, self.max_align_yaw)
            return MODE_TUMBLING, cmd

        if self.state == State.SETTLE:
            if self._elapsed() > self.settle_duration:
                self._enter(State.IDLE)
            return MODE_ROLLING, Twist()

        if self.state == State.ABORT:
            # Reverse clear of the flight, then hand control back.
            if self._elapsed() > self.backoff_distance / max(self.approach_speed, 1e-3):
                self._enter(State.IDLE)
                return MODE_ROLLING, Twist()
            cmd = Twist()
            cmd.linear.x = -self.approach_speed
            return MODE_ROLLING, cmd

        return MODE_STOPPED, Twist()

    def _stalled(self) -> bool:
        now = time.time()
        if now - self._window_start < self.stall_window:
            return False
        # Magnitude, not signed: descending a flight is negative progress on
        # this axis but is not a stall.
        progress = abs(self._rise - self._rise_at_window_start)
        self._rise_at_window_start = self._rise
        self._window_start = now
        return progress < self.stall_eps


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def main(args=None):
    rclpy.init(args=args)
    node = ClimbFsm()
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
