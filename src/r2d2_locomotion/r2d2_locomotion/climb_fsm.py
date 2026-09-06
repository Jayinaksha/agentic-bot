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
          the riser. Contact is detected as a STALL - commanded travel with no
          achieved travel - and not as body pitch. In rolling mode the carriers
          are phase-locked, so a sub-wheel meeting a riser face stops dead
          rather than tipping the chassis; waiting for pitch here means waiting
          forever and timing out having never started the climb.
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

Direction
---------
The same states serve both directions, but the two are not mirror images.

Going UP, the riser stops the platform and the resulting stall is an unambiguous
"you are here" event. Going DOWN there is no such event: the ToF beams see the
drop while the wheels are still on solid floor, and a robot that keeps rolling
drives off the top step. Descent is therefore dead-reckoned over a short,
measured creep - the beam spot sits a known distance ahead of the front cluster
contact, so the robot creeps exactly that far, less a margin, before committing.

Descent is OFF by default (`allow_descent`). It is the more dangerous
manoeuvre, it has not been validated on hardware or in simulation, and a robot
that gets it wrong falls down a flight of stairs. With it off the FSM refuses a
descent explicitly rather than failing as "no riser found", which is what it
used to do - the route planner emits descend legs, so silence here meant the
robot could go upstairs and never come back down.
"""

import json
import math
import time
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, String

from r2d2_locomotion.kinematics import (DIRECTION_DOWN, DIRECTION_UP,
                                        EdgeApproach, RiserContact,
                                        descent_creep_distance,
                                        descent_is_geometrically_safe)


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
            ('contact_stall_ratio', 0.35),
            ('contact_confirm_s', 0.6),
            ('approach_timeout', 15.0),
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
            # Descent: off by default. See the module docstring.
            ('allow_descent', False),
            ('descent_speed', 0.07),
            ('descent_margin', 0.05),
            ('edge_confirm_cycles', 4),
            # ToF geometry, mirrored from terrain_monitor, used to work out how
            # far the platform may creep after first seeing a drop.
            ('tof_forward_offset', 0.180),
            ('tof_spot_ahead', 0.290),
            ('wheelbase', 0.260),
        ])
        g = self.get_parameter
        self.rate = g('rate').value
        self.entry_cycles = g('entry_confirm_cycles').value
        self.exit_cycles = g('exit_confirm_cycles').value
        self.approach_speed = g('approach_speed').value
        self.approach_timeout = g('approach_timeout').value
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
        self.allow_descent = g('allow_descent').value
        self.descent_speed = g('descent_speed').value

        creep = descent_creep_distance(
            g('tof_forward_offset').value, g('tof_spot_ahead').value,
            g('wheelbase').value, g('descent_margin').value)
        self._descent_safe = descent_is_geometrically_safe(creep)
        self._edge = EdgeApproach(creep, g('edge_confirm_cycles').value)
        if self.allow_descent and not self._descent_safe:
            self.get_logger().error(
                f'descent is enabled but the ToF beams land {abs(creep):.3f} m '
                f'BEHIND the front cluster contact: the robot cannot see an '
                f'edge before its wheels reach it. Descent disabled.')
            self.allow_descent = False

        self.state = State.IDLE
        self.terrain = {}
        self._riser_streak = 0
        self._level_streak = 0
        self._state_entered = time.time()
        self._climb_started = 0.0
        self._requested = False
        self._direction = DIRECTION_UP
        self._last_result = 'none'

        # IMU-integrated vertical rise, the only trustworthy progress signal
        # while tumbling.
        self._rise = 0.0
        self._vz = 0.0
        self._last_imu_t = None
        self._rise_at_window_start = 0.0
        self._window_start = time.time()

        self._passthrough = Twist()

        # Stall-based riser contact. The trigger for switching into tumbling.
        self._contact = RiserContact(stall_ratio=g('contact_stall_ratio').value,
                                     confirm_s=g('contact_confirm_s').value)
        self._travel_since_tick = 0.0
        self._last_odom_xy = None
        self._commanded_v = 0.0

        sensor_qos = QoSProfile(depth=5,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)

        self.mode_pub = self.create_publisher(String, '/locomotion/mode', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/climb/state', 10)

        self.create_subscription(String, '/terrain/state', self._on_terrain, 10)
        self.create_subscription(Imu, '/imu/data', self._on_imu, sensor_qos)
        # Wheel odometry, valid in rolling mode, is what makes the stall
        # visible. It deliberately is not the filtered estimate: during a climb
        # the EKF is dead reckoning and would report motion that is not there.
        self.create_subscription(Odometry, '/odom_wheel', self._on_wheel_odom,
                                 sensor_qos)
        self.create_subscription(Bool, '/terrain/estop', self._on_estop, 10)
        # Navigation writes here; we gate it and republish on /cmd_vel.
        self.create_subscription(Twist, '/cmd_vel_nav', self._on_nav_cmd, 10)
        self.create_subscription(Bool, '/climb/request', self._on_request, 10)

        self.create_timer(1.0 / self.rate, self._tick)
        self.get_logger().info(
            f'climb FSM up. Autonomous entry: '
            f'{"on" if self.autonomous_entry else "off"}. Descent: '
            f'{f"on, creeping {creep:.3f} m to the edge" if self.allow_descent else "off"}')

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

    def _on_wheel_odom(self, msg: Odometry):
        xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self._last_odom_xy is not None:
            self._travel_since_tick += math.dist(self._last_odom_xy, xy)
        self._last_odom_xy = xy

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
        if state in (State.ALIGN, State.APPROACH):
            self._contact.reset()
            self._edge.reset()
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
        travelled = self._travel_since_tick
        self._travel_since_tick = 0.0
        mode, cmd = self._step(travelled)
        self._commanded_v = cmd.linear.x

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
            'direction': self._direction,
            'descent_enabled': self.allow_descent,
            'contact_evidence_s': round(self._contact.evidence_s, 2),
            'edge_creep_m': round(self._edge.travelled, 3),
            'last_result': self._last_result,
        })
        self.state_pub.publish(s)

    def _step(self, travelled: float = 0.0):
        """Return (transmission_mode, cmd_vel) for this cycle."""
        t = self.terrain
        riser_ahead = bool(t.get('riser_ahead'))
        cliff_ahead = bool(t.get('cliff_ahead'))
        skew = t.get('skew')
        pitch = abs(t.get('pitch', 0.0))
        descending = self._direction == DIRECTION_DOWN

        if self.state == State.IDLE:
            self._riser_streak = self._riser_streak + 1 if riser_ahead else 0
            triggered = self._requested or (
                self.autonomous_entry
                and self._riser_streak >= self.entry_cycles
                and self._passthrough.linear.x > 0.02)
            if triggered:
                # Which way we are going is read from the terrain, not from the
                # request: a riser ahead means up, a drop means down.
                if riser_ahead:
                    self._direction = DIRECTION_UP
                elif cliff_ahead:
                    if not self.allow_descent:
                        self._last_result = (
                            'refused: a stair descent was requested but '
                            'allow_descent is off. Descent is dead-reckoned '
                            'over the last few centimetres and has not been '
                            'validated on this platform; enable it knowingly.')
                        self.get_logger().warn(self._last_result)
                        self._requested = False
                        return MODE_ROLLING, Twist()
                    self._direction = DIRECTION_DOWN
                else:
                    self._last_result = (
                        'refused: neither a mountable riser nor a drop is '
                        'ahead, so there is no staircase here to use')
                    self.get_logger().warn(self._last_result)
                    self._requested = False
                    return MODE_ROLLING, Twist()
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
            # Feed the detectors before any early return, so evidence is not
            # lost on the cycle that would otherwise have confirmed it.
            if descending:
                ready = self._edge.update(travelled, cliff_ahead)
                feature_present = cliff_ahead
            else:
                ready = self._contact.update(
                    self._commanded_v, travelled, 1.0 / self.rate, riser_ahead)
                feature_present = riser_ahead

            if self._elapsed() > self.approach_timeout:
                self._abort(
                    f'approached the {"edge" if descending else "riser"} for '
                    f'{self.approach_timeout:.0f} s without reaching it'
                    if descending else
                    f'drove at the riser for {self.approach_timeout:.0f} s '
                    f'without the wheels ever stalling against it, so the '
                    f'clusters are probably not reaching the step')
                return MODE_STOPPED, Twist()

            if not feature_present and self._elapsed() > 2.0:
                # It was a chair leg, a passing person, or a dark patch of floor.
                self._last_result = (
                    'the drop is no longer visible' if descending
                    else 'no riser found')
                self._requested = False
                self._enter(State.IDLE)
                return MODE_ROLLING, Twist()

            if descending:
                # No contact event exists going down, so the only trigger is the
                # measured creep. Pitch is deliberately NOT a fallback here: by
                # the time the chassis pitches over an edge it is already
                # committed, and reacting then is too late.
                if ready:
                    self._enter(State.CLIMB)
                    return MODE_TUMBLING, Twist()
            elif ready or pitch > 0.08:
                # Contact is the stall. Pitch is kept as a secondary trigger for
                # the case where the platform does ride partway up a shallow
                # nosing, but it must never be the only one.
                self._enter(State.CLIMB)
                return MODE_TUMBLING, Twist()

            cmd = Twist()
            cmd.linear.x = (self.descent_speed if descending
                            else self.approach_speed)
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
            feature_remaining = cliff_ahead if descending else riser_ahead
            if self._level_streak >= self.exit_cycles and not feature_remaining:
                self._last_result = f'reached landing after {self._rise:+.2f} m rise'
                self._requested = False
                self._level_streak = 0
                self._enter(State.SETTLE)
                return MODE_ROLLING, Twist()

            cmd = Twist()
            cmd.linear.x = self.descent_speed if descending else self.climb_speed
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
