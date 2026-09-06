#!/usr/bin/env python3
"""
Measure the skid-steer yaw slip factor.

    ros2 run r2d2_locomotion tristar_controller     # in another terminal
    python3 scripts/calibrate_slip.py
    python3 scripts/calibrate_slip.py --turns 3 --rate 0.8

Why this is needed
------------------
A four-contact skid-steer does not turn at the rate the ideal differential model
predicts. Turning drags all four clusters sideways across the floor, and that
scrub eats part of the commanded yaw. `tristar_controller` compensates with

    v_left, v_right = v -/+ yaw_slip * w * track / 2

where `yaw_slip` > 1. Its default of 1.25 is an estimate for this footprint, not
a measurement. It is worth measuring because the error is systematic: every
turn is short by the same fraction, so the odometry heading drifts one way and
the EKF spends the whole run fighting it.

How it works
------------
Command a steady spin, integrate the IMU's yaw rate to get what the robot
actually did, and compare with what was asked. The IMU is the reference because
it is the one heading source that does not depend on the wheels.

    yaw_slip_new = yaw_slip_current * (commanded / achieved)

Run it on a flat, uncluttered patch of floor, on the surface the robot will
actually work on - carpet and tile give noticeably different numbers.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Imu
    from std_msgs.msg import String
except ImportError:
    sys.exit('This script needs a sourced ROS 2 environment.')


class SlipCalibrator(Node):

    def __init__(self, rate: float, turns: float, current_slip: float,
                 settle: float):
        super().__init__('slip_calibrator')
        self.rate = rate
        self.target_yaw = turns * 2 * math.pi
        self.current_slip = current_slip
        self.settle = settle

        self.integrated = 0.0
        self.samples = 0
        self._last_stamp = None
        self._started = None

        qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Imu, '/imu/data', self._on_imu, qos)
        # Publish where the climb FSM expects navigation to write, so the same
        # gate and safety chain apply as in normal operation.
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        self.mode_pub = self.create_publisher(String, '/locomotion/mode', 10)

    def _on_imu(self, msg: Imu):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_stamp is None:
            self._last_stamp = stamp
            return
        dt = stamp - self._last_stamp
        self._last_stamp = stamp
        if self._started is None or dt <= 0 or dt > 0.5:
            return
        self.integrated += msg.angular_velocity.z * dt
        self.samples += 1

    def spin_test(self) -> float:
        """Spin until the commanded yaw is reached; return what the IMU saw."""
        duration = self.target_yaw / self.rate
        command = Twist()
        command.angular.z = self.rate

        print(f'commanding {math.degrees(self.rate):.0f} deg/s for '
              f'{duration:.1f} s ({self.target_yaw / (2 * math.pi):.1f} turns)')
        print('keep clear of the robot\n')

        for remaining in range(int(self.settle), 0, -1):
            print(f'  starting in {remaining}...', end='\r', flush=True)
            self._pump(1.0)
        print(' ' * 30, end='\r')

        self._started = time.time()
        deadline = self._started + duration
        while time.time() < deadline:
            self.cmd_pub.publish(command)
            self._pump(0.05)
            done = abs(self.integrated) / self.target_yaw
            print(f'  {done * 100:5.1f}%  integrated '
                  f'{math.degrees(abs(self.integrated)):7.1f} deg', end='\r',
                  flush=True)

        self.cmd_pub.publish(Twist())
        # Let the platform stop before the last samples are counted.
        self._pump(1.5)
        print()
        return abs(self.integrated)

    def _pump(self, seconds: float):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rate', type=float, default=0.8,
                        help='yaw rate to command, rad/s (default 0.8)')
    parser.add_argument('--turns', type=float, default=3.0,
                        help='how many full turns to average over (default 3)')
    parser.add_argument('--current-slip', type=float, default=1.25,
                        help='yaw_slip_factor currently in locomotion.yaml')
    parser.add_argument('--settle', type=float, default=3.0,
                        help='countdown before moving, seconds')
    args = parser.parse_args()

    rclpy.init()
    node = SlipCalibrator(args.rate, args.turns, args.current_slip, args.settle)
    try:
        achieved = node.spin_test()
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

    if node.samples < 20:
        print(f'\nonly {node.samples} IMU samples: is /imu/data publishing?')
        return 1
    if achieved < 1e-3:
        print('\nthe robot did not turn at all. Is the controller running, and '
              'is the climb FSM in its idle state?')
        return 1

    commanded = node.target_yaw
    ratio = commanded / achieved
    suggested = args.current_slip * ratio

    print(f'\ncommanded : {math.degrees(commanded):8.1f} deg')
    print(f'achieved  : {math.degrees(achieved):8.1f} deg '
          f'({node.samples} IMU samples)')
    print(f'shortfall : {(1 - achieved / commanded) * 100:8.1f} %')
    print(f'\ncurrent yaw_slip_factor : {args.current_slip:.3f}')
    print(f'suggested               : {suggested:.3f}')

    if abs(suggested - args.current_slip) < 0.03:
        print('\nclose enough; leave it alone.')
    elif suggested < 1.0:
        print('\nA factor below 1.0 means the robot over-rotated, which a '
              'skid-steer should not do. Check that the IMU z axis points up '
              'and that the track_width in locomotion.yaml is right.')
    else:
        print(f'\nSet yaw_slip_factor to {suggested:.2f} in '
              f'src/r2d2_locomotion/config/locomotion.yaml, then re-run to '
              f'confirm the shortfall has gone.')
        print('Repeat on each surface the robot works on; carpet and tile '
              'differ by more than you would expect.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
