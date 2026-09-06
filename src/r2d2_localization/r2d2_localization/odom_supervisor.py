#!/usr/bin/env python3
"""
Odometry supervisor: decides which estimator the robot is allowed to trust.

The problem this exists to solve
--------------------------------
This platform has three odometry sources and each of them fails in a different,
predictable place:

    wheels        wrong whenever a cluster tumbles (no fixed contact radius)
    scan matcher  wrong on stairs (planar scanner pointing at a ceiling) and
                  degenerate in a bare corridor (no features along the axis)
    IMU           always available, always drifting

A single fixed EKF configuration cannot be right in all three regimes. Running
one anyway is how a stack ends up confidently lost: the filter keeps reporting a
tight covariance while fusing a source that has quietly become fiction.

How the regime is enforced
--------------------------
robot_localization cannot be reconfigured at runtime, so the regime is applied
where it actually can be - in the covariances the filter consumes:

    /odom_scan_raw --> [this node] --> /odom_scan --> EKF

This node is a gate on the scan-matcher stream. When the scan is degenerate or
the robot is climbing, it republishes the message with its covariance inflated
to effectively infinite, which makes the EKF's Kalman gain for that source go to
zero. The wheel stream is gated the same way at its source: tristar_controller
already publishes 1e6 covariance whenever the transmission is tumbling.

The result is one filter configuration that behaves like three, without any
service calls, node restarts or lifecycle transitions in the middle of a climb.
ekf_climb.yaml is kept for the case where you would rather run a second,
3D filter during transitions; localization.launch.py has a flag for it.

It also publishes an honest health signal on /odometry/health so the navigation
and MCP layers can refuse to accept a precise goal when the pose underneath it
is not precise. Reporting "I do not know where I am well enough for that" is a
feature; silently accepting the goal and driving into a wall is not.

Degeneracy detection
--------------------
A laser scan in a long bare corridor constrains the robot across the corridor
but barely along it. We detect that from the scan itself: bin the returns by
angle, and if almost all the structure is perpendicular to one axis, the
along-axis constraint is weak. Cheap, and it catches the exact case where the
scan matcher slides.
"""

import json
import math
from enum import Enum
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


class Regime(Enum):
    NORMAL = 'normal'
    CLIMBING = 'climbing'
    DEGENERATE = 'degenerate'


class OdomSupervisor(Node):

    def __init__(self):
        super().__init__('odom_supervisor')

        self.declare_parameters('', [
            ('rate', 10.0),
            ('scan_degeneracy_ratio', 0.82),
            ('scan_min_returns', 80),
            ('wheel_stale_timeout', 1.0),
            ('scan_stale_timeout', 1.5),
            # Drift budget while dead reckoning, metres per second of climb.
            # Used only to report expected error, not to correct anything.
            ('climb_drift_rate', 0.05),
            # Covariance written onto a gated scan-matcher message. Large enough
            # that the EKF's gain for the source is numerically zero.
            ('gated_covariance', 1e6),
        ])
        g = self.get_parameter
        self.rate = g('rate').value
        self.degeneracy_ratio = g('scan_degeneracy_ratio').value
        self.scan_min_returns = g('scan_min_returns').value
        self.wheel_timeout = g('wheel_stale_timeout').value
        self.scan_timeout = g('scan_stale_timeout').value
        self.climb_drift_rate = g('climb_drift_rate').value
        self.gated_covariance = g('gated_covariance').value

        self.regime = Regime.NORMAL
        self._wheel_valid = False
        self._wheel_stamp: Optional[float] = None
        self._scan_stamp: Optional[float] = None
        self._climb_state = 'idle'
        self._climb_entered: Optional[float] = None
        self._degenerate = False
        self._degeneracy_score = 0.0
        self._accumulated_drift = 0.0
        self._filtered_cov_xy = 0.0

        sensor_qos = QoSProfile(depth=5,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)

        self.health_pub = self.create_publisher(String, '/odometry/health', 10)
        self.regime_pub = self.create_publisher(String, '/odometry/regime', 10)
        # The gated scan-matcher stream the EKF actually subscribes to.
        self.scan_odom_pub = self.create_publisher(Odometry, '/odom_scan', 10)

        self.create_subscription(Bool, '/odom_wheel/valid', self._on_wheel_valid, 10)
        self.create_subscription(Odometry, '/odom_wheel', self._on_wheel, 10)
        self.create_subscription(Odometry, '/odom_scan_raw', self._on_scan_odom, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self._on_filtered, 10)
        self.create_subscription(LaserScan, '/scan', self._on_scan, sensor_qos)
        self.create_subscription(String, '/climb/state', self._on_climb, 10)

        self.create_timer(1.0 / self.rate, self._tick)
        self.get_logger().info('odometry supervisor up')

    # ---------------------------------------------------------------- inputs

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_wheel_valid(self, msg: Bool):
        self._wheel_valid = msg.data

    def _on_wheel(self, msg: Odometry):
        self._wheel_stamp = self._now()

    def _on_scan_odom(self, msg: Odometry):
        """Gate the scan matcher into the EKF.

        Republished unchanged in NORMAL, and with an effectively infinite
        covariance otherwise. Dropping the message instead would be worse: the
        EKF would see a silent stream and its sensor_timeout diagnostics would
        report a fault where there is only a known-bad regime.
        """
        self._scan_stamp = self._now()
        gated = self.regime is not Regime.NORMAL
        if gated:
            big = self.gated_covariance
            for i in (0, 7, 14, 21, 28, 35):
                msg.pose.covariance[i] = big
                msg.twist.covariance[i] = big
        self.scan_odom_pub.publish(msg)

    def _on_filtered(self, msg: Odometry):
        self._filtered_cov_xy = math.sqrt(
            max(msg.pose.covariance[0], 0.0) + max(msg.pose.covariance[7], 0.0))

    def _on_climb(self, msg: String):
        try:
            state = json.loads(msg.data).get('state', 'idle')
        except json.JSONDecodeError:
            return
        climbing = state in ('approach', 'climb')
        if climbing and self._climb_entered is None:
            self._climb_entered = self._now()
            self._accumulated_drift = 0.0
        elif not climbing and self._climb_entered is not None:
            duration = self._now() - self._climb_entered
            self._accumulated_drift = duration * self.climb_drift_rate
            self.get_logger().warn(
                f'climb finished after {duration:.1f} s of dead reckoning; '
                f'expected position error ~{self._accumulated_drift:.2f} m. '
                f'Relocalisation required before precise goals are accepted.')
            self._climb_entered = None
        self._climb_state = state

    def _on_scan(self, msg: LaserScan):
        self._degeneracy_score, self._degenerate = self._scan_degeneracy(msg)

    def _scan_degeneracy(self, scan: LaserScan):
        """Fraction of scan structure aligned with a single axis.

        Each valid return contributes its surface direction, approximated by the
        beam angle. If nearly everything is perpendicular to one axis - two
        parallel walls and nothing else - the matcher is free to slide along the
        corridor and its along-axis estimate is not to be trusted.
        """
        xs = 0.0
        ys = 0.0
        count = 0
        angle = scan.angle_min
        for r in scan.ranges:
            a = angle
            angle += scan.angle_increment
            if not math.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            count += 1
            # Weight by 1/r: near structure constrains the match far more than a
            # distant wall, where a small angular error is a large position error.
            w = 1.0 / max(r, 0.1)
            xs += w * abs(math.cos(a))
            ys += w * abs(math.sin(a))

        if count < self.scan_min_returns:
            return 1.0, True                    # too little data: assume the worst
        total = xs + ys
        if total <= 1e-6:
            return 1.0, True
        ratio = max(xs, ys) / total
        return ratio, ratio > self.degeneracy_ratio

    # -------------------------------------------------------------- main loop

    def _classify(self) -> Regime:
        if self._climb_state in ('approach', 'climb'):
            return Regime.CLIMBING
        if self._degenerate:
            return Regime.DEGENERATE
        return Regime.NORMAL

    def _tick(self):
        new = self._classify()
        if new != self.regime:
            self.get_logger().info(
                f'odometry regime {self.regime.value} -> {new.value} '
                f'(degeneracy {self._degeneracy_score:.2f}, '
                f'climb "{self._climb_state}")')
            self.regime = new
            msg = String()
            msg.data = new.value
            self.regime_pub.publish(msg)

        now = self._now()
        wheel_fresh = (self._wheel_stamp is not None
                       and now - self._wheel_stamp < self.wheel_timeout)
        scan_fresh = (self._scan_stamp is not None
                      and now - self._scan_stamp < self.scan_timeout)

        if self._climb_entered is not None:
            live_drift = (now - self._climb_entered) * self.climb_drift_rate
        else:
            live_drift = self._accumulated_drift

        sources = {
            'wheel': {
                'fresh': wheel_fresh,
                'valid': self._wheel_valid,
                'used': wheel_fresh and self._wheel_valid and self.regime != Regime.CLIMBING,
            },
            'scan_match': {
                'fresh': scan_fresh,
                'valid': not self._degenerate,
                'used': scan_fresh and self.regime == Regime.NORMAL,
            },
            'imu': {'fresh': True, 'valid': True, 'used': True},
        }
        used = [k for k, v in sources.items() if v['used']]

        # A single honest quality number for consumers that only want one.
        if self.regime == Regime.CLIMBING:
            quality = 'dead_reckoning'
        elif len(used) >= 3:
            quality = 'good'
        elif len(used) == 2:
            quality = 'degraded'
        else:
            quality = 'poor'

        health = String()
        health.data = json.dumps({
            'stamp': now,
            'regime': self.regime.value,
            'quality': quality,
            'sources': sources,
            'active_sources': used,
            'scan_degeneracy': round(self._degeneracy_score, 3),
            'filtered_sigma_xy': round(self._filtered_cov_xy, 4),
            'dead_reckoning_drift_estimate': round(live_drift, 3),
            # The contract with the navigation and MCP layers: below this,
            # decline precise goals rather than pretending.
            'precise_goals_allowed': quality in ('good', 'degraded') and live_drift < 0.25,
        })
        self.health_pub.publish(health)


def main(args=None):
    rclpy.init(args=args)
    node = OdomSupervisor()
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
