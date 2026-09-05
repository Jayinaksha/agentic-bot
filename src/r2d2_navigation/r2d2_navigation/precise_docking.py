#!/usr/bin/env python3
"""
Final-approach servo for centimetre-level positioning.

Why this exists
---------------
Nav2 on a skid-steer reliably delivers about +/-0.20 m and +/-0.25 rad. That is
fine for "go to the kitchen" and useless for "stop square in front of the
fridge so the camera can read the label". Tightening Nav2's goal checker does
not fix it: the error is not the checker being lenient, it is that the global
pose itself carries that much uncertainty, and the controller is chasing a plan
expressed in a frame that is only good to a few centimetres.

The fix is to stop using the global frame for the last stretch. This node
servos on the *live laser scan* against a local geometric feature, so the
control loop closes on what the robot can see right now rather than on where it
believes it is. Map error drops out of the loop entirely.

Two target types, both extracted from the same 2D scan:

    surface   Fit a line to the returns in a window around the target bearing,
              then drive to a pose a fixed standoff from that line, normal to
              it. This is the fridge door, the wall below a light switch, the
              front face of a cabinet.
    gap       Find the two edges of an opening and drive to its centre, squared
              to the opening. This is a doorway, and it is what makes a 0.90 m
              door passable by a 0.30 m robot without clipping the frame.

Control is a straightforward decoupled P law with deadbands, run at 20 Hz on
fresh scans. There is no need for anything cleverer: the residual is small, the
platform is holonomic in yaw, and every extra term is another thing to tune.

Safety: the servo commands through /cmd_vel_nav like everything else, so the
climb FSM and the collision monitor still gate it. It gives up rather than
grinding if the feature disappears or the residual stops shrinking.
"""

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from r2d2_navigation.scan_features import (fit_line, find_gap, line_pose_error,
                                           gap_pose_error, polar_to_xy)


class PreciseDocking(Node):

    def __init__(self):
        super().__init__('precise_docking')

        self.declare_parameters('', [
            ('rate', 20.0),
            ('standoff', 0.35),          # m from the surface to stop at
            ('window_deg', 40.0),        # +/- around the target bearing to search
            ('xy_tolerance', 0.02),      # m
            ('yaw_tolerance', 0.035),    # rad (~2 deg)
            ('k_linear', 0.9),
            ('k_lateral', 1.1),
            ('k_yaw', 1.6),
            ('max_linear', 0.12),
            ('max_angular', 0.5),
            ('min_linear', 0.02),        # below this the wheels stall, not creep
            ('min_angular', 0.06),
            ('timeout', 25.0),
            ('stall_timeout', 6.0),
            ('min_inliers', 12),
            ('line_inlier_dist', 0.03),  # m, RANSAC-ish inlier band
        ])
        g = self.get_parameter
        self.rate = g('rate').value
        self.standoff = g('standoff').value
        self.window = math.radians(g('window_deg').value)
        self.xy_tol = g('xy_tolerance').value
        self.yaw_tol = g('yaw_tolerance').value
        self.k_lin = g('k_linear').value
        self.k_lat = g('k_lateral').value
        self.k_yaw = g('k_yaw').value
        self.max_lin = g('max_linear').value
        self.max_ang = g('max_angular').value
        self.min_lin = g('min_linear').value
        self.min_ang = g('min_angular').value
        self.timeout = g('timeout').value
        self.stall_timeout = g('stall_timeout').value
        self.min_inliers = g('min_inliers').value
        self.inlier_dist = g('line_inlier_dist').value

        self._scan: Optional[LaserScan] = None

        sensor_qos = QoSProfile(depth=5,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(LaserScan, '/scan', self._on_scan, sensor_qos)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        self.status_pub = self.create_publisher(String, '/docking/status', 10)

        self.get_logger().info(
            f'precise docking up: standoff {self.standoff:.2f} m, '
            f'tolerance {self.xy_tol*100:.0f} mm / '
            f'{math.degrees(self.yaw_tol):.1f} deg')

    def _on_scan(self, msg: LaserScan):
        self._scan = msg

    # ----------------------------------------------------------------- public

    def dock(self, target_bearing: float, target_type: str = 'surface',
             standoff: Optional[float] = None) -> Tuple[bool, str]:
        """Blocking servo onto a feature at `target_bearing` radians.

        Returns (success, message). Intended to be driven from the MCP tool
        layer, which already runs its calls off the main executor thread.
        """
        standoff = self.standoff if standoff is None else standoff
        start = self._now()
        best_residual = float('inf')
        best_at = start
        period = 1.0 / self.rate

        while rclpy.ok():
            now = self._now()
            if now - start > self.timeout:
                self._halt()
                return False, f'docking timed out after {self.timeout:.0f} s'
            if now - best_at > self.stall_timeout:
                self._halt()
                return False, (f'docking stalled with {best_residual*100:.1f} cm '
                               f'of residual error')

            scan = self._scan
            if scan is None:
                rclpy.spin_once(self, timeout_sec=period)
                continue

            error = self._feature_error(scan, target_bearing, target_type, standoff)
            if error is None:
                self._halt()
                return False, (f'no {target_type} found within '
                               f'{math.degrees(self.window):.0f} deg of bearing '
                               f'{math.degrees(target_bearing):.0f} deg')

            e_forward, e_lateral, e_yaw = error
            residual = math.hypot(e_forward, e_lateral)

            if residual < self.xy_tol and abs(e_yaw) < self.yaw_tol:
                self._halt()
                return True, (f'docked: {residual*100:.1f} cm, '
                              f'{math.degrees(e_yaw):.1f} deg from target')

            if residual < best_residual - 0.005:
                best_residual, best_at = residual, now

            self.cmd_pub.publish(self._control(e_forward, e_lateral, e_yaw))
            self._publish_status(target_type, e_forward, e_lateral, e_yaw)
            rclpy.spin_once(self, timeout_sec=period)

        self._halt()
        return False, 'docking interrupted'

    # ------------------------------------------------------------------ inner

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _feature_error(self, scan: LaserScan, bearing: float, target_type: str,
                       standoff: float):
        points = self._points_in_window(scan, bearing)
        if len(points) < self.min_inliers:
            return None

        if target_type == 'gap':
            gap = find_gap(points)
            if gap is None:
                return None
            return gap_pose_error(gap, standoff)

        line = fit_line(points, self.inlier_dist, self.min_inliers)
        if line is None:
            return None
        return line_pose_error(line, standoff)

    def _points_in_window(self, scan: LaserScan,
                          bearing: float) -> List[Tuple[float, float]]:
        out = []
        angle = scan.angle_min
        for r in scan.ranges:
            a = angle
            angle += scan.angle_increment
            if abs(_wrap(a - bearing)) > self.window:
                continue
            if not math.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            out.append(polar_to_xy(r, a))
        return out

    def _control(self, e_forward: float, e_lateral: float, e_yaw: float) -> Twist:
        """Decoupled P law with stiction deadbands.

        A skid-steer cannot move sideways, so lateral error is corrected by
        turning into it: the yaw command carries both the heading error and a
        lateral term. Below min_linear/min_angular the motors stall instead of
        creeping, so commands in that band are pushed up to the threshold rather
        than left to buzz.
        """
        cmd = Twist()

        if abs(e_forward) > self.xy_tol:
            v = _clamp(self.k_lin * e_forward, self.max_lin)
            cmd.linear.x = _deadband(v, self.min_lin)

        w = self.k_yaw * e_yaw + self.k_lat * e_lateral
        if abs(w) > 1e-3:
            w = _clamp(w, self.max_ang)
            cmd.angular.z = _deadband(w, self.min_ang)

        # Turning and driving at once smears the correction on a skid-steer,
        # where yaw costs lateral scrub. Fix heading first, then close distance.
        if abs(cmd.angular.z) > self.min_ang * 1.5:
            cmd.linear.x *= 0.3

        return cmd

    def _halt(self):
        self.cmd_pub.publish(Twist())

    def _publish_status(self, target_type, e_forward, e_lateral, e_yaw):
        msg = String()
        msg.data = (f'{{"target":"{target_type}",'
                    f'"forward":{e_forward:.4f},'
                    f'"lateral":{e_lateral:.4f},'
                    f'"yaw":{e_yaw:.4f}}}')
        self.status_pub.publish(msg)


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _deadband(value: float, minimum: float) -> float:
    """Push a small non-zero command up to the motors' stiction threshold."""
    if value == 0.0:
        return 0.0
    return math.copysign(max(abs(value), minimum), value)


def main(args=None):
    rclpy.init(args=args)
    node = PreciseDocking()
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
