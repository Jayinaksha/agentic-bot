#!/usr/bin/env python3
"""
Tri-star cluster drive controller.

Turns /cmd_vel into 16 joint velocities and publishes mode-aware wheel odometry.

Kinematics
----------
The platform is a skid-steer: left and right sides each get one rate.

    v_left  = v - (w * yaw_slip * track / 2)
    v_right = v + (w * yaw_slip * track / 2)

`yaw_slip` accounts for the lateral scrub of a four-contact skid-steer, which
makes the achieved yaw rate lower than the ideal differential prediction. It is
a measured constant, not a fudge factor - see scripts/calibrate_slip.py.

Transmission modes
------------------
ROLLING   side rate -> sub-wheel joints at v/r_sub, carrier joints held at 0.
          Odometry is valid: the contact radius really is r_sub.
TUMBLING  side rate -> carrier joints at v/r_cluster, sub-wheels held at 0.
          Odometry is NOT valid and is flagged as such, because during a tumble
          the contact point walks around the carrier and the wheel-to-ground
          transmission ratio is undefined. odom_supervisor watches the flag and
          leans on IMU integration for the duration.

The mode is commanded by climb_fsm on /locomotion/mode; this node never decides
to climb on its own, it only executes.
"""

import math
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String, Bool

from r2d2_locomotion.kinematics import (CLUSTERS, LEFT_CLUSTERS, MODE_ROLLING,
                                        MODE_TUMBLING, MODES, URDF_PHASES,
                                        body_twist, integrate_arc,
                                        joint_commands, rate_limit, side_speeds)


class TristarController(Node):

    def __init__(self):
        super().__init__('tristar_controller')

        self.declare_parameters('', [
            ('control_rate', 50.0),
            ('sub_wheel_radius', 0.050),
            ('cluster_circumradius', 0.115),
            ('track_width', 0.340),
            ('wheelbase', 0.260),
            ('yaw_slip_factor', 1.25),
            ('max_wheel_rate', 12.0),
            ('max_cluster_rate', 3.5),
            ('cmd_timeout', 0.5),
            ('max_lin_accel_roll', 1.2),
            ('max_lin_accel_climb', 0.35),
            ('carrier_hold_gain', 6.0),
            ('publish_tf', False),
            ('odom_frame', 'odom_wheel'),
            ('base_frame', 'base_link'),
        ])
        g = self.get_parameter
        self.rate = g('control_rate').value
        self.r_sub = g('sub_wheel_radius').value
        self.r_clu = g('cluster_circumradius').value
        self.track = g('track_width').value
        self.slip = g('yaw_slip_factor').value
        self.max_wheel = g('max_wheel_rate').value
        self.max_cluster = g('max_cluster_rate').value
        self.cmd_timeout = g('cmd_timeout').value
        self.acc_roll = g('max_lin_accel_roll').value
        self.acc_climb = g('max_lin_accel_climb').value
        self.hold_gain = g('carrier_hold_gain').value
        self.odom_frame = g('odom_frame').value
        self.base_frame = g('base_frame').value

        self.mode = MODE_ROLLING
        self.cmd = Twist()
        self.last_cmd_time = self.get_clock().now()

        # Rate-limited state, so a step change in /cmd_vel does not become a
        # step change in wheel torque.
        self._v_applied = 0.0
        self._w_applied = 0.0

        # Odometry integration state.
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._last_positions = None
        self._last_odom_time = None
        # Live carrier angles, needed to park the clusters at a known ride
        # height in rolling mode rather than wherever they last stopped.
        self._carrier_positions = {}

        sensor_qos = QoSProfile(depth=10,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)

        self.cmd_pub = self.create_publisher(
            Float64MultiArray, '/tristar_velocity_controller/commands', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom_wheel', 10)
        self.odom_valid_pub = self.create_publisher(Bool, '/odom_wheel/valid', 10)

        self.create_subscription(Twist, '/cmd_vel', self._on_cmd, 10)
        self.create_subscription(String, '/locomotion/mode', self._on_mode, 10)
        self.create_subscription(JointState, '/joint_states', self._on_joints, sensor_qos)

        self.create_timer(1.0 / self.rate, self._tick)

        self.get_logger().info(
            f'tri-star controller up: r_sub={self.r_sub} m, r_cluster={self.r_clu} m, '
            f'track={self.track} m, yaw_slip={self.slip}')

    # ---------------------------------------------------------------- inputs

    def _on_cmd(self, msg: Twist):
        self.cmd = msg
        self.last_cmd_time = self.get_clock().now()

    def _on_mode(self, msg: String):
        new = msg.data.strip().lower()
        if new not in MODES:
            self.get_logger().warn(f'ignoring unknown locomotion mode "{msg.data}"')
            return
        if new != self.mode:
            self.get_logger().info(f'transmission mode {self.mode} -> {new}')
            # Reset the odometry baseline: joint positions accumulated in the
            # old mode must not be differenced against the new mode.
            self._last_positions = None
            self.mode = new

    # ------------------------------------------------------------- kinematics

    def _build_command(self, v_left: float, v_right: float) -> List[float]:
        return joint_commands(v_left, v_right, self.mode,
                              self.r_sub, self.r_clu,
                              self.max_wheel, self.max_cluster,
                              carrier_positions=self._carrier_positions,
                              urdf_phases=URDF_PHASES,
                              hold_gain=self.hold_gain)

    # ---------------------------------------------------------------- odometry

    def _on_joints(self, msg: JointState):
        """Integrate wheel odometry from joint positions.

        Only meaningful in ROLLING mode. In TUMBLING mode the pose is frozen and
        the validity flag goes false, which is the honest thing to publish: a
        tumbling cluster has no fixed contact radius, so any number produced
        here would be fiction that the EKF would happily fuse.
        """
        now = self.get_clock().now()
        name_to_pos = dict(zip(msg.name, msg.position))

        for cluster in CLUSTERS:
            joint = f'{cluster}_carrier_joint'
            if joint in name_to_pos:
                self._carrier_positions[cluster] = name_to_pos[joint]

        if self.mode != MODE_ROLLING:
            self._last_positions = None
            self._last_odom_time = now
            self._publish_odom(valid=False)
            return

        left_names = [f'{c}_wheel0_joint' for c in CLUSTERS if c in LEFT_CLUSTERS]
        right_names = [f'{c}_wheel0_joint' for c in CLUSTERS if c not in LEFT_CLUSTERS]
        if not all(n in name_to_pos for n in left_names + right_names):
            return

        left = sum(name_to_pos[n] for n in left_names) / len(left_names)
        right = sum(name_to_pos[n] for n in right_names) / len(right_names)

        if self._last_positions is None or self._last_odom_time is None:
            self._last_positions = (left, right)
            self._last_odom_time = now
            return

        dt = (now - self._last_odom_time).nanoseconds * 1e-9
        if dt <= 0.0:
            return

        d_left = (left - self._last_positions[0]) * self.r_sub
        d_right = (right - self._last_positions[1]) * self.r_sub
        self._last_positions = (left, right)
        self._last_odom_time = now

        d_center, d_yaw = body_twist(d_left, d_right, self.track, self.slip)
        self._x, self._y, self._yaw = integrate_arc(
            self._x, self._y, self._yaw, d_center, d_yaw)

        self._publish_odom(valid=True, vx=d_center / dt, wz=d_yaw / dt)

    def _publish_odom(self, valid: bool, vx: float = 0.0, wz: float = 0.0):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame
        msg.pose.pose.position.x = self._x
        msg.pose.pose.position.y = self._y
        msg.pose.pose.orientation.z = math.sin(self._yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(self._yaw / 2.0)
        msg.twist.twist.linear.x = vx
        msg.twist.twist.angular.z = wz

        # Covariance is the contract with the EKF. Skid-steer yaw from wheels is
        # poor even on flat ground, so it is inflated by an order of magnitude
        # relative to x; when invalid it is set huge so any fusion is a no-op.
        big = 1e6
        pos_var = 0.02 if valid else big
        yaw_var = 0.20 if valid else big
        msg.pose.covariance[0] = pos_var
        msg.pose.covariance[7] = pos_var
        msg.pose.covariance[35] = yaw_var
        msg.twist.covariance[0] = pos_var
        msg.twist.covariance[35] = yaw_var

        self.odom_pub.publish(msg)
        flag = Bool()
        flag.data = valid
        self.odom_valid_pub.publish(flag)

    # -------------------------------------------------------------- main loop

    def _tick(self):
        dt = 1.0 / self.rate
        now = self.get_clock().now()
        stale = (now - self.last_cmd_time).nanoseconds * 1e-9 > self.cmd_timeout

        v_target = 0.0 if stale else self.cmd.linear.x
        w_target = 0.0 if stale else self.cmd.angular.z

        accel = self.acc_climb if self.mode == MODE_TUMBLING else self.acc_roll
        self._v_applied = rate_limit(v_target, self._v_applied, accel, dt)
        self._w_applied = rate_limit(w_target, self._w_applied, accel * 3.0, dt)

        v_left, v_right = side_speeds(self._v_applied, self._w_applied,
                                      self.track, self.slip)

        msg = Float64MultiArray()
        msg.data = self._build_command(v_left, v_right)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TristarController()
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
