#!/usr/bin/env python3
"""
Terrain monitor: turns four cheap ToF beams plus an IMU into traversability.

This is the whole terrain-understanding budget of the robot - about USD 15 of
sensing and well under 1% of a CPU core. There is no depth camera and no 3D
LiDAR, so the job is to extract the maximum from a very sparse signal.

Geometry
--------
Each ToF is mounted `h` above the floor looking `tilt` radians below horizontal.
On level ground the return is a constant

    r_flat = h / sin(tilt)

A step up shortens the return, a step down or a stair nosing lengthens it. The
implied floor-height change under the beam spot is

    dz = h - r * sin(tilt + pitch)

where `pitch` is the IMU-derived body pitch, which matters a great deal: without
it, driving onto the 12 degree ramp reads as a wall of phantom steps.

Classification
--------------
    |dz| < step_up_min                     flat (or a doorway sill: drive over)
    step_up_min <= dz <= step_up_max       climbable riser -> tell climb_fsm
    dz > step_up_max                       obstacle, cluster cannot mount it
    dz < -cliff_drop                       fall hazard (stair top, landing edge)

Outputs
-------
/terrain/hazards   PointCloud2, consumed directly by the Nav2 obstacle layer.
                   Only genuine no-go returns become points; climbable risers
                   deliberately do NOT, otherwise Nav2 would plan around the
                   staircase the robot is supposed to use.
/terrain/state     JSON status for the MCP layer and the climb FSM.
/terrain/estop     Latching attitude e-stop. Independent of the planner.
/terrain/attitude  Filtered roll/pitch.
"""

import json
import math
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Imu, LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Header, String

from r2d2_locomotion.kinematics import (BLOCKED, CLIFF, RISER, classify_delta,
                                        flat_return, ground_spot_distance,
                                        height_delta)

TOF_TOPICS = {
    'front_left': '/tof/front_left',
    'front_right': '/tof/front_right',
    'rear_left': '/tof/rear_left',
    'rear_right': '/tof/rear_right',
}
FRONT = ('front_left', 'front_right')
REAR = ('rear_left', 'rear_right')


class TerrainMonitor(Node):

    def __init__(self):
        super().__init__('terrain_monitor')

        self.declare_parameters('', [
            ('rate', 20.0),
            ('tof_mount_height', 0.180),
            ('tof_tilt', 0.5236),
            ('tof_forward_offset', 0.180),
            ('step_up_min', 0.035),
            ('step_up_max', 0.150),
            ('cliff_drop', 0.060),
            ('range_valid_max', 1.20),
            ('max_pitch_flat', 0.17),
            ('max_pitch_climb', 0.70),
            ('max_roll', 0.35),
            ('attitude_filter_alpha', 0.25),
            ('hazard_frame', 'base_link'),
            ('hazard_point_height', 0.35),
        ])
        g = self.get_parameter
        self.rate = g('rate').value
        self.h = g('tof_mount_height').value
        self.tilt = g('tof_tilt').value
        self.fwd = g('tof_forward_offset').value
        self.step_min = g('step_up_min').value
        self.step_max = g('step_up_max').value
        self.cliff = g('cliff_drop').value
        self.range_max = g('range_valid_max').value
        self.pitch_flat = g('max_pitch_flat').value
        self.pitch_climb = g('max_pitch_climb').value
        self.roll_max = g('max_roll').value
        self.alpha = g('attitude_filter_alpha').value
        self.hazard_frame = g('hazard_frame').value
        self.hazard_z = g('hazard_point_height').value

        self.r_flat = flat_return(self.h, self.tilt)
        self.spot_ahead = ground_spot_distance(self.h, self.tilt)

        self._ranges: Dict[str, Optional[float]] = {k: None for k in TOF_TOPICS}
        self._roll = 0.0
        self._pitch = 0.0
        self._have_imu = False
        self._estop_latched = False
        self._estop_reason = ''
        self._climb_mode = False

        sensor_qos = QoSProfile(depth=5,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)

        for key, topic in TOF_TOPICS.items():
            self.create_subscription(
                LaserScan, topic,
                lambda msg, k=key: self._on_tof(k, msg), sensor_qos)
        self.create_subscription(Imu, '/imu/data', self._on_imu, sensor_qos)
        self.create_subscription(String, '/locomotion/mode', self._on_mode, 10)
        self.create_subscription(Bool, '/terrain/estop_reset', self._on_estop_reset, 10)

        self.hazard_pub = self.create_publisher(PointCloud2, '/terrain/hazards', 5)
        self.state_pub = self.create_publisher(String, '/terrain/state', 10)
        self.estop_pub = self.create_publisher(Bool, '/terrain/estop', 10)
        self.att_pub = self.create_publisher(Vector3Stamped, '/terrain/attitude', 10)

        self.create_timer(1.0 / self.rate, self._tick)
        self.get_logger().info(
            f'terrain monitor up: flat return {self.r_flat:.3f} m, '
            f'spot {self.spot_ahead:.3f} m ahead, '
            f'riser window [{self.step_min:.3f}, {self.step_max:.3f}] m')

    # ---------------------------------------------------------------- inputs

    def _on_tof(self, key: str, msg: LaserScan):
        if not msg.ranges:
            self._ranges[key] = None
            return
        r = msg.ranges[0]
        if not math.isfinite(r) or r <= msg.range_min or r > self.range_max:
            # An out-of-range return means the beam found no floor within the
            # window, which for a downward beam is itself evidence of a drop.
            self._ranges[key] = math.inf
            return
        self._ranges[key] = r

    def _on_imu(self, msg: Imu):
        q = msg.orientation
        # roll/pitch from quaternion (REP-103: +pitch is nose-down about +y).
        sinr = 2.0 * (q.w * q.x + q.y * q.z)
        cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr, cosr)
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))

        if not self._have_imu:
            self._roll, self._pitch, self._have_imu = roll, pitch, True
        else:
            self._roll += self.alpha * (roll - self._roll)
            self._pitch += self.alpha * (pitch - self._pitch)

    def _on_mode(self, msg: String):
        self._climb_mode = msg.data.strip().lower() == 'tumbling'

    def _on_estop_reset(self, msg: Bool):
        if msg.data and self._estop_latched:
            self.get_logger().warn(f'terrain e-stop cleared (was: {self._estop_reason})')
            self._estop_latched = False
            self._estop_reason = ''

    # ------------------------------------------------------------ classifying

    def _height_delta(self, key: str) -> Optional[float]:
        """Floor-height change under one beam spot, metres. + is up."""
        return height_delta(self._ranges.get(key), self.h, self.tilt,
                            self._pitch, self.cliff, rear=key in REAR)

    def _classify(self, dz: Optional[float]) -> str:
        return classify_delta(dz, self.step_min, self.step_max, self.cliff)

    def _check_attitude(self):
        limit = self.pitch_climb if self._climb_mode else (self.pitch_flat * 2.5)
        if abs(self._pitch) > limit:
            self._latch(f'pitch {math.degrees(self._pitch):.1f} deg exceeds '
                        f'{math.degrees(limit):.1f} deg')
        if abs(self._roll) > self.roll_max:
            self._latch(f'roll {math.degrees(self._roll):.1f} deg exceeds '
                        f'{math.degrees(self.roll_max):.1f} deg')

    def _latch(self, reason: str):
        if not self._estop_latched:
            self.get_logger().error(f'TERRAIN E-STOP: {reason}')
            self._estop_latched = True
            self._estop_reason = reason

    # -------------------------------------------------------------- main loop

    def _tick(self):
        if not self._have_imu:
            return

        self._check_attitude()

        deltas = {k: self._height_delta(k) for k in TOF_TOPICS}
        classes = {k: self._classify(dz) for k, dz in deltas.items()}

        # A riser only counts if BOTH front beams agree. One beam seeing a step
        # and the other flat means the robot is skewed to the flight, or looking
        # at a chair leg; either way it must not commit to a climb.
        front_classes = [classes[k] for k in FRONT]
        front_deltas = [deltas[k] for k in FRONT if deltas[k] is not None]
        riser_ahead = all(c == RISER for c in front_classes)
        cliff_ahead = any(c == CLIFF for c in front_classes)
        blocked_ahead = any(c == BLOCKED for c in front_classes)
        cliff_behind = any(classes[k] == CLIFF for k in REAR)

        # Skew: the difference between the two front beams tells us how square
        # the platform is to the riser. climb_fsm servos this to zero.
        skew = None
        if deltas['front_left'] is not None and deltas['front_right'] is not None:
            skew = deltas['front_left'] - deltas['front_right']

        self._publish_hazards(classes, deltas)

        att = Vector3Stamped()
        att.header.stamp = self.get_clock().now().to_msg()
        att.header.frame_id = self.hazard_frame
        att.vector.x = self._roll
        att.vector.y = self._pitch
        self.att_pub.publish(att)

        estop = Bool()
        estop.data = self._estop_latched
        self.estop_pub.publish(estop)

        state = String()
        state.data = json.dumps({
            'stamp': self.get_clock().now().nanoseconds * 1e-9,
            'roll': round(self._roll, 4),
            'pitch': round(self._pitch, 4),
            'on_slope': abs(self._pitch) > self.pitch_flat,
            'riser_ahead': riser_ahead,
            'riser_height': round(sum(front_deltas) / len(front_deltas), 4)
                            if riser_ahead and front_deltas else None,
            'cliff_ahead': cliff_ahead,
            'cliff_behind': cliff_behind,
            'blocked_ahead': blocked_ahead,
            'skew': round(skew, 4) if skew is not None else None,
            'estop': self._estop_latched,
            'estop_reason': self._estop_reason,
            'beams': {k: {'dz': round(v, 4) if v is not None else None,
                          'class': classes[k]} for k, v in deltas.items()},
        })
        self.state_pub.publish(state)

    def _publish_hazards(self, classes, deltas):
        """Emit only true no-go returns as obstacle points.

        A climbable riser is intentionally absent: it is the robot's route
        upstairs, and putting it in the costmap would make Nav2 plan around the
        one feature the platform was built to use.
        """
        points = []
        for key, cls in classes.items():
            if cls not in (CLIFF, BLOCKED):
                continue
            sign = 1.0 if key in FRONT else -1.0
            x = sign * (self.fwd + self.spot_ahead)
            y = 0.13 if key.endswith('left') else -0.13
            # A short vertical stack, because the obstacle layer marks by ray
            # and a single point at floor level is easy for it to miss.
            for k in range(4):
                points.append((x, y, 0.05 + k * self.hazard_z / 4.0))

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.hazard_frame
        self.hazard_pub.publish(point_cloud2.create_cloud_xyz32(header, points))


def main(args=None):
    rclpy.init(args=args)
    node = TerrainMonitor()
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
