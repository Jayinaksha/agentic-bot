#!/usr/bin/env python3
"""
Multi-floor map manager.

A two-storey house is not one map. Directly above the kitchen is the bedroom,
and a single occupancy grid would fuse the two into nonsense. The standard
answer, and the one used here, is a stack of independent 2D maps indexed by
floor, with explicit transition nodes joining them.

    floor 0  map: house_f0.yaml   z range [-0.5, 1.2)
    floor 1  map: house_f1.yaml   z range [ 1.2, 2.9)
                  ^ transition: staircase foot (x, y) <-> staircase head (x, y)

Responsibilities:

  * Decide which floor the robot is on. Floor is NOT read from the EKF: the
    filter runs in two_d_mode, so its z is identically zero, and turning that
    off just to get a floor index would buy a badly drifting height from double
    integrated accelerometer noise. Instead the floor is a discrete counter
    advanced by completed climb events - climb_fsm reports the IMU-integrated
    rise of each flight, and a flight that gained roughly one storey moves the
    counter one floor in the direction of travel. Floors change only when the
    robot climbs, so an event counter is both cheaper and more reliable than a
    continuous estimate.
  * Swap the active map when the floor changes, by re-loading nav2_map_server.
  * Force relocalisation on arrival. After a climb the pose estimate has drifted
    - it was dead reckoning on an IMU for the whole flight - so AMCL is seeded
    at the known head-of-stairs pose and asked to re-converge.
  * Publish the floor and the transition graph so the MCP layer can plan across
    floors instead of guessing.

Maps are expected under `map_directory` as `<map_prefix><floor>.yaml`. When a
floor has never been mapped, the manager says so rather than silently serving
an empty grid, because "the map is empty" and "the robot is lost" look
identical to a planner and very different to a person.
"""

import json
import math
import os
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile

from geometry_msgs.msg import PoseWithCovarianceStamped, Vector3Stamped
from nav2_msgs.srv import LoadMap
from std_msgs.msg import Int32, String
from std_srvs.srv import Empty


class FloorManager(Node):

    def __init__(self):
        super().__init__('floor_manager')

        self.declare_parameters('', [
            ('map_directory', os.path.expanduser('~/r2d2_maps')),
            ('map_prefix', 'house_f'),
            # Floor boundaries in metres of EKF z. Derived from the world:
            # stairs.riser * stairs.steps = 1.80 m floor-to-floor.
            ('floor_heights', [0.0, 1.80]),
            # A completed climb counts as a storey change when its integrated
            # rise is within this fraction of the nominal floor-to-floor height.
            # Wide, because IMU double integration over a 12-step flight is a
            # rough number; it only has to distinguish "a flight" from "a sill".
            ('storey_rise_tolerance', 0.45),
            ('min_storey_rise', 0.50),
            ('relocalize_after_transition', True),
            ('relocalize_cov_xy', 0.35),
            ('relocalize_cov_yaw', 0.30),
            # Transition graph, flattened because ROS 2 parameters do not take
            # nested structures: [from_floor, to_floor, foot_x, foot_y,
            # head_x, head_y, heading] repeated.
            #
            # The real values come from config/floors.yaml, which the Gazebo
            # world generator writes from the same constants it builds the
            # staircase from. This default is EMPTY on purpose: a plausible
            # hard-coded staircase is worse than none, because the robot drives
            # confidently to a flight that is not there. An empty graph makes
            # the navigation layer say so instead.
            ('transitions', []),
        ])
        g = self.get_parameter
        self.map_dir = g('map_directory').value
        self.map_prefix = g('map_prefix').value
        self.floor_heights: List[float] = list(g('floor_heights').value)
        self.rise_tolerance = g('storey_rise_tolerance').value
        self.min_storey_rise = g('min_storey_rise').value
        self.do_relocalize = g('relocalize_after_transition').value
        self.cov_xy = g('relocalize_cov_xy').value
        self.cov_yaw = g('relocalize_cov_yaw').value
        self.transitions = self._parse_transitions(g('transitions').value)

        self.current_floor = 0
        self._pending_floor: Optional[int] = None
        self._climb_state = 'idle'
        self._last_climb_rise = 0.0
        self._pitch = 0.0

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)

        self.floor_pub = self.create_publisher(Int32, '/floor/current', latched)
        self.graph_pub = self.create_publisher(String, '/floor/graph', latched)
        self.initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)

        self.create_subscription(String, '/climb/state', self._on_climb, 10)
        self.create_subscription(Vector3Stamped, '/terrain/attitude', self._on_attitude, 10)
        self.create_subscription(Int32, '/floor/set', self._on_set_floor, 10)

        self.load_map_client = self.create_client(LoadMap, '/map_server/load_map')
        self.clear_costmaps_clients = [
            self.create_client(Empty, '/local_costmap/clear_entirely_local_costmap'),
            self.create_client(Empty, '/global_costmap/clear_entirely_global_costmap'),
        ]

        self._publish_floor()
        self._publish_graph()
        self.create_timer(0.5, self._tick)

        if not self.transitions and len(self.floor_heights) > 1:
            self.get_logger().error(
                f'{len(self.floor_heights)} floors are configured but no '
                f'transitions between them, so no cross-floor route can be '
                f'planned. Load config/floors.yaml, which '
                f'src/r2d2_sim/worlds/generate_house.py writes from the same '
                f'constants the staircase is built from.')

        self.get_logger().info(
            f'floor manager up: {len(self.floor_heights)} floors, '
            f'maps in {self.map_dir}, {len(self.transitions)} transitions')

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _parse_transitions(flat: List[float]) -> List[Dict]:
        """Unflatten the transition parameter into usable records."""
        out = []
        stride = 7
        if len(flat) % stride:
            raise ValueError(
                f'transitions must be a multiple of {stride} values, got {len(flat)}')
        for i in range(0, len(flat), stride):
            f, t, fx, fy, hx, hy, heading = flat[i:i + stride]
            out.append({
                'from_floor': int(f),
                'to_floor': int(t),
                'foot': {'x': fx, 'y': fy},
                'head': {'x': hx, 'y': hy},
                'heading': heading,
            })
        return out

    def _map_path(self, floor: int) -> str:
        return os.path.join(self.map_dir, f'{self.map_prefix}{floor}.yaml')

    def _floor_after_climb(self, rise: float, ascending: bool) -> Optional[int]:
        """Which floor a finished flight left the robot on, or None.

        `rise` is climb_fsm's IMU-integrated vertical gain over the flight. It
        is compared against the nominal storey height rather than used as an
        absolute altitude, because the integration is only good to a few tens of
        centimetres over a flight - plenty to tell one storey from none, useless
        as a height.
        """
        if abs(rise) < self.min_storey_rise:
            # Climbed something, but not a storey: a threshold or a single step.
            return None

        step = 1 if ascending else -1
        target = self.current_floor + step
        if not 0 <= target < len(self.floor_heights):
            self.get_logger().warn(
                f'climb of {rise:.2f} m implies floor {target}, which does not '
                f'exist; staying on floor {self.current_floor}')
            return None

        expected = abs(self.floor_heights[target] - self.floor_heights[self.current_floor])
        if expected > 0 and abs(abs(rise) - expected) > self.rise_tolerance * expected:
            self.get_logger().warn(
                f'climb rise {abs(rise):.2f} m does not match the {expected:.2f} m '
                f'between floors {self.current_floor} and {target}; not switching. '
                f'Force it with: ros2 topic pub --once /floor/set std_msgs/Int32 '
                f'"{{data: {target}}}"')
            return None
        return target

    def transition_for(self, from_floor: int, to_floor: int) -> Optional[Dict]:
        """The transition joining two floors, in either direction."""
        for t in self.transitions:
            if t['from_floor'] == from_floor and t['to_floor'] == to_floor:
                return t
            if t['from_floor'] == to_floor and t['to_floor'] == from_floor:
                # Reverse: descending uses the same flight the other way round.
                return {
                    'from_floor': from_floor,
                    'to_floor': to_floor,
                    'foot': t['head'],
                    'head': t['foot'],
                    'heading': t['heading'] + math.pi,
                }
        return None

    # ---------------------------------------------------------------- inputs

    def _on_attitude(self, msg: Vector3Stamped):
        # REP-103: positive pitch is nose-down, so a nose-up climb reads
        # negative. Sampled continuously; what matters is its sign during the
        # flight, captured when the climb ends.
        self._pitch = msg.vector.y

    def _on_climb(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        state = payload.get('state', 'idle')

        if state in ('climb', 'settle'):
            rise = payload.get('rise')
            if rise is not None:
                self._last_climb_rise = float(rise)

        # Only re-evaluate once the flight is over and the platform has settled.
        # Mid-climb the robot is genuinely between floors and neither map applies.
        if self._climb_state == 'settle' and state == 'idle':
            ascending = self._last_climb_rise >= 0.0
            target = self._floor_after_climb(self._last_climb_rise, ascending)
            if target is not None:
                self.get_logger().info(
                    f'flight of {self._last_climb_rise:+.2f} m completed: '
                    f'floor {self.current_floor} -> {target}')
                self._pending_floor = target
            self._last_climb_rise = 0.0
        self._climb_state = state

    def _on_set_floor(self, msg: Int32):
        """Manual override, used when the robot is carried between floors."""
        if 0 <= msg.data < len(self.floor_heights):
            self.get_logger().info(f'floor forced to {msg.data} by operator')
            self._pending_floor = msg.data
        else:
            self.get_logger().warn(f'ignoring out-of-range floor {msg.data}')

    # -------------------------------------------------------------- main loop

    def _tick(self):
        if self._climb_state not in ('idle', 'settle'):
            return
        target = self._pending_floor
        self._pending_floor = None
        if target is None or target == self.current_floor:
            return
        self._switch_floor(target)

    def _switch_floor(self, floor: int):
        path = self._map_path(floor)
        if not os.path.exists(path):
            self.get_logger().error(
                f'floor {floor} selected but {path} does not exist. Map that '
                f'floor first (slam.launch.py floor:={floor}); refusing to '
                f'serve an empty grid, which a planner cannot distinguish from '
                f'being lost.')
            return

        previous = self.current_floor
        self.get_logger().info(f'switching floor {previous} -> {floor} ({path})')

        if self.load_map_client.wait_for_service(timeout_sec=2.0):
            req = LoadMap.Request()
            req.map_url = path
            future = self.load_map_client.call_async(req)
            future.add_done_callback(
                lambda f, fl=floor, pv=previous: self._on_map_loaded(f, fl, pv))
        else:
            self.get_logger().error('/map_server/load_map unavailable; floor not switched')
            return

        self.current_floor = floor
        self._publish_floor()

    def _on_map_loaded(self, future, floor: int, previous: int):
        try:
            result = future.result()
        except Exception as exc:                      # noqa: BLE001 - service errors vary
            self.get_logger().error(f'map load failed: {exc}')
            return
        if result.result != LoadMap.Response.RESULT_SUCCESS:
            self.get_logger().error(f'map server rejected {floor}: code {result.result}')
            return

        for client in self.clear_costmaps_clients:
            if client.wait_for_service(timeout_sec=1.0):
                client.call_async(Empty.Request())

        if self.do_relocalize:
            self._seed_pose(previous, floor)

    def _seed_pose(self, previous: int, floor: int):
        """Re-seed AMCL at the known head-of-flight pose.

        After a climb the filter has been dead reckoning on an IMU for the whole
        flight, so its x/y is wrong by an amount that grows with flight length.
        The one thing we do know is where the robot physically must be: standing
        at the head of the staircase it just climbed.
        """
        t = self.transition_for(previous, floor)
        if t is None:
            self.get_logger().warn(
                f'no transition recorded between floors {previous} and {floor}; '
                f'AMCL will have to converge from wherever it thinks it is')
            return

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(t['head']['x'])
        msg.pose.pose.position.y = float(t['head']['y'])
        msg.pose.pose.orientation.z = math.sin(t['heading'] / 2.0)
        msg.pose.pose.orientation.w = math.cos(t['heading'] / 2.0)
        msg.pose.covariance[0] = self.cov_xy ** 2
        msg.pose.covariance[7] = self.cov_xy ** 2
        msg.pose.covariance[35] = self.cov_yaw ** 2
        self.initialpose_pub.publish(msg)
        self.get_logger().info(
            f'seeded AMCL at head of flight ({t["head"]["x"]:.2f}, '
            f'{t["head"]["y"]:.2f}) on floor {floor}')

    def _publish_floor(self):
        msg = Int32()
        msg.data = self.current_floor
        self.floor_pub.publish(msg)

    def _publish_graph(self):
        msg = String()
        msg.data = json.dumps({
            'floors': [
                {
                    'index': i,
                    'height': h,
                    'map': self._map_path(i),
                    'mapped': os.path.exists(self._map_path(i)),
                }
                for i, h in enumerate(self.floor_heights)
            ],
            'transitions': self.transitions,
        })
        self.graph_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FloorManager()
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
