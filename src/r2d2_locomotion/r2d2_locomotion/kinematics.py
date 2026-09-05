#!/usr/bin/env python3
"""
Pure kinematics and terrain geometry for the tri-star platform.

Deliberately free of any ROS import so the maths can be unit-tested on a plain
Python interpreter (see test/test_kinematics.py). The ROS nodes are thin
wrappers around these functions.
"""

import math
from typing import Dict, List, Optional, Tuple

# Command vector layout, matching r2d2_locomotion/config/controllers.yaml.
CLUSTERS: Tuple[str, ...] = ('front_left', 'front_right', 'rear_left', 'rear_right')
LEFT_CLUSTERS = frozenset({'front_left', 'rear_left'})
JOINTS_PER_CLUSTER = 4  # carrier, wheel0, wheel1, wheel2

MODE_ROLLING = 'rolling'
MODE_TUMBLING = 'tumbling'
MODE_STOPPED = 'stopped'
MODES = frozenset({MODE_ROLLING, MODE_TUMBLING, MODE_STOPPED})

# Terrain classes.
FLAT = 'flat'
RISER = 'climbable_riser'
BLOCKED = 'blocked'
CLIFF = 'cliff'
UNKNOWN = 'unknown'


def clamp(value: float, limit: float) -> float:
    """Symmetric clamp to [-limit, +limit]."""
    return max(-limit, min(limit, value))


def wrap_angle(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def side_speeds(v: float, w: float, track: float, yaw_slip: float) -> Tuple[float, float]:
    """Contact-patch speeds of the left and right sides, m/s.

    `yaw_slip` > 1 models the lateral scrub of a four-contact skid-steer: to
    achieve a commanded yaw rate the sides must be driven further apart than the
    ideal differential model predicts.
    """
    half = yaw_slip * w * track / 2.0
    return v - half, v + half


def body_twist(v_left: float, v_right: float, track: float,
               yaw_slip: float) -> Tuple[float, float]:
    """Inverse of side_speeds: recover (v, w) from measured side speeds."""
    v = (v_left + v_right) / 2.0
    w = (v_right - v_left) / (yaw_slip * track)
    return v, w


def joint_commands(v_left: float, v_right: float, mode: str,
                   r_sub: float, r_cluster: float,
                   max_wheel: float, max_cluster: float) -> List[float]:
    """Expand two side speeds into the 16-element joint velocity vector.

    ROLLING  drives the sub-wheels, holds the carriers.
    TUMBLING drives the carriers, holds the sub-wheels.
    STOPPED  holds everything.
    """
    if mode not in MODES:
        raise ValueError(f'unknown transmission mode: {mode!r}')

    out: List[float] = []
    for cluster in CLUSTERS:
        v_side = v_left if cluster in LEFT_CLUSTERS else v_right
        if mode == MODE_TUMBLING:
            carrier, wheel = clamp(v_side / r_cluster, max_cluster), 0.0
        elif mode == MODE_ROLLING:
            carrier, wheel = 0.0, clamp(v_side / r_sub, max_wheel)
        else:
            carrier, wheel = 0.0, 0.0
        out.extend([carrier, wheel, wheel, wheel])
    return out


def integrate_arc(x: float, y: float, yaw: float,
                  d_center: float, d_yaw: float) -> Tuple[float, float, float]:
    """Exact constant-curvature integration of one odometry increment.

    The midpoint approximation drifts noticeably during the tight, repeated
    turns a house layout forces, so the arc form is used instead.
    """
    if abs(d_yaw) < 1e-9:
        return x + d_center * math.cos(yaw), y + d_center * math.sin(yaw), yaw
    radius = d_center / d_yaw
    nx = x + radius * (math.sin(yaw + d_yaw) - math.sin(yaw))
    ny = y - radius * (math.cos(yaw + d_yaw) - math.cos(yaw))
    return nx, ny, wrap_angle(yaw + d_yaw)


def rate_limit(target: float, current: float, max_rate: float, dt: float) -> float:
    """First-order slew limit."""
    delta = target - current
    cap = max_rate * dt
    if delta > cap:
        return current + cap
    if delta < -cap:
        return current - cap
    return target


# --------------------------------------------------------------- ToF geometry

def flat_return(mount_height: float, tilt: float) -> float:
    """Range a downward ToF reports on level ground."""
    return mount_height / math.sin(tilt)


def ground_spot_distance(mount_height: float, tilt: float) -> float:
    """How far ahead of the sensor the beam meets level ground."""
    return mount_height / math.tan(tilt)


def height_delta(range_m: Optional[float], mount_height: float, tilt: float,
                 pitch: float, cliff_drop: float,
                 rear: bool = False) -> Optional[float]:
    """Floor-height change under one ToF spot, metres, positive = step up.

    `pitch` is the IMU body pitch (REP-103: positive is nose-down). Without this
    correction a 12 degree ramp reads as a continuous wall of phantom steps.
    Rear-facing beams see pitch with the opposite sign.
    """
    if range_m is None:
        return None
    eff_tilt = tilt + (-pitch if rear else pitch)
    if eff_tilt <= 0.05:
        # Beam no longer points at the floor at all.
        return None
    if math.isinf(range_m):
        # No floor found inside the window: for a downward beam that is itself
        # evidence of a drop, reported as unambiguously past the cliff limit.
        return -2.0 * cliff_drop
    return mount_height - range_m * math.sin(eff_tilt)


def classify_delta(dz: Optional[float], step_min: float, step_max: float,
                   cliff_drop: float) -> str:
    """Map a height delta onto a terrain class."""
    if dz is None:
        return UNKNOWN
    if dz < -cliff_drop:
        return CLIFF
    if dz > step_max:
        return BLOCKED
    if dz >= step_min:
        return RISER
    return FLAT


def climb_envelope(cluster_circumradius: float, sub_wheel_radius: float,
                   riser: float, tread: float) -> Dict[str, float]:
    """Whether this cluster geometry can mount this staircase.

    Two constraints from the tri-wheel stair-climbing literature:

      reach = cluster_circumradius + sub_wheel_radius  must exceed the riser,
      tread must exceed the sub-wheel diameter, else a cluster bridges two
      risers instead of landing on the tread.
    """
    reach = cluster_circumradius + sub_wheel_radius
    return {
        'reach': reach,
        'riser': riser,
        'margin': (reach - riser) / riser if riser > 0 else float('inf'),
        'can_mount': reach > riser,
        'tread_ok': tread > 2.0 * sub_wheel_radius,
    }
