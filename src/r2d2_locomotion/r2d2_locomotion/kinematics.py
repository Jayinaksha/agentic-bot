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

# Per-cluster phase offsets baked into the URDF, mirrored here so the carrier
# hold can compute the right rest angle. Keep in step with r2d2_tristar.urdf.xacro.
URDF_PHASES = {
    'front_left': 0.0,
    'front_right': math.pi / 3.0,
    'rear_left': math.pi / 3.0,
    'rear_right': 0.0,
}
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


# A three-spoke carrier has two rest positions: balanced on one sub-wheel, or
# straddling two. Straddling puts the axle at r_cluster*cos(60 deg) + r_sub and
# the other at r_cluster + r_sub - for this platform, 0.108 m against 0.165 m.
# The carrier must therefore be held at a defined PHASE in rolling mode, not
# merely commanded to zero velocity: a velocity hold parks it wherever it
# happened to stop, and the resulting 57 mm of ride-height variation is larger
# than the 35 mm step the ToF terrain monitor is trying to detect. The robot
# would see phantom stairs on a flat floor.
#
# Straddling is the phase we want: lower centre of mass, and stable rather than
# balanced on a knife edge.
CARRIER_PERIOD = 2.0 * math.pi / 3.0     # three-fold symmetry
STRADDLE_PHASE = -math.pi / 6.0          # sub-wheels at -30 and 210 deg


def carrier_rest_phase(urdf_phase: float) -> float:
    """Carrier joint angle that leaves two sub-wheels straddling the ground.

    The URDF bakes a per-cluster phase offset into the sub-wheel origins (left
    and right sides run 60 deg apart so a synchronous tumble never lifts both
    sides at once), so the joint angle needed to reach the straddle position
    differs per cluster.
    """
    return wrap_to_period(STRADDLE_PHASE - urdf_phase, CARRIER_PERIOD)


def wrap_to_period(angle: float, period: float) -> float:
    """Wrap into (-period/2, period/2]."""
    wrapped = math.fmod(angle, period)
    if wrapped > period / 2.0:
        wrapped -= period
    elif wrapped <= -period / 2.0:
        wrapped += period
    return wrapped


def carrier_hold_velocity(position: float, rest_phase: float,
                          gain: float = 6.0, max_rate: float = 2.0) -> float:
    """Velocity that servos a carrier onto its nearest rest phase.

    Nearest matters: with three-fold symmetry every phase has an equivalent
    120 deg away, so the error is wrapped to that period and the carrier never
    turns more than 60 deg to settle. Without the wrap it could take the long
    way round and lift the chassis on the way.
    """
    error = wrap_to_period(rest_phase - position, CARRIER_PERIOD)
    return clamp(gain * error, max_rate)


def joint_commands(v_left: float, v_right: float, mode: str,
                   r_sub: float, r_cluster: float,
                   max_wheel: float, max_cluster: float,
                   carrier_positions: Optional[Dict[str, float]] = None,
                   urdf_phases: Optional[Dict[str, float]] = None,
                   hold_gain: float = 6.0) -> List[float]:
    """Expand two side speeds into the 16-element joint velocity vector.

    ROLLING  drives the sub-wheels; carriers are servoed onto their straddle
             phase when their positions are known, and held at zero otherwise.
    TUMBLING drives the carriers, holds the sub-wheels.
    STOPPED  holds the wheels, but still parks the carriers, so the robot comes
             to rest at a known ride height.
    """
    if mode not in MODES:
        raise ValueError(f'unknown transmission mode: {mode!r}')

    out: List[float] = []
    for cluster in CLUSTERS:
        v_side = v_left if cluster in LEFT_CLUSTERS else v_right

        if mode == MODE_TUMBLING:
            carrier, wheel = clamp(v_side / r_cluster, max_cluster), 0.0
        else:
            wheel = clamp(v_side / r_sub, max_wheel) if mode == MODE_ROLLING else 0.0
            carrier = 0.0
            if carrier_positions and cluster in carrier_positions:
                phase = (urdf_phases or {}).get(cluster, 0.0)
                carrier = carrier_hold_velocity(
                    carrier_positions[cluster], carrier_rest_phase(phase),
                    gain=hold_gain, max_rate=max_cluster)

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


# ---------------------------------------------------------- riser contact
#
# Detecting the moment the platform is against a riser and must switch to
# tumbling.
#
# The obvious signal - body pitch - does not work. In rolling mode the carriers
# are phase-locked (see carrier_hold_velocity), so a sub-wheel meeting a riser
# face simply stops: the chassis does not tip, it stalls. Waiting for pitch means
# waiting forever, and the approach times out having never started the climb.
#
# The signal that does work is the stall itself. Wheel odometry is valid in
# rolling mode, so comparing commanded travel against achieved travel says
# plainly whether the robot is still moving. Combined with the ToF beams
# agreeing there is a mountable riser ahead, that is the mount trigger.

class RiserContact:
    """Confirms the platform has driven up against a riser.

    Contact is declared when, for `confirm_s` of continuous evidence:
      * both front ToF beams see a climbable riser, and
      * achieved travel falls below `stall_ratio` of commanded travel.

    Requiring both matters in each direction. Stall alone fires on a chair leg,
    a rug, or a wheel caught on a threshold. Riser-ahead alone fires while the
    robot is still a lookahead-distance away, and tumbling in free space walks
    the platform forward on its cluster corners instead of driving.
    """

    __slots__ = ('stall_ratio', 'confirm_s', 'min_commanded', '_evidence_s',
                 '_commanded', '_travelled', '_window_s')

    def __init__(self, stall_ratio: float = 0.35, confirm_s: float = 0.6,
                 min_commanded: float = 0.02):
        self.stall_ratio = stall_ratio
        self.confirm_s = confirm_s
        # Below this much commanded travel in a tick there is nothing to
        # compare against and a stall reading would be meaningless.
        self.min_commanded = min_commanded
        self._evidence_s = 0.0
        self._commanded = 0.0
        self._travelled = 0.0
        self._window_s = 0.0

    def reset(self) -> None:
        self._evidence_s = 0.0
        self._commanded = 0.0
        self._travelled = 0.0
        self._window_s = 0.0

    def update(self, commanded_v: float, travelled_m: float, dt: float,
               riser_ahead: bool) -> bool:
        """Feed one control cycle. Returns True once contact is confirmed."""
        if dt <= 0.0:
            return self._evidence_s >= self.confirm_s

        self._commanded += abs(commanded_v) * dt
        self._travelled += abs(travelled_m)
        self._window_s += dt

        if not riser_ahead:
            # Lost sight of the step: whatever the wheels are doing, this is not
            # the bottom of a flight.
            self.reset()
            return False

        if self._commanded < self.min_commanded:
            # Not enough commanded travel yet to judge. Keep accumulating; the
            # window's own elapsed time is what will be credited, so a slow
            # creep is not penalised for taking several cycles to gather
            # a measurable distance.
            return False

        ratio = self._travelled / self._commanded
        if ratio < self.stall_ratio:
            # Credit the whole window, not one tick. At 0.10 m/s and 20 Hz it
            # takes four cycles to accumulate min_commanded, so crediting a
            # single dt would make the detector run four times slow and the
            # approach would time out before confirming a stall it had already
            # seen.
            self._evidence_s += self._window_s
        else:
            # Moving freely again: start the evidence over rather than letting
            # a slow patch of carpet accumulate towards a false trigger.
            self._evidence_s = 0.0
        self._commanded = 0.0
        self._travelled = 0.0
        self._window_s = 0.0

        return self._evidence_s >= self.confirm_s

    @property
    def evidence_s(self) -> float:
        return self._evidence_s
