#!/usr/bin/env python3
"""Unit tests for the tri-star kinematics and ToF terrain geometry.

These run on a bare Python interpreter - no ROS, no Gazebo - so the maths that
decides whether the robot climbs or falls down the stairs can be checked in CI:

    python3 -m pytest src/r2d2_locomotion/test/test_kinematics.py
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from r2d2_locomotion.kinematics import (  # noqa: E402
    BLOCKED, CLIFF, FLAT, MODE_ROLLING, MODE_STOPPED, MODE_TUMBLING, RISER,
    UNKNOWN, body_twist, classify_delta, clamp, climb_envelope, flat_return,
    ground_spot_distance, height_delta, integrate_arc, joint_commands,
    rate_limit, side_speeds, wrap_angle)

# Platform constants, mirroring r2d2_description/config/robot_params.yaml.
R_SUB = 0.050
R_CLUSTER = 0.115
TRACK = 0.340
SLIP = 1.25
MAX_WHEEL = 12.0
MAX_CLUSTER = 3.5

TOF_H = 0.180
TOF_TILT = 0.5236          # 30 deg
STEP_MIN = 0.035
STEP_MAX = 0.150
CLIFF_DROP = 0.060


# ------------------------------------------------------------------ kinematics

def test_straight_line_has_no_side_difference():
    left, right = side_speeds(0.4, 0.0, TRACK, SLIP)
    assert left == pytest.approx(0.4)
    assert right == pytest.approx(0.4)


def test_spin_in_place_is_antisymmetric():
    left, right = side_speeds(0.0, 1.0, TRACK, SLIP)
    assert left == pytest.approx(-right)


def test_side_speeds_and_body_twist_round_trip():
    for v, w in [(0.0, 0.0), (0.35, 0.0), (0.0, 0.9), (0.22, -0.6), (-0.15, 1.3)]:
        left, right = side_speeds(v, w, TRACK, SLIP)
        v_back, w_back = body_twist(left, right, TRACK, SLIP)
        assert v_back == pytest.approx(v)
        assert w_back == pytest.approx(w)


def test_slip_factor_widens_the_side_difference():
    """A larger slip factor must drive the sides further apart for the same yaw."""
    ideal = side_speeds(0.0, 1.0, TRACK, 1.0)
    slipping = side_speeds(0.0, 1.0, TRACK, 1.5)
    assert abs(slipping[1] - slipping[0]) > abs(ideal[1] - ideal[0])


# ------------------------------------------------------------ joint expansion

def test_command_vector_is_sixteen_long():
    cmd = joint_commands(0.2, 0.2, MODE_ROLLING, R_SUB, R_CLUSTER,
                         MAX_WHEEL, MAX_CLUSTER)
    assert len(cmd) == 16


def test_rolling_drives_wheels_and_holds_carriers():
    cmd = joint_commands(0.2, 0.2, MODE_ROLLING, R_SUB, R_CLUSTER,
                         MAX_WHEEL, MAX_CLUSTER)
    carriers = cmd[0::4]
    wheels = [c for i, c in enumerate(cmd) if i % 4 != 0]
    assert carriers == [0.0, 0.0, 0.0, 0.0]
    assert all(w == pytest.approx(0.2 / R_SUB) for w in wheels)


def test_tumbling_drives_carriers_and_holds_wheels():
    cmd = joint_commands(0.12, 0.12, MODE_TUMBLING, R_SUB, R_CLUSTER,
                         MAX_WHEEL, MAX_CLUSTER)
    carriers = cmd[0::4]
    wheels = [c for i, c in enumerate(cmd) if i % 4 != 0]
    assert all(c == pytest.approx(0.12 / R_CLUSTER) for c in carriers)
    assert all(w == 0.0 for w in wheels)


def test_tumbling_is_slower_per_unit_speed_than_rolling():
    """Same ground speed costs fewer rad/s on the cluster than on a sub-wheel,
    because the cluster's effective radius is larger. This is the whole reason
    odometry cannot be shared between the two modes."""
    v = 0.15
    roll = joint_commands(v, v, MODE_ROLLING, R_SUB, R_CLUSTER, MAX_WHEEL, MAX_CLUSTER)
    tumble = joint_commands(v, v, MODE_TUMBLING, R_SUB, R_CLUSTER, MAX_WHEEL, MAX_CLUSTER)
    assert tumble[0] < roll[1]


def test_stopped_holds_everything():
    cmd = joint_commands(0.9, -0.9, MODE_STOPPED, R_SUB, R_CLUSTER,
                         MAX_WHEEL, MAX_CLUSTER)
    assert cmd == [0.0] * 16


def test_joint_rates_are_clamped():
    cmd = joint_commands(50.0, 50.0, MODE_ROLLING, R_SUB, R_CLUSTER,
                         MAX_WHEEL, MAX_CLUSTER)
    assert max(cmd) == pytest.approx(MAX_WHEEL)
    cmd = joint_commands(50.0, 50.0, MODE_TUMBLING, R_SUB, R_CLUSTER,
                         MAX_WHEEL, MAX_CLUSTER)
    assert max(cmd) == pytest.approx(MAX_CLUSTER)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        joint_commands(0.1, 0.1, 'hovering', R_SUB, R_CLUSTER, MAX_WHEEL, MAX_CLUSTER)


# --------------------------------------------------------------- odometry maths

def test_straight_integration():
    x, y, yaw = integrate_arc(0.0, 0.0, 0.0, 1.0, 0.0)
    assert (x, y, yaw) == pytest.approx((1.0, 0.0, 0.0))


def test_quarter_turn_arc_lands_on_the_circle():
    """Quarter circle of radius 1: from the origin heading +x, ending at
    (1, 1) heading +y. The midpoint approximation gets this measurably wrong."""
    d_yaw = math.pi / 2
    x, y, yaw = integrate_arc(0.0, 0.0, 0.0, d_yaw * 1.0, d_yaw)
    assert x == pytest.approx(1.0, abs=1e-9)
    assert y == pytest.approx(1.0, abs=1e-9)
    assert yaw == pytest.approx(math.pi / 2)


def test_full_circle_returns_to_start():
    x, y, yaw = 0.0, 0.0, 0.0
    steps = 360
    d_yaw = 2 * math.pi / steps
    for _ in range(steps):
        x, y, yaw = integrate_arc(x, y, yaw, d_yaw * 1.0, d_yaw)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)


def test_wrap_angle():
    # atan2 puts the half-turn on either boundary of [-pi, pi] depending on the
    # sign of the input, so compare magnitudes there.
    assert abs(wrap_angle(3 * math.pi)) == pytest.approx(math.pi)
    assert abs(wrap_angle(-3 * math.pi)) == pytest.approx(math.pi)
    assert wrap_angle(0.5) == pytest.approx(0.5)
    assert wrap_angle(2 * math.pi + 0.5) == pytest.approx(0.5)
    assert all(-math.pi - 1e-9 <= wrap_angle(a) <= math.pi + 1e-9
               for a in [i * 0.37 for i in range(-100, 100)])


def test_rate_limit_respects_the_cap_and_settles():
    v = 0.0
    for _ in range(100):
        v = rate_limit(1.0, v, 1.2, 0.02)
    assert v == pytest.approx(1.0)
    assert rate_limit(1.0, 0.0, 1.2, 0.02) == pytest.approx(0.024)


def test_clamp():
    assert clamp(5.0, 2.0) == 2.0
    assert clamp(-5.0, 2.0) == -2.0
    assert clamp(1.0, 2.0) == 1.0


# --------------------------------------------------------------- ToF geometry

def test_flat_return_matches_mounting_geometry():
    assert flat_return(TOF_H, TOF_TILT) == pytest.approx(0.360, abs=1e-3)
    assert ground_spot_distance(TOF_H, TOF_TILT) == pytest.approx(0.312, abs=1e-3)


def test_level_ground_reads_as_flat():
    r = flat_return(TOF_H, TOF_TILT)
    dz = height_delta(r, TOF_H, TOF_TILT, pitch=0.0, cliff_drop=CLIFF_DROP)
    assert dz == pytest.approx(0.0, abs=1e-9)
    assert classify_delta(dz, STEP_MIN, STEP_MAX, CLIFF_DROP) == FLAT


def test_a_150mm_riser_is_seen_as_climbable():
    """A step 0.15 m high shortens the return by 0.15 / sin(tilt)."""
    r = flat_return(TOF_H, TOF_TILT) - 0.150 / math.sin(TOF_TILT)
    dz = height_delta(r, TOF_H, TOF_TILT, pitch=0.0, cliff_drop=CLIFF_DROP)
    assert dz == pytest.approx(0.150, abs=1e-6)
    assert classify_delta(dz, STEP_MIN, STEP_MAX, CLIFF_DROP) == RISER


def test_a_door_sill_is_not_a_riser():
    for sill in (0.018, 0.022, 0.030):
        r = flat_return(TOF_H, TOF_TILT) - sill / math.sin(TOF_TILT)
        dz = height_delta(r, TOF_H, TOF_TILT, pitch=0.0, cliff_drop=CLIFF_DROP)
        assert classify_delta(dz, STEP_MIN, STEP_MAX, CLIFF_DROP) == FLAT, sill


def test_a_step_taller_than_the_cluster_can_mount_is_blocked():
    r = flat_return(TOF_H, TOF_TILT) - 0.30 / math.sin(TOF_TILT)
    dz = height_delta(r, TOF_H, TOF_TILT, pitch=0.0, cliff_drop=CLIFF_DROP)
    assert classify_delta(dz, STEP_MIN, STEP_MAX, CLIFF_DROP) == BLOCKED


def test_a_drop_is_a_cliff():
    r = flat_return(TOF_H, TOF_TILT) + 0.20 / math.sin(TOF_TILT)
    dz = height_delta(r, TOF_H, TOF_TILT, pitch=0.0, cliff_drop=CLIFF_DROP)
    assert classify_delta(dz, STEP_MIN, STEP_MAX, CLIFF_DROP) == CLIFF


def test_no_return_at_all_is_a_cliff():
    dz = height_delta(math.inf, TOF_H, TOF_TILT, pitch=0.0, cliff_drop=CLIFF_DROP)
    assert classify_delta(dz, STEP_MIN, STEP_MAX, CLIFF_DROP) == CLIFF


def test_missing_beam_is_unknown_not_flat():
    assert classify_delta(None, STEP_MIN, STEP_MAX, CLIFF_DROP) == UNKNOWN


def test_ramp_does_not_produce_phantom_steps():
    """The regression this correction exists for.

    On the 12.1 degree ramp in the house world the robot pitches nose-up, which
    lengthens every ToF return. Without pitch compensation that reads as a
    continuous cliff and the robot refuses to use the ramp at all.
    """
    pitch = -math.radians(12.1)                  # nose-up on the climb
    eff_tilt = TOF_TILT + pitch
    # What the beam actually measures on a surface that is itself inclined by
    # the same angle the body is inclined by: the floor stays underfoot.
    r = TOF_H / math.sin(eff_tilt)

    naive = TOF_H - r * math.sin(TOF_TILT)       # no pitch correction
    assert classify_delta(naive, STEP_MIN, STEP_MAX, CLIFF_DROP) == CLIFF

    corrected = height_delta(r, TOF_H, TOF_TILT, pitch=pitch, cliff_drop=CLIFF_DROP)
    assert classify_delta(corrected, STEP_MIN, STEP_MAX, CLIFF_DROP) == FLAT


def test_rear_beams_take_the_opposite_pitch_sign():
    pitch = 0.15
    r = flat_return(TOF_H, TOF_TILT)
    front = height_delta(r, TOF_H, TOF_TILT, pitch, CLIFF_DROP, rear=False)
    rear = height_delta(r, TOF_H, TOF_TILT, pitch, CLIFF_DROP, rear=True)
    assert front != pytest.approx(rear)
    assert (front - 0.0) * (rear - 0.0) < 0.0    # deltas point opposite ways


def test_extreme_nose_up_reports_unknown_rather_than_garbage():
    assert height_delta(0.4, TOF_H, TOF_TILT, pitch=-0.6,
                        cliff_drop=CLIFF_DROP) is None


# ------------------------------------------------------------- climb envelope

def test_designed_platform_can_mount_the_designed_stairs():
    env = climb_envelope(R_CLUSTER, R_SUB, riser=0.150, tread=0.280)
    assert env['can_mount']
    assert env['tread_ok']
    assert env['margin'] == pytest.approx(0.10, abs=1e-9)


def test_a_tall_riser_is_outside_the_envelope():
    env = climb_envelope(R_CLUSTER, R_SUB, riser=0.200, tread=0.280)
    assert not env['can_mount']


def test_a_shallow_tread_bridges_two_risers():
    env = climb_envelope(R_CLUSTER, R_SUB, riser=0.150, tread=0.090)
    assert not env['tread_ok']
