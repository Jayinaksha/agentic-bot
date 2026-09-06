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
    BLOCKED, CLIFF, CLUSTERS, FLAT, MODE_ROLLING, MODE_STOPPED, MODE_TUMBLING,
    RISER, UNKNOWN, URDF_PHASES, body_twist, carrier_hold_velocity,
    carrier_rest_phase, classify_delta, clamp, climb_envelope, flat_return,
    ground_spot_distance, height_delta, integrate_arc, joint_commands,
    DIRECTION_DOWN, DIRECTION_UP, EdgeApproach, RiserContact,
    descent_creep_distance, descent_is_geometrically_safe, rate_limit,
    side_speeds, wrap_angle, wrap_to_period)

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


# --------------------------------------------------- carrier phase hold
#
# A three-spoke carrier has two rest positions, 57 mm apart in ride height. A
# zero-velocity hold parks it at whichever it happened to reach, and 57 mm is
# larger than the 35 mm step the ToF monitor is trying to detect - so the robot
# would read phantom stairs on a flat floor. These tests cover the servo that
# stops that happening. See scripts/analyse_climb.py, check "ride height".

STRADDLE = -math.pi / 6
PERIOD = 2 * math.pi / 3


def test_rest_phase_puts_two_wheels_on_the_ground():
    """Straddle phase means sub-wheels at -30 and 210 deg, either side of the
    -90 deg contact point, with the third at the top."""
    rest = carrier_rest_phase(0.0)
    angles = [wrap_angle(rest + k * 2 * math.pi / 3) for k in range(3)]
    below = [a for a in angles if math.sin(a) < -0.1]
    assert len(below) == 2
    assert math.sin(below[0]) == pytest.approx(math.sin(below[1]), abs=1e-9)


def test_rest_phase_compensates_for_the_urdf_offset():
    """Left and right clusters are 60 deg apart in the URDF, so they need
    different joint angles to reach the same physical straddle."""
    assert carrier_rest_phase(0.0) != pytest.approx(carrier_rest_phase(math.pi / 3))
    for phase in URDF_PHASES.values():
        rest = carrier_rest_phase(phase)
        physical = wrap_to_period(rest + phase, PERIOD)
        assert physical == pytest.approx(wrap_to_period(STRADDLE, PERIOD), abs=1e-9)


def test_every_cluster_has_a_rest_phase():
    assert set(URDF_PHASES) == set(CLUSTERS)


def test_hold_is_silent_at_the_rest_phase():
    rest = carrier_rest_phase(0.0)
    assert carrier_hold_velocity(rest, rest) == pytest.approx(0.0)


def test_hold_is_silent_at_an_equivalent_phase():
    """Three-fold symmetry: 120 deg away is the same physical position."""
    rest = carrier_rest_phase(0.0)
    assert carrier_hold_velocity(rest + PERIOD, rest) == pytest.approx(0.0, abs=1e-9)
    assert carrier_hold_velocity(rest - PERIOD, rest) == pytest.approx(0.0, abs=1e-9)


def test_hold_never_takes_the_long_way_round():
    """The correction must stay within +/-60 deg, or the carrier lifts the
    chassis on its way to a position it was already at."""
    rest = carrier_rest_phase(0.0)
    for offset in [i * 0.05 for i in range(-200, 200)]:
        velocity = carrier_hold_velocity(rest + offset, rest, gain=1.0,
                                         max_rate=99.0)
        assert abs(velocity) <= PERIOD / 2 + 1e-9, offset


def test_hold_drives_towards_the_rest_phase():
    rest = carrier_rest_phase(0.0)
    assert carrier_hold_velocity(rest - 0.2, rest) > 0
    assert carrier_hold_velocity(rest + 0.2, rest) < 0


def test_hold_respects_the_rate_limit():
    rest = carrier_rest_phase(0.0)
    assert abs(carrier_hold_velocity(rest + 1.0, rest, gain=50.0,
                                     max_rate=3.5)) == pytest.approx(3.5)


def test_hold_converges():
    """Iterating the servo must settle, not oscillate."""
    rest = carrier_rest_phase(0.0)
    position = rest + 0.9
    for _ in range(400):
        position += carrier_hold_velocity(position, rest, gain=6.0,
                                          max_rate=2.0) * 0.02
    assert wrap_to_period(position - rest, PERIOD) == pytest.approx(0.0, abs=1e-3)


def test_wrap_to_period():
    assert wrap_to_period(0.1, PERIOD) == pytest.approx(0.1)
    assert wrap_to_period(PERIOD + 0.1, PERIOD) == pytest.approx(0.1)
    assert wrap_to_period(-PERIOD - 0.1, PERIOD) == pytest.approx(-0.1)
    for a in [i * 0.13 for i in range(-100, 100)]:
        assert -PERIOD / 2 - 1e-9 <= wrap_to_period(a, PERIOD) <= PERIOD / 2 + 1e-9


# ---------------------------------------------- carrier hold in joint commands

def _positions(offset=0.0):
    return {c: carrier_rest_phase(URDF_PHASES[c]) + offset for c in CLUSTERS}


def test_rolling_holds_carriers_still_when_already_parked():
    cmd = joint_commands(0.2, 0.2, MODE_ROLLING, R_SUB, R_CLUSTER,
                         MAX_WHEEL, MAX_CLUSTER,
                         carrier_positions=_positions(),
                         urdf_phases=URDF_PHASES)
    assert cmd[0::4] == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_rolling_corrects_a_drifted_carrier():
    cmd = joint_commands(0.2, 0.2, MODE_ROLLING, R_SUB, R_CLUSTER,
                         MAX_WHEEL, MAX_CLUSTER,
                         carrier_positions=_positions(offset=-0.3),
                         urdf_phases=URDF_PHASES)
    assert all(c > 0 for c in cmd[0::4])
    # Sub-wheels keep driving while the carrier is corrected.
    assert cmd[1] == pytest.approx(0.2 / R_SUB)


def test_stopped_still_parks_the_carriers():
    """Coming to rest at a known ride height matters as much as driving at one."""
    cmd = joint_commands(0.0, 0.0, MODE_STOPPED, R_SUB, R_CLUSTER,
                         MAX_WHEEL, MAX_CLUSTER,
                         carrier_positions=_positions(offset=0.4),
                         urdf_phases=URDF_PHASES)
    assert all(c < 0 for c in cmd[0::4])
    assert [c for i, c in enumerate(cmd) if i % 4 != 0] == [0.0] * 12


def test_tumbling_ignores_the_hold_entirely():
    """During a climb the carrier is the drive; parking it would stop the robot
    mid-riser."""
    cmd = joint_commands(0.12, 0.12, MODE_TUMBLING, R_SUB, R_CLUSTER,
                         MAX_WHEEL, MAX_CLUSTER,
                         carrier_positions=_positions(offset=0.5),
                         urdf_phases=URDF_PHASES)
    assert all(c == pytest.approx(0.12 / R_CLUSTER) for c in cmd[0::4])


def test_without_positions_the_behaviour_is_the_old_zero_hold():
    """Before /joint_states arrives there is nothing to servo against, so the
    controller must degrade to a plain hold rather than command nonsense."""
    cmd = joint_commands(0.2, 0.2, MODE_ROLLING, R_SUB, R_CLUSTER,
                         MAX_WHEEL, MAX_CLUSTER)
    assert cmd[0::4] == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_hold_velocity_is_capped_by_max_cluster_rate():
    cmd = joint_commands(0.0, 0.0, MODE_ROLLING, R_SUB, R_CLUSTER,
                         MAX_WHEEL, MAX_CLUSTER,
                         carrier_positions=_positions(offset=-1.0),
                         urdf_phases=URDF_PHASES,
                         hold_gain=100.0)
    assert max(cmd[0::4]) == pytest.approx(MAX_CLUSTER)


# ------------------------------------------------------- riser contact
#
# The trigger that switches the platform from rolling to tumbling. Getting this
# wrong is fatal in one direction and dangerous in the other: too strict and the
# robot never starts a climb, too loose and it starts tumbling in open floor,
# walking on its cluster corners instead of driving.

CONTACT_DT = 0.05          # 20 Hz, matching climb_fsm's rate


def _drive(detector, seconds, commanded, travelled_ratio, riser=True):
    """Feed the detector `seconds` of driving at a given achieved/commanded ratio."""
    fired = False
    for _ in range(int(seconds / CONTACT_DT)):
        fired = detector.update(commanded, commanded * travelled_ratio * CONTACT_DT,
                                CONTACT_DT, riser) or fired
    return fired


def test_free_driving_never_triggers():
    """Rolling along normally, riser in sight, must not fire."""
    detector = RiserContact()
    assert not _drive(detector, seconds=10.0, commanded=0.10,
                      travelled_ratio=1.0)


def test_a_stall_against_a_riser_triggers():
    detector = RiserContact(confirm_s=0.6)
    assert _drive(detector, seconds=3.0, commanded=0.10, travelled_ratio=0.0)


def test_a_stall_takes_the_confirm_time_to_trigger():
    """It must not fire on a single slow cycle."""
    detector = RiserContact(confirm_s=0.6)
    assert not _drive(detector, seconds=0.3, commanded=0.10, travelled_ratio=0.0)
    assert _drive(detector, seconds=0.5, commanded=0.10, travelled_ratio=0.0)


def test_a_stall_with_no_riser_in_sight_never_triggers():
    """A wheel caught on a rug is not the bottom of a staircase."""
    detector = RiserContact()
    assert not _drive(detector, seconds=5.0, commanded=0.10,
                      travelled_ratio=0.0, riser=False)


def test_losing_sight_of_the_riser_clears_the_evidence():
    detector = RiserContact(confirm_s=0.6)
    _drive(detector, seconds=0.4, commanded=0.10, travelled_ratio=0.0)
    assert detector.evidence_s > 0.0
    detector.update(0.10, 0.0, CONTACT_DT, riser_ahead=False)
    assert detector.evidence_s == 0.0


def test_moving_again_restarts_the_evidence():
    """A slow patch of carpet must not accumulate towards a false trigger."""
    detector = RiserContact(confirm_s=0.6)
    for _ in range(6):
        _drive(detector, seconds=0.25, commanded=0.10, travelled_ratio=0.0)
        _drive(detector, seconds=0.25, commanded=0.10, travelled_ratio=1.0)
    assert detector.evidence_s == 0.0


def test_partial_slip_is_not_a_stall():
    """Climbing a threshold slows the robot without stopping it."""
    detector = RiserContact(stall_ratio=0.35)
    assert not _drive(detector, seconds=5.0, commanded=0.10,
                      travelled_ratio=0.6)


def test_severe_slip_is_a_stall():
    detector = RiserContact(stall_ratio=0.35)
    assert _drive(detector, seconds=3.0, commanded=0.10, travelled_ratio=0.15)


def test_a_stationary_robot_does_not_trigger():
    """No command means no evidence either way; otherwise a parked robot beside
    a staircase would eventually decide it was against it."""
    detector = RiserContact()
    assert not _drive(detector, seconds=10.0, commanded=0.0, travelled_ratio=0.0)
    assert detector.evidence_s == 0.0


def test_reset_clears_everything():
    detector = RiserContact()
    _drive(detector, seconds=0.4, commanded=0.10, travelled_ratio=0.0)
    detector.reset()
    assert detector.evidence_s == 0.0


def test_zero_dt_is_harmless():
    detector = RiserContact()
    assert detector.update(0.1, 0.0, 0.0, True) is False


def test_the_pitch_only_trigger_would_never_have_fired():
    """Regression for the bug this detector replaced.

    With the carriers phase-locked in rolling mode, a sub-wheel meeting a riser
    face stalls the robot without tipping it. A pitch-threshold trigger sees
    nothing, the approach times out, and the platform never climbs a single
    step. This asserts the stall path fires well inside that timeout.
    """
    detector = RiserContact(confirm_s=0.6)
    elapsed = 0.0
    approach_timeout = 15.0
    while elapsed < approach_timeout:
        pitch = 0.0                      # locked carriers: the chassis cannot tip
        if detector.update(0.10, 0.0, CONTACT_DT, True) or pitch > 0.08:
            break
        elapsed += CONTACT_DT
    assert elapsed < 1.0, 'contact must be confirmed long before the timeout'


# ---------------------------------------------------------- stair descent
#
# Descent is the dangerous direction. Going up, the riser stops the robot and
# the stall says "you are here". Going down there is no such event - a robot
# that keeps rolling drives off the top step - so the last stretch is
# dead-reckoned from geometry, and these tests cover that geometry.

def test_creep_distance_matches_the_sensor_geometry():
    """Beam lands 0.47 m ahead of base_link, front contact is 0.13 m ahead, so
    the gap is 0.34 m; stop 0.05 m short of it."""
    creep = descent_creep_distance(tof_forward_offset=0.180, spot_ahead=0.290,
                                   wheelbase=0.260, margin=0.05)
    assert creep == pytest.approx(0.290)
    assert descent_is_geometrically_safe(creep)


def test_a_bigger_margin_commits_earlier():
    near = descent_creep_distance(0.180, 0.290, 0.260, margin=0.05)
    early = descent_creep_distance(0.180, 0.290, 0.260, margin=0.15)
    assert early < near


def test_a_robot_that_cannot_see_the_edge_in_time_is_refused():
    """A long wheelbase with a short lookahead puts the beam behind the front
    wheels: there is no warning to act on, so descent must be refused."""
    creep = descent_creep_distance(tof_forward_offset=0.05, spot_ahead=0.05,
                                   wheelbase=0.60)
    assert creep < 0
    assert not descent_is_geometrically_safe(creep)


def test_edge_approach_waits_for_a_consistent_cliff():
    """One frame of dark floor reads the same as a void."""
    edge = EdgeApproach(creep_distance=0.29, confirm_cycles=4)
    for _ in range(3):
        assert not edge.update(0.01, cliff_ahead=True)
    assert not edge.armed


def test_edge_approach_measures_only_after_arming():
    """Travel before the cliff is confirmed must not count towards the creep,
    or the robot commits early and tumbles into thin air."""
    edge = EdgeApproach(creep_distance=0.29, confirm_cycles=4)
    for _ in range(4):
        edge.update(0.50, cliff_ahead=True)
    assert edge.armed
    assert edge.travelled == pytest.approx(0.0)


def test_edge_approach_fires_after_the_creep_distance():
    edge = EdgeApproach(creep_distance=0.29, confirm_cycles=4)
    for _ in range(4):
        assert not edge.update(0.0, cliff_ahead=True)
    fired = False
    travelled = 0.0
    for _ in range(200):
        travelled += 0.005
        if edge.update(0.005, cliff_ahead=True):
            fired = True
            break
    assert fired
    assert travelled == pytest.approx(0.29, abs=0.01)


def test_edge_approach_does_not_fire_early():
    edge = EdgeApproach(creep_distance=0.29, confirm_cycles=4)
    for _ in range(4):
        edge.update(0.0, cliff_ahead=True)
    for _ in range(40):                       # 40 * 0.005 = 0.20 m, short of 0.29
        assert not edge.update(0.005, cliff_ahead=True)


def test_losing_the_cliff_discards_the_measurement():
    """A drop that stops being visible was a misreading or the robot turned
    away. Committing on a stale measurement is how a robot falls downstairs."""
    edge = EdgeApproach(creep_distance=0.29, confirm_cycles=4)
    for _ in range(4):
        edge.update(0.0, cliff_ahead=True)
    for _ in range(40):
        edge.update(0.005, cliff_ahead=True)
    assert edge.travelled > 0.0

    edge.update(0.005, cliff_ahead=False)
    assert not edge.armed
    assert edge.travelled == pytest.approx(0.0)


def test_the_measurement_restarts_cleanly_after_a_dropout():
    edge = EdgeApproach(creep_distance=0.29, confirm_cycles=4)
    for _ in range(4):
        edge.update(0.0, cliff_ahead=True)
    for _ in range(30):
        edge.update(0.005, cliff_ahead=True)
    edge.update(0.005, cliff_ahead=False)

    for _ in range(4):
        assert not edge.update(0.0, cliff_ahead=True)
    fired = sum(1 for _ in range(200) if edge.update(0.005, cliff_ahead=True))
    assert fired > 0


def test_no_cliff_means_no_descent_ever():
    edge = EdgeApproach(creep_distance=0.29, confirm_cycles=4)
    assert not any(edge.update(0.05, cliff_ahead=False) for _ in range(200))


def test_reset_clears_the_edge_approach():
    edge = EdgeApproach(creep_distance=0.29, confirm_cycles=4)
    for _ in range(10):
        edge.update(0.01, cliff_ahead=True)
    edge.reset()
    assert not edge.armed
    assert edge.travelled == pytest.approx(0.0)


def test_the_two_directions_are_distinct_constants():
    assert DIRECTION_UP != DIRECTION_DOWN
