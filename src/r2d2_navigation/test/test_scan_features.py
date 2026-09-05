#!/usr/bin/env python3
"""Unit tests for the docking geometry.

    python3 -m pytest src/r2d2_navigation/test/test_scan_features.py

These check the maths that decides where the robot stops. Getting the sign of
the yaw error wrong here means the servo drives away from the target, which is
exactly the kind of bug that is invisible until a real robot is in front of a
real fridge.
"""

import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from r2d2_navigation.scan_features import (  # noqa: E402
    Gap, Line, find_gap, fit_line, gap_pose_error, line_pose_error,
    polar_to_xy, synth_wall)

STANDOFF = 0.35


# ---------------------------------------------------------------------- basics

def test_polar_to_xy():
    assert polar_to_xy(1.0, 0.0) == pytest.approx((1.0, 0.0))
    assert polar_to_xy(2.0, math.pi / 2) == pytest.approx((0.0, 2.0), abs=1e-12)


def test_line_normal_is_unit_and_faces_away_from_robot():
    line = Line(3.0, 4.0, 5.0, inliers=20)
    assert math.hypot(line.nx, line.ny) == pytest.approx(1.0)
    assert line.d > 0.0


def test_line_sign_is_normalised():
    """A normal pointing back at the robot is flipped, so d stays a distance."""
    line = Line(-1.0, 0.0, -2.0, inliers=20)
    assert line.d == pytest.approx(2.0)
    assert line.nx == pytest.approx(1.0)


# -------------------------------------------------------------- line fitting

def test_fits_a_wall_dead_ahead():
    points = synth_wall(distance=1.0, bearing=0.0)
    line = fit_line(points)
    assert line is not None
    assert line.d == pytest.approx(1.0, abs=1e-6)
    assert line.normal_bearing == pytest.approx(0.0, abs=1e-6)


def test_fits_a_wall_parallel_to_the_y_axis():
    """The case ordinary least squares cannot handle.

    A robot squared up to a fridge door sees a surface whose points all share an
    x coordinate. OLS on y = mx + c has infinite slope here; total least squares
    does not care.
    """
    points = [(1.0, -0.6 + 0.03 * i) for i in range(41)]
    line = fit_line(points)
    assert line is not None
    assert line.d == pytest.approx(1.0, abs=1e-9)
    assert line.normal_bearing == pytest.approx(0.0, abs=1e-9)


def test_fits_a_wall_at_an_angle():
    for bearing in (-0.6, -0.2, 0.2, 0.6):
        points = synth_wall(distance=1.2, bearing=bearing)
        line = fit_line(points)
        assert line is not None, bearing
        assert line.d == pytest.approx(1.2, abs=1e-6)
        assert line.normal_bearing == pytest.approx(bearing, abs=1e-6)


def test_survives_realistic_scan_noise():
    rng = random.Random(20260905)
    points = synth_wall(distance=1.0, bearing=0.15, count=60,
                        noise=0.012, rng=rng)
    line = fit_line(points, inlier_dist=0.04)
    assert line is not None
    assert line.d == pytest.approx(1.0, abs=0.01)
    assert line.normal_bearing == pytest.approx(0.15, abs=0.03)


def test_rejects_outliers_from_clutter():
    """A door handle and a stray return must not drag the fit."""
    points = synth_wall(distance=1.0, bearing=0.0, count=50)
    clean = fit_line(points)
    points += [(0.4, 0.2), (0.45, 0.22), (2.5, -0.3)]
    dirty = fit_line(points, inlier_dist=0.03, min_inliers=12)
    assert dirty is not None
    assert dirty.d == pytest.approx(clean.d, abs=0.01)


def test_too_few_points_is_no_fit():
    assert fit_line([(1.0, 0.0), (1.0, 0.1)], min_inliers=12) is None


def test_empty_input_is_no_fit():
    assert fit_line([]) is None


# ----------------------------------------------------------- line pose error

def test_squared_up_at_standoff_has_zero_error():
    points = synth_wall(distance=STANDOFF, bearing=0.0)
    line = fit_line(points)
    forward, lateral, yaw = line_pose_error(line, STANDOFF)
    assert forward == pytest.approx(0.0, abs=1e-6)
    assert lateral == pytest.approx(0.0, abs=1e-6)
    assert yaw == pytest.approx(0.0, abs=1e-6)


def test_too_far_away_asks_to_drive_forward():
    points = synth_wall(distance=1.0, bearing=0.0)
    forward, _, _ = line_pose_error(fit_line(points), STANDOFF)
    assert forward == pytest.approx(1.0 - STANDOFF, abs=1e-6)
    assert forward > 0.0


def test_too_close_asks_to_back_off():
    points = synth_wall(distance=0.20, bearing=0.0)
    forward, _, _ = line_pose_error(fit_line(points), STANDOFF)
    assert forward < 0.0


def test_surface_on_the_left_asks_for_a_left_turn():
    """Sign convention regression.

    A wall whose normal bears +0.4 rad (to the robot's left) must produce a
    positive yaw error, since positive angular.z is CCW, i.e. to the left. Get
    this backwards and the servo turns away from the target every time.
    """
    points = synth_wall(distance=1.0, bearing=0.4)
    _, _, yaw = line_pose_error(fit_line(points), STANDOFF)
    assert yaw > 0.0
    assert yaw == pytest.approx(0.4, abs=1e-6)


def test_surface_on_the_right_asks_for_a_right_turn():
    points = synth_wall(distance=1.0, bearing=-0.4)
    _, _, yaw = line_pose_error(fit_line(points), STANDOFF)
    assert yaw < 0.0


def test_error_shrinks_monotonically_on_approach():
    """Walking the robot in along the normal must reduce the residual each step.

    This is the property the servo's stall detector depends on.
    """
    previous = None
    for distance in (1.2, 1.0, 0.8, 0.6, 0.45, 0.36):
        line = fit_line(synth_wall(distance=distance, bearing=0.0))
        forward, lateral, _ = line_pose_error(line, STANDOFF)
        residual = math.hypot(forward, lateral)
        if previous is not None:
            assert residual < previous, distance
        previous = residual


# ------------------------------------------------------------------ doorways

def _doorway_points(width: float, distance: float, frame_depth: float = 1.8):
    """Near wall, opening, near wall - the classic doorway signature."""
    points = []
    half = width / 2.0
    for i in range(30):                        # right-hand wall face
        y = -half - 0.02 * i
        points.append((distance, y))
    for i in range(12):                        # returns through the opening
        y = -half + width * (i + 0.5) / 12.0
        points.append((distance + frame_depth, y))
    for i in range(30):                        # left-hand wall face
        y = half + 0.02 * i
        points.append((distance, y))
    return points


def test_finds_a_standard_doorway():
    points = _doorway_points(width=0.90, distance=1.0)
    gap = find_gap(points)
    assert gap is not None
    assert gap.width == pytest.approx(0.90, abs=0.08)


def test_doorway_ahead_bears_straight_on():
    gap = find_gap(_doorway_points(width=0.90, distance=1.0))
    assert gap.bearing == pytest.approx(0.0, abs=0.05)


def test_ignores_an_opening_too_narrow_for_the_robot():
    """0.30 m of robot cannot be threaded through a 0.35 m gap with pose error."""
    assert find_gap(_doorway_points(width=0.35, distance=1.0)) is None


def test_ignores_an_open_side_of_a_room():
    """A 3 m opening is not a doorway; it is the absence of a wall."""
    assert find_gap(_doorway_points(width=3.0, distance=1.0)) is None


def test_no_gap_in_a_flat_wall():
    assert find_gap(synth_wall(distance=1.0, bearing=0.0, count=60)) is None


def test_gap_pose_error_drives_towards_the_opening():
    gap = find_gap(_doorway_points(width=0.90, distance=1.0))
    forward, _, yaw = gap_pose_error(gap, STANDOFF)
    assert forward > 0.0
    assert abs(yaw) < 0.05


def test_offset_doorway_asks_to_turn_towards_it():
    """A doorway to the left must produce a positive (leftward) yaw error."""
    points = [(x, y + 0.7) for x, y in _doorway_points(width=0.90, distance=1.2)]
    gap = find_gap(points)
    assert gap is not None
    _, lateral, yaw = gap_pose_error(gap, STANDOFF)
    assert yaw > 0.0
    assert lateral > 0.0


def test_gap_geometry_properties():
    gap = Gap(left=(1.0, 0.45), right=(1.0, -0.45))
    assert gap.width == pytest.approx(0.90)
    assert gap.centre == pytest.approx((1.0, 0.0))
    assert gap.distance == pytest.approx(1.0)
    assert gap.bearing == pytest.approx(0.0)
