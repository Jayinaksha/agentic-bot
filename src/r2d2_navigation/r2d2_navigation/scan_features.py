#!/usr/bin/env python3
"""
Geometric feature extraction from a 2D laser scan.

Used by precise_docking to close the last 20 cm of an approach against what the
robot can currently see, rather than against a map pose that is only good to a
few centimetres.

No ROS imports, so the geometry is unit-testable on a bare interpreter
(test/test_scan_features.py).

Conventions throughout: robot frame, +x forward, +y left, angles CCW from +x.
"""

import math
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]


def polar_to_xy(range_m: float, angle: float) -> Point:
    return range_m * math.cos(angle), range_m * math.sin(angle)


class Line:
    """A 2D line in normal form: points p satisfy dot(n, p) = d, |n| = 1.

    `n` points from the origin towards the line, so `d` is the perpendicular
    distance from the robot to the surface and atan2(n) is the bearing of the
    surface normal.
    """

    __slots__ = ('nx', 'ny', 'd', 'inliers')

    def __init__(self, nx: float, ny: float, d: float, inliers: int):
        # Normalise so d is a true distance and the normal faces away from the
        # robot; the sign convention is what makes standoff arithmetic simple.
        norm = math.hypot(nx, ny)
        if norm < 1e-12:
            raise ValueError('degenerate line normal')
        nx, ny, d = nx / norm, ny / norm, d / norm
        if d < 0.0:
            nx, ny, d = -nx, -ny, -d
        self.nx, self.ny, self.d, self.inliers = nx, ny, d, inliers

    @property
    def normal_bearing(self) -> float:
        """Bearing of the surface normal, radians in the robot frame."""
        return math.atan2(self.ny, self.nx)

    def distance_to(self, p: Point) -> float:
        return abs(self.nx * p[0] + self.ny * p[1] - self.d)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f'Line(d={self.d:.3f}, '
                f'bearing={math.degrees(self.normal_bearing):.1f}deg, '
                f'inliers={self.inliers})')


def _fit_total_least_squares(points: Sequence[Point]) -> Optional[Line]:
    """Best-fit line through points, minimising perpendicular distance.

    Ordinary least squares minimises vertical residuals and blows up on a wall
    that happens to be parallel to the y axis - which, for a robot facing a
    fridge door, is the normal case. Total least squares (the principal axis of
    the scatter) has no preferred direction.
    """
    n = len(points)
    if n < 2:
        return None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n

    sxx = syy = sxy = 0.0
    for x, y in points:
        dx, dy = x - mx, y - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy

    # Smaller eigenvector of the 2x2 scatter matrix is the line normal.
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    nx, ny = -math.sin(theta), math.cos(theta)
    d = nx * mx + ny * my
    if abs(d) < 1e-9:
        # Line passes through the robot: no usable standoff geometry.
        return None
    try:
        return Line(nx, ny, d, n)
    except ValueError:
        return None


def fit_line(points: Sequence[Point], inlier_dist: float = 0.03,
             min_inliers: int = 12, iterations: int = 2) -> Optional[Line]:
    """Fit a surface, rejecting outliers by iterative trimming.

    Full RANSAC is overkill here: the search window is already narrow, so the
    dominant structure is the target surface and a couple of trimming passes
    removes the door handle, the skirting board and the odd stray return.
    """
    if len(points) < min_inliers:
        return None

    working = list(points)
    line = _fit_total_least_squares(working)
    if line is None:
        return None

    for _ in range(iterations):
        kept = [p for p in working if line.distance_to(p) <= inlier_dist]
        if len(kept) < min_inliers:
            break
        if len(kept) == len(working):
            break
        working = kept
        refined = _fit_total_least_squares(working)
        if refined is None:
            break
        line = refined

    if line.inliers < min_inliers:
        return None
    return line


def line_pose_error(line: Line, standoff: float) -> Tuple[float, float, float]:
    """Pose error relative to "square on, `standoff` metres off the surface".

    Returns (forward, lateral, yaw), all in the robot frame:

        forward  how much further to drive along +x, positive means approach
        lateral  how far the target centre sits to the left, positive means left
        yaw      how much to rotate CCW to face the surface square on

    Facing the surface square on means the surface normal points straight back
    along -x from the robot, i.e. its bearing is 0 in the robot frame.
    """
    yaw_error = _wrap(line.normal_bearing)
    # Perpendicular distance is what standoff is defined against; the component
    # along the current heading follows from the yaw error.
    forward = (line.d - standoff) * math.cos(yaw_error)
    lateral = (line.d - standoff) * math.sin(yaw_error)
    return forward, lateral, yaw_error


class Gap:
    """An opening between two edges, e.g. a doorway."""

    __slots__ = ('left', 'right')

    def __init__(self, left: Point, right: Point):
        self.left = left
        self.right = right

    @property
    def width(self) -> float:
        return math.dist(self.left, self.right)

    @property
    def centre(self) -> Point:
        return ((self.left[0] + self.right[0]) / 2.0,
                (self.left[1] + self.right[1]) / 2.0)

    @property
    def bearing(self) -> float:
        cx, cy = self.centre
        return math.atan2(cy, cx)

    @property
    def distance(self) -> float:
        return math.hypot(*self.centre)

    @property
    def normal_bearing(self) -> float:
        """Bearing of the opening's axis normal - the direction to pass through.

        The gap's two edges define a chord; passing through it squarely means
        travelling perpendicular to that chord.
        """
        dx = self.left[0] - self.right[0]
        dy = self.left[1] - self.right[1]
        return math.atan2(-dx, dy)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f'Gap(width={self.width:.2f}, '
                f'bearing={math.degrees(self.bearing):.1f}deg, '
                f'distance={self.distance:.2f})')


def find_gap(points: Sequence[Point], min_width: float = 0.55,
             max_width: float = 1.60,
             min_jump: float = 0.30) -> Optional[Gap]:
    """Find the widest passable opening in an angularly sorted point set.

    A doorway shows up in a scan as a pair of large range discontinuities: the
    beam hits the near frame, then jumps past it into the next room, then comes
    back on the far frame. We look for that jump pair.

    min_width is the narrowest opening worth attempting. The platform is 0.30 m
    wide, so 0.55 m leaves 12 cm of clearance on each side - about the least a
    skid-steer with pose uncertainty can be asked to thread.
    """
    if len(points) < 4:
        return None

    ordered = sorted(points, key=lambda p: math.atan2(p[1], p[0]))
    ranges = [math.hypot(x, y) for x, y in ordered]

    # Rising edges (near -> far) and falling edges (far -> near).
    rising = [i for i in range(len(ranges) - 1)
              if ranges[i + 1] - ranges[i] > min_jump]
    falling = [i for i in range(len(ranges) - 1)
               if ranges[i] - ranges[i + 1] > min_jump]

    best: Optional[Gap] = None
    for r in rising:
        for f in falling:
            if f <= r:
                continue
            gap = Gap(left=ordered[f + 1], right=ordered[r])
            if not min_width <= gap.width <= max_width:
                continue
            if best is None or gap.width > best.width:
                best = gap
    return best


def gap_pose_error(gap: Gap, standoff: float) -> Tuple[float, float, float]:
    """Pose error relative to "centred on the gap, `standoff` short of it"."""
    yaw_error = _wrap(gap.bearing)
    forward = gap.distance - standoff
    lateral = gap.distance * math.sin(yaw_error)
    return forward * math.cos(yaw_error), lateral, yaw_error


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def synth_wall(distance: float, bearing: float, half_extent: float = 0.6,
               count: int = 40, noise: float = 0.0,
               rng=None) -> List[Point]:
    """Generate scan points lying on a flat surface. Used by the tests."""
    nx, ny = math.cos(bearing), math.sin(bearing)
    tx, ty = -ny, nx
    out = []
    for i in range(count):
        t = -half_extent + 2 * half_extent * i / max(count - 1, 1)
        x = distance * nx + t * tx
        y = distance * ny + t * ty
        if noise and rng is not None:
            x += rng.gauss(0.0, noise)
            y += rng.gauss(0.0, noise)
        out.append((x, y))
    return out
