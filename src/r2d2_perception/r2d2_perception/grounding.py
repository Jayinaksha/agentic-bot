#!/usr/bin/env python3
"""
Turning a VLA's 2D detections into map coordinates.

The problem: a VLA returns pixel boxes. Navigation needs metres in the map
frame. With no depth camera the only range source is the 2D LiDAR, so grounding
is a bearing-plus-range fusion:

    1. Bounding box centre -> bearing in the camera frame, via the pinhole model
       and the camera's horizontal FOV.
    2. Bearing -> LiDAR range, by sampling the scan in a wedge around that
       bearing and taking a robust statistic (not the mean: a doorway behind the
       object drags a mean badly).
    3. (bearing, range) + robot pose -> world x, y.

Accuracy is limited by the LiDAR plane. The scanner sits at one height; an object
whose lower body is not at that height (a wall-mounted switch, a mug on a table)
returns the range of whatever *is* at scanner height along that bearing - usually
the wall or the table edge behind it. That is a real limitation of a cheap
sensor suite rather than a bug, so this module reports a `grounding_quality`
alongside every estimate and lets the caller decide. Objects that cannot be
ranged are still stored, with range None, so the map records "there is a light
switch on this wall" even when it cannot say exactly where.

ROS-free so the projection maths can be tested directly.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Grounding quality bands, reported with every estimate.
QUALITY_GOOD = 'good'          # tight, consistent LiDAR support
QUALITY_COARSE = 'coarse'      # ranged, but the returns disagree
QUALITY_BEARING_ONLY = 'bearing_only'   # direction known, distance not

# Above this, a box coordinate is read as pixels rather than a normalised
# fraction. See parse_vla_detections for why it is not 1.0.
PIXEL_BOX_THRESHOLD = 1.5


@dataclass
class Detection:
    """One VLA detection, before grounding."""

    label: str
    description: str
    confidence: float
    # Normalised box in [0, 1]: (x_min, y_min, x_max, y_max).
    box: Tuple[float, float, float, float]

    @property
    def centre(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    @property
    def width(self) -> float:
        return abs(self.box[2] - self.box[0])


@dataclass
class Grounded:
    """A detection placed in the world."""

    label: str
    description: str
    confidence: float
    bearing: float                 # radians, robot frame, CCW positive
    range_m: Optional[float]
    world_x: Optional[float]
    world_y: Optional[float]
    quality: str
    support: int = 0               # LiDAR returns backing the range
    spread: Optional[float] = None  # metres, disagreement among those returns

    def as_dict(self) -> dict:
        out = {
            'label': self.label,
            'description': self.description,
            'confidence': round(self.confidence, 3),
            'bearing': round(self.bearing, 4),
            'grounding_quality': self.quality,
            'support': self.support,
        }
        if self.range_m is not None:
            out['range_m'] = round(self.range_m, 3)
        if self.spread is not None:
            out['range_spread'] = round(self.spread, 3)
        if self.world_x is not None:
            out['world_x'] = round(self.world_x, 3)
            out['world_y'] = round(self.world_y, 3)
        return out


def pixel_to_bearing(u_normalised: float, hfov: float) -> float:
    """Horizontal image position to bearing in the camera frame.

    Pinhole model. u = 0 is the left edge, 1 the right; bearing is CCW positive,
    so the left of the image is a positive bearing. Using the tangent form
    rather than a linear interpolation matters at the edges of a wide lens,
    where linear is out by several degrees.
    """
    # Map [0, 1] to [-1, 1] with +1 at the left edge.
    x = 1.0 - 2.0 * u_normalised
    return math.atan(x * math.tan(hfov / 2.0))


def angular_half_width(box_width: float, hfov: float) -> float:
    """Half-angle subtended by a box of this normalised width."""
    return max(math.atan(box_width * math.tan(hfov / 2.0)), math.radians(1.0))


@dataclass
class ScanSlice:
    """The LiDAR returns falling inside one bearing wedge."""

    ranges: List[float]

    def robust_range(self) -> Tuple[Optional[float], Optional[float], int]:
        """(range, spread, support) from the wedge.

        The median, not the mean: a detection whose wedge clips a doorway gets
        a handful of far returns from the next room, and a mean is dragged
        metres by them while a median is not. Spread is the interquartile range,
        which is the honest measure of whether the wedge is looking at one
        surface or several.
        """
        valid = sorted(r for r in self.ranges if math.isfinite(r) and r > 0.0)
        if not valid:
            return None, None, 0
        if len(valid) == 1:
            return valid[0], 0.0, 1

        median = statistics.median(valid)
        lower = statistics.median(valid[:len(valid) // 2])
        upper = statistics.median(valid[(len(valid) + 1) // 2:])
        return median, upper - lower, len(valid)


def slice_scan(ranges: Sequence[float], angle_min: float,
               angle_increment: float, bearing: float, half_width: float,
               range_min: float = 0.05,
               range_max: float = 12.0) -> ScanSlice:
    """Collect the scan returns inside [bearing - half_width, + half_width]."""
    out: List[float] = []
    angle = angle_min
    for r in ranges:
        a = angle
        angle += angle_increment
        if abs(_wrap(a - bearing)) > half_width:
            continue
        if not math.isfinite(r) or r < range_min or r > range_max:
            continue
        out.append(r)
    return ScanSlice(out)


def classify_quality(support: int, spread: Optional[float],
                     range_m: Optional[float],
                     spread_tolerance: float = 0.35,
                     min_support: int = 3) -> str:
    """How much to trust a grounding.

    Few returns, or returns that disagree by more than a body width, means the
    wedge is not looking at a single object. Saying so is more useful than
    emitting a confident coordinate that is actually the wall behind it.
    """
    if range_m is None or support == 0:
        return QUALITY_BEARING_ONLY
    if support < min_support:
        return QUALITY_COARSE
    if spread is not None and spread > spread_tolerance:
        return QUALITY_COARSE
    return QUALITY_GOOD


def project_to_world(robot_x: float, robot_y: float, robot_yaw: float,
                     bearing: float, range_m: float,
                     camera_offset_x: float = 0.0,
                     camera_offset_y: float = 0.0) -> Tuple[float, float]:
    """Robot-frame (bearing, range) to world coordinates."""
    # Camera mount offset, rotated into the world frame.
    ox = camera_offset_x * math.cos(robot_yaw) - camera_offset_y * math.sin(robot_yaw)
    oy = camera_offset_x * math.sin(robot_yaw) + camera_offset_y * math.cos(robot_yaw)
    heading = robot_yaw + bearing
    return (robot_x + ox + range_m * math.cos(heading),
            robot_y + oy + range_m * math.sin(heading))


def ground_detection(detection: Detection,
                     scan_ranges: Sequence[float],
                     angle_min: float, angle_increment: float,
                     robot_pose: Tuple[float, float, float],
                     hfov: float = 1.089,
                     camera_offset_x: float = 0.18,
                     camera_offset_y: float = 0.0,
                     scan_range_max: float = 12.0) -> Grounded:
    """Place one detection in the world, honestly reporting how well."""
    u, _ = detection.centre
    bearing = pixel_to_bearing(u, hfov)
    half_width = angular_half_width(detection.width, hfov)

    wedge = slice_scan(scan_ranges, angle_min, angle_increment,
                       bearing, half_width, range_max=scan_range_max)
    range_m, spread, support = wedge.robust_range()
    quality = classify_quality(support, spread, range_m)

    world_x = world_y = None
    if range_m is not None:
        world_x, world_y = project_to_world(
            robot_pose[0], robot_pose[1], robot_pose[2],
            bearing, range_m, camera_offset_x, camera_offset_y)

    return Grounded(
        label=detection.label,
        description=detection.description,
        confidence=detection.confidence,
        bearing=bearing,
        range_m=range_m,
        world_x=world_x,
        world_y=world_y,
        quality=quality,
        support=support,
        spread=spread,
    )


def parse_vla_detections(payload: dict) -> List[Detection]:
    """Normalise a VLA response into Detection records.

    Written against Cosmos Reason 2's object-localisation output, which returns
    2D boxes with labels and a reasoning string, but tolerant of the shapes
    other VLMs emit: boxes may arrive normalised or in pixels, as
    [x0, y0, x1, y1] or as a dict. Anything unparseable is skipped rather than
    guessed at - a detection in the wrong place is worse than a missing one.
    """
    out: List[Detection] = []
    width = float(payload.get('image_width') or 0) or None
    height = float(payload.get('image_height') or 0) or None

    for raw in payload.get('objects', payload.get('detections', [])):
        if not isinstance(raw, dict):
            continue
        label = raw.get('label') or raw.get('name')
        if not label:
            continue

        box = raw.get('box_2d') or raw.get('bbox') or raw.get('box')
        if isinstance(box, dict):
            box = [box.get('x_min'), box.get('y_min'),
                   box.get('x_max'), box.get('y_max')]
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        try:
            coords = [float(v) for v in box]
        except (TypeError, ValueError):
            continue

        # Pixel boxes are converted using the reported image size. The
        # threshold is 1.5 rather than 1.0 because VLMs routinely return
        # normalised boxes that overrun the frame slightly on a clipped object;
        # treating those as pixels would drop exactly the large, close objects
        # that matter most. Genuine pixel coordinates are orders of magnitude
        # larger, so there is no ambiguity in practice. Without an image size we
        # cannot tell 400 px from 400 units, so those are dropped rather than
        # mis-scaled - a detection in the wrong place is worse than a missing
        # one.
        if max(coords) > PIXEL_BOX_THRESHOLD:
            if not (width and height):
                continue
            coords = [coords[0] / width, coords[1] / height,
                      coords[2] / width, coords[3] / height]

        coords = [min(max(c, 0.0), 1.0) for c in coords]
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            continue

        out.append(Detection(
            label=str(label).strip().lower(),
            description=str(raw.get('description') or raw.get('reasoning') or ''),
            confidence=float(raw.get('confidence', raw.get('score', 0.6))),
            box=(coords[0], coords[1], coords[2], coords[3]),
        ))
    return out


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull a JSON object out of a model response.

    Reasoning models wrap their answer in prose or a markdown fence however
    firmly the prompt asks otherwise, and Cosmos Reason emits a chain of thought
    before the answer. Rather than trying to forbid that, find the outermost
    balanced object in the response - taking the last one, since the answer
    follows the reasoning.
    """
    if not text:
        return None

    fence = text.rfind('```')
    if fence != -1:
        opening = text.rfind('```', 0, fence)
        if opening != -1:
            candidate = text[opening + 3:fence]
            if candidate.startswith('json'):
                candidate = candidate[4:]
            parsed = _try_json(candidate)
            if parsed is not None:
                return parsed

    depth = 0
    end = -1
    for i in range(len(text) - 1, -1, -1):
        char = text[i]
        if char == '}':
            if depth == 0:
                end = i
            depth += 1
        elif char == '{':
            depth -= 1
            if depth == 0 and end != -1:
                parsed = _try_json(text[i:end + 1])
                if parsed is not None:
                    return parsed
                end = -1
    return None


def _try_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None
