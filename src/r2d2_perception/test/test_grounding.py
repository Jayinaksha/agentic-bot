#!/usr/bin/env python3
"""Unit tests for VLA detection grounding.

    python3 -m pytest src/r2d2_perception/test/test_grounding.py

This is the code that decides where an object goes on the map. A sign error in
the bearing puts every detection on the wrong side of the robot, which looks
plausible right up until navigation drives the wrong way.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from r2d2_perception.grounding import (  # noqa: E402
    QUALITY_BEARING_ONLY, QUALITY_COARSE, QUALITY_GOOD, Detection, ScanSlice,
    angular_half_width, classify_quality, extract_json, ground_detection,
    parse_vla_detections, pixel_to_bearing, project_to_world, slice_scan)

HFOV = 1.089            # 62.4 deg, matching the sim camera
ANGLE_MIN = -math.pi
INCREMENT = 2 * math.pi / 720


def flat_scan(distance: float, count: int = 720):
    return [distance] * count


# ------------------------------------------------------------------ bearings

def test_image_centre_is_straight_ahead():
    assert pixel_to_bearing(0.5, HFOV) == pytest.approx(0.0, abs=1e-12)


def test_left_of_image_is_a_positive_bearing():
    """Sign regression: +y is left in REP-103, and so is +bearing."""
    assert pixel_to_bearing(0.25, HFOV) > 0.0


def test_right_of_image_is_a_negative_bearing():
    assert pixel_to_bearing(0.75, HFOV) < 0.0


def test_image_edges_land_on_the_fov_limits():
    assert pixel_to_bearing(0.0, HFOV) == pytest.approx(HFOV / 2, abs=1e-9)
    assert pixel_to_bearing(1.0, HFOV) == pytest.approx(-HFOV / 2, abs=1e-9)


def test_bearing_is_monotonic_across_the_image():
    bearings = [pixel_to_bearing(u / 20.0, HFOV) for u in range(21)]
    assert all(b > a for a, b in zip(bearings[1:], bearings[:-1]))


def test_tangent_model_differs_from_linear_at_the_edge():
    """Why the tangent form is used: a linear map is several degrees out."""
    u = 0.05
    tangent = pixel_to_bearing(u, HFOV)
    linear = (0.5 - u) * HFOV
    assert abs(math.degrees(tangent - linear)) > 0.5


def test_wide_boxes_subtend_wider_wedges():
    assert angular_half_width(0.4, HFOV) > angular_half_width(0.1, HFOV)


def test_tiny_boxes_still_get_a_usable_wedge():
    """A one-pixel box must not produce a wedge containing zero LiDAR beams."""
    assert angular_half_width(0.0, HFOV) >= math.radians(1.0)


# -------------------------------------------------------------- scan slicing

def test_slice_picks_up_beams_near_the_bearing():
    sliced = slice_scan(flat_scan(2.0), ANGLE_MIN, INCREMENT,
                        bearing=0.0, half_width=0.1)
    assert sliced.ranges
    assert all(r == 2.0 for r in sliced.ranges)


def test_slice_rejects_out_of_range_returns():
    ranges = [0.01] * 720
    assert not slice_scan(ranges, ANGLE_MIN, INCREMENT, 0.0, 0.2,
                          range_min=0.05).ranges


def test_slice_rejects_infinities():
    assert not slice_scan([math.inf] * 720, ANGLE_MIN, INCREMENT, 0.0, 0.2).ranges


def test_median_ignores_a_doorway_behind_the_object():
    """The reason a median is used instead of a mean.

    A wedge that clips an open doorway picks up a few 9 m returns from the next
    room. A mean is dragged more than a metre; the median is not.
    """
    ranges = [1.5] * 12 + [9.0] * 4
    sliced = ScanSlice(ranges)
    median, spread, support = sliced.robust_range()
    assert median == pytest.approx(1.5)
    assert support == 16
    assert abs(sum(ranges) / len(ranges) - 1.5) > 1.0     # a mean would fail


def test_single_return_has_no_spread():
    median, spread, support = ScanSlice([2.5]).robust_range()
    assert (median, spread, support) == (2.5, 0.0, 1)


def test_empty_slice_reports_nothing():
    assert ScanSlice([]).robust_range() == (None, None, 0)


# ------------------------------------------------------------------- quality

def test_consistent_returns_are_good():
    assert classify_quality(support=20, spread=0.04, range_m=1.5) == QUALITY_GOOD


def test_disagreeing_returns_are_coarse():
    assert classify_quality(support=20, spread=1.2, range_m=1.5) == QUALITY_COARSE


def test_thin_support_is_coarse():
    assert classify_quality(support=2, spread=0.01, range_m=1.5) == QUALITY_COARSE


def test_no_range_is_bearing_only():
    assert classify_quality(support=0, spread=None, range_m=None) == QUALITY_BEARING_ONLY


# ---------------------------------------------------------------- projection

def test_object_ahead_of_a_robot_at_the_origin():
    x, y = project_to_world(0.0, 0.0, 0.0, bearing=0.0, range_m=2.0,
                            camera_offset_x=0.0)
    assert (x, y) == pytest.approx((2.0, 0.0))


def test_object_to_the_left():
    x, y = project_to_world(0.0, 0.0, 0.0, bearing=math.pi / 2, range_m=2.0,
                            camera_offset_x=0.0)
    assert (x, y) == pytest.approx((0.0, 2.0), abs=1e-9)


def test_robot_heading_rotates_the_projection():
    """Robot facing +y; an object dead ahead must land on +y, not +x."""
    x, y = project_to_world(0.0, 0.0, math.pi / 2, bearing=0.0, range_m=3.0,
                            camera_offset_x=0.0)
    assert (x, y) == pytest.approx((0.0, 3.0), abs=1e-9)


def test_camera_offset_is_rotated_with_the_robot():
    x, y = project_to_world(1.0, 1.0, math.pi / 2, bearing=0.0, range_m=1.0,
                            camera_offset_x=0.2)
    assert (x, y) == pytest.approx((1.0, 2.2), abs=1e-9)


def test_projection_preserves_distance_from_the_camera():
    for yaw in (0.0, 0.7, -1.9, 3.0):
        for bearing in (-0.4, 0.0, 0.5):
            x, y = project_to_world(2.0, -1.0, yaw, bearing, 4.0,
                                    camera_offset_x=0.0)
            assert math.dist((2.0, -1.0), (x, y)) == pytest.approx(4.0)


# ------------------------------------------------------------ end to end

def test_grounds_a_centred_detection():
    detection = Detection('chair', 'a wooden chair', 0.8, (0.45, 0.4, 0.55, 0.8))
    grounded = ground_detection(detection, flat_scan(2.0), ANGLE_MIN, INCREMENT,
                                robot_pose=(0.0, 0.0, 0.0), hfov=HFOV,
                                camera_offset_x=0.0)
    assert grounded.quality == QUALITY_GOOD
    assert grounded.range_m == pytest.approx(2.0)
    assert grounded.world_x == pytest.approx(2.0, abs=0.05)
    assert grounded.world_y == pytest.approx(0.0, abs=0.05)


def test_a_detection_on_the_left_lands_on_the_left():
    detection = Detection('fridge', '', 0.9, (0.05, 0.2, 0.25, 0.9))
    grounded = ground_detection(detection, flat_scan(3.0), ANGLE_MIN, INCREMENT,
                                robot_pose=(0.0, 0.0, 0.0), hfov=HFOV,
                                camera_offset_x=0.0)
    assert grounded.bearing > 0.0
    assert grounded.world_y > 0.0


def test_an_unrangeable_detection_keeps_its_bearing():
    """A light switch above the scan plane: direction known, distance not.

    It must still be reported, because "there is a switch on this wall" is
    useful even without a coordinate.
    """
    detection = Detection('light switch', '', 0.7, (0.48, 0.1, 0.52, 0.2))
    grounded = ground_detection(detection, [math.inf] * 720, ANGLE_MIN,
                                INCREMENT, robot_pose=(1.0, 2.0, 0.0),
                                hfov=HFOV)
    assert grounded.quality == QUALITY_BEARING_ONLY
    assert grounded.range_m is None
    assert grounded.world_x is None
    assert grounded.bearing == pytest.approx(0.0, abs=0.05)


def test_grounded_serialisation_omits_what_it_does_not_know():
    detection = Detection('switch', '', 0.7, (0.48, 0.1, 0.52, 0.2))
    grounded = ground_detection(detection, [math.inf] * 720, ANGLE_MIN,
                                INCREMENT, robot_pose=(0.0, 0.0, 0.0))
    d = grounded.as_dict()
    assert 'world_x' not in d
    assert 'range_m' not in d
    assert d['grounding_quality'] == QUALITY_BEARING_ONLY


# --------------------------------------------------------------- VLA parsing

def test_parses_normalised_boxes():
    detections = parse_vla_detections({
        'objects': [{'label': 'Chair', 'box_2d': [0.1, 0.2, 0.3, 0.6],
                     'confidence': 0.85, 'description': 'wooden'}]})
    assert len(detections) == 1
    assert detections[0].label == 'chair'          # normalised to lower case
    assert detections[0].confidence == pytest.approx(0.85)


def test_parses_pixel_boxes_when_the_image_size_is_given():
    detections = parse_vla_detections({
        'image_width': 640, 'image_height': 480,
        'objects': [{'name': 'sofa', 'bbox': [64, 96, 192, 288]}]})
    assert detections[0].box == pytest.approx((0.1, 0.2, 0.3, 0.6))


def test_drops_pixel_boxes_with_no_image_size():
    """Guessing the scale would put the object somewhere arbitrary."""
    assert parse_vla_detections({
        'objects': [{'label': 'sofa', 'bbox': [64, 96, 192, 288]}]}) == []


def test_parses_dict_shaped_boxes():
    detections = parse_vla_detections({
        'detections': [{'label': 'table',
                        'box': {'x_min': 0.2, 'y_min': 0.3,
                                'x_max': 0.5, 'y_max': 0.7}}]})
    assert detections[0].box == pytest.approx((0.2, 0.3, 0.5, 0.7))


def test_skips_unlabelled_and_malformed_entries():
    detections = parse_vla_detections({'objects': [
        {'box_2d': [0.1, 0.1, 0.2, 0.2]},                 # no label
        {'label': 'x', 'box_2d': [0.1, 0.1]},             # short box
        {'label': 'y', 'box_2d': ['a', 'b', 'c', 'd']},   # non-numeric
        {'label': 'z', 'box_2d': [0.5, 0.5, 0.4, 0.4]},   # inverted
        'not a dict',
        {'label': 'good', 'box_2d': [0.1, 0.1, 0.2, 0.2]},
    ]})
    assert [d.label for d in detections] == ['good']


def test_clamps_boxes_that_run_off_the_image():
    detections = parse_vla_detections({
        'objects': [{'label': 'wall', 'box_2d': [-0.2, -0.1, 0.5, 1.4]}]})
    assert detections[0].box == pytest.approx((0.0, 0.0, 0.5, 1.0))


def test_uses_reasoning_as_a_description_when_present():
    """Cosmos Reason 2 returns a reasoning string rather than a description."""
    detections = parse_vla_detections({
        'objects': [{'label': 'chair', 'box_2d': [0.1, 0.1, 0.2, 0.2],
                     'reasoning': 'four legs and a back, near the table'}]})
    assert 'four legs' in detections[0].description


def test_empty_payload_is_no_detections():
    assert parse_vla_detections({}) == []


# ------------------------------------------------ model response extraction

def test_extracts_a_bare_json_object():
    assert extract_json('{"room": "kitchen"}') == {'room': 'kitchen'}


def test_extracts_from_a_markdown_fence():
    text = 'Here is what I see:\n```json\n{"room": "study"}\n```\n'
    assert extract_json(text) == {'room': 'study'}


def test_extracts_from_an_unlabelled_fence():
    assert extract_json('```\n{"room": "hall"}\n```') == {'room': 'hall'}


def test_extracts_the_answer_after_a_chain_of_thought():
    """Cosmos Reason emits reasoning before its answer; take the last object."""
    text = ('Let me think. The scene has a table {this is not json} and chairs.\n'
            'Final answer:\n{"room": "kitchen", "objects": []}')
    assert extract_json(text) == {'room': 'kitchen', 'objects': []}


def test_handles_nested_objects():
    text = 'answer: {"objects": [{"label": "chair", "box_2d": [0,0,1,1]}]}'
    parsed = extract_json(text)
    assert parsed['objects'][0]['label'] == 'chair'


def test_prose_with_no_json_returns_none():
    assert extract_json('I am not sure what I am looking at.') is None


def test_empty_response_returns_none():
    assert extract_json('') is None
    assert extract_json(None) is None


def test_a_bare_array_is_not_accepted():
    """The contract is an object; an array would break every consumer."""
    assert extract_json('[1, 2, 3]') is None


def test_truncated_json_returns_none_rather_than_guessing():
    assert extract_json('{"room": "kitchen", "objects": [{"label"') is None


# ------------------------------------------------ credential redaction
#
# vla_node keeps its own copy of this helper rather than importing from
# r2d2_mcp: perception must run without the agent installed. Duplication of a
# four-line security helper is the better trade, but it has to be tested in both
# places or the copies drift.

def _redact(text):
    import re
    return re.sub(r'(\w+://)[^/@\s]+@', r'\1***@', text)


def test_vla_redaction_strips_userinfo():
    assert _redact('https://admin:HUNTER2@nim.internal:8000/v1') == \
        'https://***@nim.internal:8000/v1'


def test_vla_redaction_leaves_clean_urls_alone():
    url = 'https://integrate.api.nvidia.com/v1'
    assert _redact(url) == url


def test_vla_redaction_matches_the_mcp_copy():
    """The two copies must behave identically, or one of them is wrong."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                    'r2d2_mcp'))
    from r2d2_mcp.config import redact_url
    for sample in ('https://a:b@h/v1',
                   'https://clean.example/v1',
                   'postgresql://u:p@db:5432/x',
                   'no url here',
                   'two https://a:b@one/v1 and https://c:d@two/v1'):
        assert _redact(sample) == redact_url(sample), sample
