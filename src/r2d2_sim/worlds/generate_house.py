#!/usr/bin/env python3
"""
Generate a two-storey house world for Gazebo Sim (Harmonic / gz-sim 8).

Why generate instead of hand-writing SDF:

  * The staircase must match the robot. Riser height, tread depth and stair
    width are read from r2d2_description/config/robot_params.yaml - the same
    file the URDF is built from - so the platform can never end up facing a
    staircase it is geometrically unable to mount. Change the cluster radius
    and both the robot and the stairs follow.
  * Room layout is data, not markup. Editing ROOMS below is far easier than
    editing several hundred lines of <model> blocks.

Usage:
    python3 generate_house.py                 # writes house_two_floor.sdf
    python3 generate_house.py --out other.sdf
    python3 generate_house.py --no-ramp       # stairs only (harder)

Layout (metres, ground floor at z=0, upper floor slab at z=FLOOR_H):

    y
    ^   +-------------------+-------------------+
    |   |    bedroom        |     study         |
    |   |                   |                   |
  U |   +----+----+---------+---------+---------+     upper floor
    |   |         hallway (landing)             |
        +-------------------+-------------------+

    ^   +-------------------+-------------------+
    |   |    kitchen        |   living_room     |
  G |   |                   |                   |
    |   +----+----+---------+----+----+---------+
    |   |         hallway  [stairs][ramp]       |
        +-------------------+-------------------+ --> x
"""

import argparse
import math
import os
import sys

try:
    import yaml
except ImportError:  # pragma: no cover - yaml ships with ROS
    sys.exit("PyYAML required: pip install pyyaml")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PARAMS = os.path.normpath(
    os.path.join(HERE, '..', '..', 'r2d2_description', 'config', 'robot_params.yaml'))

# --- Staircase placement --------------------------------------------------
# The flight runs east along the hallway at STAIR_Y, rising in +x.
#
# STAIR_X0 is not a free choice. The robot must square up to the flight before
# committing, and it aligns by yawing on the spot, so it needs at least its own
# length plus turning room clear of the bottom step. At 0.20 m (the first draft)
# a 0.36 m robot could not approach the stairs at all.
STAIR_X0 = 1.40        # m, west edge of the bottom step
STAIR_Y = 0.55         # m, centreline of the flight
MIN_APPROACH_CLEARANCE = 1.00   # m of clear floor west of the bottom step

# --- House envelope -------------------------------------------------------
HOUSE_W = 9.0          # x extent (m)
HOUSE_D = 7.0          # y extent (m)
WALL_T = 0.12          # wall thickness (m)
WALL_H = 2.4           # storey clear height (m)
SLAB_T = 0.20          # upper floor slab thickness (m)
DOOR_W = 0.90          # doorway clear width (m)

# Rooms are axis-aligned boxes: (name, x0, y0, x1, y1). Interior partitions are
# derived from these; the outer shell is drawn separately.
ROOMS_GROUND = [
    ('kitchen',     0.0, 2.2, 4.4, 7.0),
    ('living_room', 4.6, 2.2, 9.0, 7.0),
    ('hallway_g',   0.0, 0.0, 9.0, 2.0),
]
ROOMS_UPPER = [
    ('bedroom',     0.0, 2.2, 4.4, 7.0),
    ('study',       4.6, 2.2, 9.0, 7.0),
    ('landing',     0.0, 0.0, 9.0, 2.0),
]

# Doorways as (wall_axis, position_along_wall, fixed_coord) gaps punched into
# partitions. Encoded as explicit gap segments in _partition().
DOORS_GROUND = [
    ('x', 2.2, 2.1),   # hallway -> kitchen
    ('x', 6.6, 2.1),   # hallway -> living room
    ('y', 4.5, 4.6),   # kitchen -> living room
]
DOORS_UPPER = [
    ('x', 2.2, 2.1),
    ('x', 6.6, 2.1),
    ('y', 4.5, 4.6),
]

# Uneven terrain: low thresholds and a slightly warped floor patch. These are
# what the platform meets in *normal* driving - the stairs are the extreme case,
# these are the everyday case that a plain 2D-LiDAR robot silently trips on.
THRESHOLDS = [
    # (x, y, yaw, length, height) - door sills
    (2.2, 2.1, 0.0, DOOR_W, 0.022),
    (6.6, 2.1, 0.0, DOOR_W, 0.030),
    (4.6, 4.5, math.pi / 2, DOOR_W, 0.018),
]
# (x, y, size_x, size_y, height) - a rucked-up rug / uneven tiling patch
FLOOR_PATCHES = [
    (1.6, 4.0, 1.2, 0.9, 0.035),
    (7.2, 5.2, 1.0, 1.0, 0.028),
]


def _box(name, x, y, z, sx, sy, sz, yaw=0.0, rgba=(0.85, 0.83, 0.80, 1.0),
         static=True, mu=1.0):
    """One static collision+visual box, emitted as its own SDF model."""
    r, g, b, a = rgba
    return f"""
    <model name="{name}">
      <static>{'true' if static else 'false'}</static>
      <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw:.5f}</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
          <surface>
            <friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction>
            <contact><ode><kp>1e7</kp><kd>100</kd></ode></contact>
          </surface>
        </collision>
        <visual name="visual">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
          <material>
            <ambient>{r} {g} {b} {a}</ambient>
            <diffuse>{r} {g} {b} {a}</diffuse>
          </material>
        </visual>
      </link>
    </model>"""


def _wall_with_gaps(name, axis, fixed, start, end, gaps, z0, height, rgba):
    """A wall along `axis` from `start` to `end` at `fixed`, minus door gaps.

    Gaps are clamped to the wall, and ones lying entirely off it are ignored.
    Door lists are shared between walls of different extents - the corridor runs
    the full width of the house, the room divider only spans the rooms - so a
    gap that misses a given wall is expected rather than an error.

    Without the clamp a gap beyond `end` walked the cursor past the wall and
    emitted a segment reaching to the gap: a door mistyped at x=99 produced a
    98.55 m wall shooting across the map instead of a 9 m one. Silent, and
    baffling to debug in a simulator.
    """
    segments = []
    cuts = []
    for gap_centre, gap_w in gaps:
        low = gap_centre - gap_w / 2.0
        high = gap_centre + gap_w / 2.0
        if high <= start or low >= end:
            continue
        cuts.append((max(low, start), min(high, end)))
    cuts.sort()

    cursor = start
    pieces = []
    for lo, hi in cuts:
        if lo > cursor:
            pieces.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < end:
        pieces.append((cursor, end))

    for i, (a, b) in enumerate(pieces):
        length = b - a
        if length <= 1e-3:
            continue
        mid = (a + b) / 2.0
        if axis == 'x':
            segments.append(_box(f"{name}_{i}", mid, fixed, z0 + height / 2.0,
                                 length, WALL_T, height, rgba=rgba))
        else:
            segments.append(_box(f"{name}_{i}", fixed, mid, z0 + height / 2.0,
                                 WALL_T, length, height, rgba=rgba))
    return segments


def build_staircase(cfg, x0, y0, yaw=0.0):
    """Straight-flight staircase built from the robot's own climb envelope.

    Each step is a solid box. Steps are emitted individually rather than as a
    single mesh so contact with the tri-star sub-wheels is crisp - a mesh
    ramp-approximation is exactly what makes climbing look wrong in sim.
    """
    st = cfg['stairs']
    riser, tread, width, n = st['riser'], st['tread'], st['width'], st['steps']

    reach = cfg['platform']['cluster_circumradius'] + cfg['platform']['sub_wheel_radius']
    if riser >= reach:
        raise SystemExit(
            f"stairs.riser ({riser} m) >= cluster reach ({reach:.3f} m): "
            "this platform cannot mount that step. Raise cluster_circumradius "
            "or lower stairs.riser in robot_params.yaml.")
    if tread <= 2 * cfg['platform']['sub_wheel_radius']:
        raise SystemExit(
            f"stairs.tread ({tread} m) <= sub-wheel diameter: the cluster will "
            "bridge two risers instead of landing on the tread.")

    out = []
    for i in range(n):
        # Step i is a box whose top face is at (i+1)*riser. Boxes are stacked
        # from the ground up so each is fully supported.
        h = (i + 1) * riser
        cx = x0 + (i + 0.5) * tread * math.cos(yaw)
        cy = y0 + (i + 0.5) * tread * math.sin(yaw)
        out.append(_box(f"stair_step_{i}", cx, cy, h / 2.0,
                        tread, width, h, yaw=yaw,
                        rgba=(0.62, 0.55, 0.45, 1.0), mu=1.1))

    # Side stringers keep the LiDAR from seeing straight through the flight,
    # which is what a real staircase looks like to a planar scan.
    flight_len = n * tread
    for side, sgn in (('l', +1), ('r', -1)):
        sx_ = x0 + flight_len / 2.0 * math.cos(yaw) - sgn * (width / 2.0 + 0.05) * math.sin(yaw)
        sy_ = y0 + flight_len / 2.0 * math.sin(yaw) + sgn * (width / 2.0 + 0.05) * math.cos(yaw)
        out.append(_box(f"stair_stringer_{side}", sx_, sy_, (n * riser) / 2.0,
                        flight_len, 0.08, n * riser, yaw=yaw,
                        rgba=(0.45, 0.40, 0.34, 1.0)))
    return out, flight_len


def build_ramp(x0, y0, floor_h, length, width, yaw=0.0):
    """A straight ramp as the 'legal' wheelchair-style route between floors.

    Slope is reported by the generator so you can check it against
    limits.max_pitch_flat before trusting the flat-ground controller on it.
    """
    slope = math.atan2(floor_h, length)
    cx = x0 + (length / 2.0) * math.cos(yaw)
    cy = y0 + (length / 2.0) * math.sin(yaw)
    cz = floor_h / 2.0
    hyp = math.hypot(length, floor_h)
    model = f"""
    <model name="floor_ramp">
      <static>true</static>
      <pose>{cx:.4f} {cy:.4f} {cz:.4f} 0 {-slope:.5f} {yaw:.5f}</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{hyp:.4f} {width:.4f} 0.08</size></box></geometry>
          <surface><friction><ode><mu>1.2</mu><mu2>1.0</mu2></ode></friction></surface>
        </collision>
        <visual name="visual">
          <geometry><box><size>{hyp:.4f} {width:.4f} 0.08</size></box></geometry>
          <material><ambient>0.5 0.5 0.55 1</ambient><diffuse>0.5 0.5 0.55 1</diffuse></material>
        </visual>
      </link>
    </model>"""
    return model, math.degrees(slope)


def floor_transitions(cfg):
    """Where the staircase is, in the form floor_manager needs.

    Derived from the same constants the world is built from, so the navigation
    layer cannot end up aiming at a staircase that has moved. `foot` is the pose
    the robot drives to before climbing and must be clear of the bottom step;
    `head` is where it ends up on the upper floor, past the stairwell opening.
    """
    st = cfg['stairs']
    flight = st['steps'] * st['tread']
    return [{
        'from_floor': 0,
        'to_floor': 1,
        'kind': 'stairs',
        # Half the clearance west of the flight: room to turn, close enough that
        # the ToF beams already see the first riser.
        'foot': {'x': round(STAIR_X0 - MIN_APPROACH_CLEARANCE / 2.0, 3),
                 'y': STAIR_Y},
        # Past the east edge of the stairwell, so the robot is on solid slab.
        'head': {'x': round(STAIR_X0 + flight + 0.55, 3), 'y': STAIR_Y},
        'heading': 0.0,
    }]


def build_world(cfg, with_ramp=True):
    st = cfg['stairs']
    floor_h = st['riser'] * st['steps']          # floor-to-floor rise
    upper_z = floor_h                            # top of upper slab surface

    parts = []
    notes = []

    # --- outer shell, both storeys -----------------------------------------
    for storey, z0 in (('g', 0.0), ('u', upper_z)):
        rgba = (0.88, 0.86, 0.82, 1.0) if storey == 'g' else (0.84, 0.86, 0.90, 1.0)
        parts += _wall_with_gaps(f"shell_{storey}_south", 'x', 0.0, 0.0, HOUSE_W,
                                 [(1.0, 1.1)] if storey == 'g' else [], z0, WALL_H, rgba)
        parts += _wall_with_gaps(f"shell_{storey}_north", 'x', HOUSE_D, 0.0, HOUSE_W,
                                 [], z0, WALL_H, rgba)
        parts += _wall_with_gaps(f"shell_{storey}_west", 'y', 0.0, 0.0, HOUSE_D,
                                 [], z0, WALL_H, rgba)
        parts += _wall_with_gaps(f"shell_{storey}_east", 'y', HOUSE_W, 0.0, HOUSE_D,
                                 [], z0, WALL_H, rgba)

    # --- interior partitions ------------------------------------------------
    for storey, z0, doors in (('g', 0.0, DOORS_GROUND), ('u', upper_z, DOORS_UPPER)):
        rgba = (0.90, 0.89, 0.86, 1.0)
        x_gaps = [(d[1], DOOR_W) for d in doors if d[0] == 'x']
        y_gaps = [(d[1], DOOR_W) for d in doors if d[0] == 'y']
        # Corridor wall running east-west at y = 2.1
        parts += _wall_with_gaps(f"part_{storey}_corridor", 'x', 2.1, 0.0, HOUSE_W,
                                 x_gaps, z0, WALL_H, rgba)
        # Room divider running north-south at x = 4.6
        parts += _wall_with_gaps(f"part_{storey}_divider", 'y', 4.6, 2.2, HOUSE_D,
                                 y_gaps, z0, WALL_H, rgba)

    # --- upper floor slab ---------------------------------------------------
    # The stairwell is left open so the robot can arrive. Its east edge is the
    # end of the flight EXACTLY: any gap there is a hole between the top tread
    # and the slab, and the robot drives into it having just finished climbing.
    flight_span = st['steps'] * st['tread']
    well_x0, well_x1 = STAIR_X0, STAIR_X0 + flight_span
    slab_z = upper_z - SLAB_T / 2.0
    # slab is split into three pieces around the well (which spans y in [0, 2.0])
    parts.append(_box('slab_north', HOUSE_W / 2.0, (2.0 + HOUSE_D) / 2.0, slab_z,
                      HOUSE_W, HOUSE_D - 2.0, SLAB_T, rgba=(0.75, 0.72, 0.68, 1)))
    if well_x1 < HOUSE_W:
        parts.append(_box('slab_south_east', (well_x1 + HOUSE_W) / 2.0, 1.0, slab_z,
                          HOUSE_W - well_x1, 2.0, SLAB_T, rgba=(0.75, 0.72, 0.68, 1)))
    if well_x0 > 0.0:
        parts.append(_box('slab_south_west', well_x0 / 2.0, 1.0, slab_z,
                          well_x0, 2.0, SLAB_T, rgba=(0.75, 0.72, 0.68, 1)))

    # --- staircase ----------------------------------------------------------
    if STAIR_X0 < MIN_APPROACH_CLEARANCE:
        raise SystemExit(
            f'STAIR_X0 ({STAIR_X0} m) leaves less than '
            f'{MIN_APPROACH_CLEARANCE} m of floor west of the bottom step. The '
            f'robot cannot square up to a flight it is already standing on.')

    stairs, flight_len = build_staircase(cfg, x0=STAIR_X0, y0=STAIR_Y, yaw=0.0)
    parts += stairs
    notes.append(f"staircase: {st['steps']} x {st['riser']}m riser / "
                 f"{st['tread']}m tread, flight {flight_len:.2f} m, "
                 f"rise {floor_h:.2f} m, x {STAIR_X0:.2f} to "
                 f"{STAIR_X0 + flight_len:.2f} at y {STAIR_Y:.2f}")
    notes.append(f"approach clearance west of the flight: {STAIR_X0:.2f} m")

    # --- ramp ---------------------------------------------------------------
    if with_ramp:
        ramp_len = 8.4
        ramp, slope_deg = build_ramp(x0=0.3, y0=1.55, floor_h=floor_h,
                                     length=ramp_len, width=0.8, yaw=0.0)
        parts.append(ramp)
        notes.append(f"ramp: {ramp_len} m run, {slope_deg:.1f} deg slope")

    # --- uneven terrain -----------------------------------------------------
    for i, (x, y, yaw, length, h) in enumerate(THRESHOLDS):
        parts.append(_box(f"threshold_{i}", x, y, h / 2.0,
                          length if yaw == 0 else 0.06,
                          0.06 if yaw == 0 else length,
                          h, rgba=(0.40, 0.30, 0.22, 1.0), mu=0.9))
        notes.append(f"threshold_{i}: {h*1000:.0f} mm sill at ({x}, {y})")
    for i, (x, y, sx, sy, h) in enumerate(FLOOR_PATCHES):
        parts.append(_box(f"floor_patch_{i}", x, y, h / 2.0, sx, sy, h,
                          rgba=(0.55, 0.35, 0.30, 1.0), mu=1.0))
        notes.append(f"floor_patch_{i}: {h*1000:.0f} mm raised patch at ({x}, {y})")

    # --- furniture (semantic landmarks for the VLA) -------------------------
    furniture = [
        ('kitchen_table',  2.0, 4.8, 0.0, 1.20, 0.80, 0.75, (0.55, 0.35, 0.20, 1)),
        ('kitchen_chair',  2.0, 3.9, 0.0, 0.42, 0.42, 0.90, (0.30, 0.30, 0.35, 1)),
        ('fridge',         0.5, 6.4, 0.0, 0.70, 0.70, 1.80, (0.85, 0.85, 0.88, 1)),
        ('sofa',           7.4, 3.2, 0.0, 1.90, 0.85, 0.80, (0.25, 0.40, 0.55, 1)),
        ('tv_stand',       7.4, 6.5, 0.0, 1.40, 0.40, 0.55, (0.15, 0.15, 0.18, 1)),
        ('bed',            2.0, 5.4, upper_z, 2.00, 1.50, 0.55, (0.70, 0.60, 0.75, 1)),
        ('wardrobe',       0.5, 3.2, upper_z, 0.60, 1.40, 2.00, (0.50, 0.38, 0.28, 1)),
        ('desk',           7.4, 5.8, upper_z, 1.30, 0.65, 0.75, (0.60, 0.45, 0.30, 1)),
        ('bookshelf',      8.6, 3.4, upper_z, 0.35, 1.60, 1.90, (0.45, 0.32, 0.22, 1)),
    ]
    for name, x, y, base_z, sx, sy, sz, rgba in furniture:
        parts.append(_box(name, x, y, base_z + sz / 2.0, sx, sy, sz, rgba=rgba))

    body = "\n".join(parts)

    world = f"""<?xml version="1.0" ?>
<!--
  GENERATED FILE - do not edit by hand.
  Regenerate with: python3 generate_house.py
  Source of truth for the stair geometry:
      r2d2_description/config/robot_params.yaml

{chr(10).join('  ' + n for n in notes)}
-->
<sdf version="1.9">
  <world name="house_two_floor">

    <physics name="fast" type="dart">
      <!-- 1 ms step with a stiff solver: the tri-star sub-wheels make and break
           contact rapidly during a tumble, and a coarser step lets them tunnel
           through a riser instead of climbing it. -->
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <dart>
        <solver><solver_type>dantzig</solver_type></solver>
        <collision_detector>bullet</collision_detector>
      </dart>
    </physics>

    <plugin filename="gz-sim-physics-system"
            name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system"
            name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system"
            name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system"
            name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-contact-system"
            name="gz::sim::systems::Contact"/>

    <gravity>0 0 -9.81</gravity>
    <scene>
      <ambient>0.55 0.55 0.55 1</ambient>
      <background>0.75 0.82 0.90 1</background>
      <shadows>true</shadows>
    </scene>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.3 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry>
          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry>
          <material>
            <ambient>0.62 0.60 0.57 1</ambient>
            <diffuse>0.62 0.60 0.57 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
{body}

  </world>
</sdf>
"""
    return world, notes, floor_h


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--params', default=DEFAULT_PARAMS,
                    help='robot_params.yaml providing the stair envelope')
    ap.add_argument('--out', default=os.path.join(HERE, 'house_two_floor.sdf'))
    ap.add_argument('--no-ramp', action='store_true',
                    help='omit the ramp, forcing the stair route')
    ap.add_argument('--floors-out', default=os.path.normpath(os.path.join(
        HERE, '..', '..', 'r2d2_localization', 'config', 'floors.yaml')),
        help='where to write the floor graph floor_manager reads')
    args = ap.parse_args()

    with open(args.params) as fh:
        cfg = yaml.safe_load(fh)

    world, notes, floor_h = build_world(cfg, with_ramp=not args.no_ramp)
    with open(args.out, 'w') as fh:
        fh.write(world)

    # The navigation layer needs to know where the staircase is. Emitting it
    # here, from the same constants, is what stops floor_manager aiming at a
    # flight that has since moved - the coordinates were previously typed out a
    # second time by hand, and the "foot of the stairs" pose was on the first step.
    if args.floors_out:
        _write_floors_config(args.floors_out, cfg, floor_h)
        print(f'wrote {args.floors_out}')

    reach = cfg['platform']['cluster_circumradius'] + cfg['platform']['sub_wheel_radius']
    margin = (reach - cfg['stairs']['riser']) / cfg['stairs']['riser'] * 100.0
    print(f"wrote {args.out}")
    print(f"  floor-to-floor rise : {floor_h:.2f} m")
    print(f"  cluster reach       : {reach:.3f} m")
    print(f"  climb margin        : {margin:.1f} %")
    for n in notes:
        print(f"  {n}")


def _write_floors_config(path, cfg, floor_h):
    """Write floor_manager's parameter file, derived from the world."""
    transitions = floor_transitions(cfg)
    flat = []
    for t in transitions:
        flat += [float(t['from_floor']), float(t['to_floor']),
                 t['foot']['x'], t['foot']['y'],
                 t['head']['x'], t['head']['y'], t['heading']]

    body = f'''# GENERATED FILE - do not edit by hand.
# Regenerate with: python3 src/r2d2_sim/worlds/generate_house.py
#
# The staircase coordinates below are derived from the same constants the
# Gazebo world is built from. They were previously typed out a second time in
# floor_manager.py, where the "foot of the stairs" pose had drifted onto the
# first step - close enough to look right, wrong enough that the robot would
# have started every climb already standing on the flight.

floor_manager:
  ros__parameters:
    floor_heights: [0.0, {floor_h:.3f}]
    # [from_floor, to_floor, foot_x, foot_y, head_x, head_y, heading] per
    # transition, flattened because ROS 2 parameters take no nested structures.
    transitions: {flat}
'''
    with open(path, 'w') as fh:
        fh.write(body)


if __name__ == '__main__':
    main()
