# R2D2-Redux v2: autonomous navigation of a two-storey house

This document covers the v2 stack: what it is, why each piece is built the way
it is, and — importantly — what has and has not been verified.

The v1 system (`brain.py`, `vlm.py`, `sementic_map.py`, `gpt_oss.py`,
`r2d2_bridge.py`) is still in the tree and still works. v2 does not replace it
file by file; it replaces the *approach* in four places, and those changes are
the substance of this document.

---

## 1. What changed, and why

| Area | v1 | v2 | Why |
|---|---|---|---|
| Platform | Differential drive, 2 motors | 4× tri-star wheel clusters, skid-steer | A diff-drive robot cannot climb stairs. Tri-star clusters can, while still being commanded as a differential drive — so Nav2, SLAM, the scan matcher and the ESP32 firmware survive unchanged |
| Terrain | None. A 2D LiDAR is blind to the floor | IMU pitch + 4× downward ToF, classified into flat/sill/riser/blocked/cliff | ~USD 15 and negligible CPU. A 2D LiDAR robot drives off a landing edge without ever seeing it |
| Odometry | Scan matcher **and** slam_toolbox both publishing `odom→base_link` | EKF fusing wheel + scan-match + IMU, single TF writer, regime-aware gating | Two writers to one transform is the classic silent failure. Worse, each source fails in a different, predictable place, and a fixed filter keeps fusing a source that has become fiction |
| Map | One 20×20 m grid | One map per floor, joined by explicit transition nodes | The bedroom is directly above the kitchen. A single grid fuses them into nonsense |
| Planning | One JSON blob containing an entire plan, executed open-loop | MCP tool calls, each returning the real outcome | The model never learned whether anything worked |
| Memory | SQLite; "find similar task" returned the most recent success regardless of the query | Hash-chained JetStream ledger + pgvector projection | Similarity is now an actual vector search, failures are kept, and the whole projection is rebuildable by replay |

---

## 2. The platform

### Why tri-star clusters

The requirement is a robot that climbs an ordinary residential staircase on a
low budget. The candidates from the literature:

| Design | Motors | Climbs stairs | Keeps diff-drive kinematics | Cost |
|---|---|---|---|---|
| Rocker-bogie | 6 | Poor (riser too tall) | Yes | Medium |
| RHex-style whegs | 6 | Yes | **No** — needs a phased gait | Medium |
| Tracked with flippers | 4+ | Yes | Yes | High |
| **Tri-star clusters** | **4** | **Yes** | **Yes** | **Low** |

Tri-star wins on the column that matters most here: it is still a skid-steer at
the `/cmd_vel` level. Everything above the wheels — Nav2, slam_toolbox, the
laser scan matcher, the ESP32 firmware — is unchanged. The stair capability is
bought entirely in the mechanism plus a transmission-mode switch.

### The climb envelope

A star-wheel cluster can only mount a step when its *reach* exceeds the riser:

```
reach = cluster_circumradius + sub_wheel_radius
      = 0.115 + 0.050
      = 0.165 m   vs a 0.150 m riser  →  10% margin
```

and the tread must exceed the sub-wheel diameter, or a cluster bridges two
risers instead of landing on a tread.

Both constraints live in `r2d2_description/config/robot_params.yaml`, which is
read by **both** the URDF and the Gazebo world generator. Change the cluster
radius and the staircase follows. `generate_house.py` refuses to emit a
staircase the platform cannot mount, so the robot and its test world cannot
drift out of sync.

The same discipline extends to *where* the staircase is. `generate_house.py`
owns its position and writes `r2d2_localization/config/floors.yaml` — the foot
and head poses the navigation layer plans routes against — from the same
constants. Those coordinates had previously been typed out a second time in
`floor_manager.py`, and had drifted: the "foot of the stairs" pose sat **on the
first step**, and the flight started 0.20 m from a wall, leaving a 0.36 m robot
no room to square up to it. Both were invisible until the two sources were
compared. CI now diffs the committed world and floor graph against a fresh
generation, and asserts the poses land on solid floor.

### Two transmission modes

```
ROLLING    sub-wheel joints driven, carriers held.
           Contact radius = 0.050 m. Odometry valid.

TUMBLING   carrier joints driven, sub-wheels held.
           Contact point walks around the carrier. Odometry MEANINGLESS.
```

That second line is the source of most of the complexity downstream. A tumbling
cluster has no fixed wheel-to-ground ratio, so `tristar_controller` publishes
`/odom_wheel` with a covariance of 1e6 and a `valid: false` flag rather than a
plausible-looking number the EKF would happily fuse.

### Carrier phase hold

A three-spoke carrier has **two** rest positions:

```
straddling two sub-wheels   axle at r_c·cos(60°) + r_w = 0.1075 m   (stable)
balanced on one sub-wheel   axle at r_c + r_w         = 0.1650 m   (unstable)
```

57 mm apart — and the ToF terrain monitor is trying to detect a 35 mm step. So
holding the carrier at *zero velocity* in rolling mode is not enough: it parks
wherever it stopped, the chassis rides anywhere in that band, and the robot
reads phantom stairs on a flat floor.

`carrier_hold_velocity` servos each carrier onto the straddle phase instead,
wrapping the error to the 120° symmetry period so it never turns more than 60°
to settle. `axle_height` and `tof_mount_height` are then derived from that
position rather than chosen.

This was caught by `scripts/analyse_climb.py`, not by inspection.

### Going back down

Descent is not ascent reversed, and until it was implemented the robot could go
upstairs and **never come back down** — the FSM only looked for risers, so on a
landing it reported "no riser found" and aborted, while the route planner
cheerfully emitted descend legs it could not execute.

The asymmetry is physical. Going up, the riser stops the platform and the stall
is an unambiguous *you are here*. Going down there is no such event: the ToF
beams see the drop while the wheels are still on solid floor, and a robot that
keeps rolling drives off the top step. So the last stretch is dead-reckoned from
geometry:

```
creep = (tof_forward_offset + spot_ahead) − wheelbase/2 − margin
      = (0.180 + 0.290) − 0.130 − 0.05
      = 0.290 m
```

Every term is measurable with a tape on the real robot. `EdgeApproach` waits for
the drop to be consistently visible, then measures exactly that far on wheel
odometry before committing to a tumble — and discards the measurement entirely
if the cliff stops being visible, because committing on a stale reading is how a
robot falls down a flight of stairs. Body pitch is deliberately *not* a fallback
here: by the time the chassis pitches over an edge it is already committed.

The geometry is checked at startup. A platform whose beams land behind its front
contact patch gets no warning at all before the edge, and descent is refused
outright.

**Descent is off by default** (`allow_descent`). It has not been validated on
hardware or in simulation, and the failure mode is the robot falling downstairs.
With it off, the FSM refuses explicitly and the MCP layer checks the whole route
*before* setting off, so you learn about it in the kitchen rather than on the
landing.

### Detecting contact with a riser

The phase hold has a consequence that is easy to miss and fatal if missed. With
the carriers locked, a sub-wheel meeting a riser face **stalls** the robot — the
chassis does not tip. So the obvious mount trigger, `pitch > threshold`, never
fires: the approach drives at the step for its full timeout and aborts having
never started the climb. The platform would not have climbed a single step.

The signal that does work is the stall itself. Wheel odometry is valid in
rolling mode, so `RiserContact` compares achieved travel against commanded
travel and declares contact when, for 0.6 s continuously:

- both front ToF beams see a climbable riser, **and**
- achieved travel falls below 35% of commanded travel

Both conditions are needed in each direction. Stall alone fires on a chair leg
or a rug. Riser-ahead alone fires while the robot is still a lookahead away, and
tumbling in free space walks the platform forward on its cluster corners instead
of driving. Pitch remains as a secondary trigger for shallow nosings the robot
does partly ride up, but it is never the only one.

---

## 3. Terrain sensing without a depth camera

Four VL53L0X-class ToF beams, mounted at the chassis top corners, angled 30°
below horizontal:

```
flat return  = h / sin(tilt) = 0.180 / sin(30°) = 0.360 m
ground spot  = h / tan(tilt) = 0.312 m ahead of the sensor
```

A step up shortens the return; a drop lengthens it. The implied floor-height
change is

```
dz = h − r · sin(tilt + pitch)
```

**The pitch term is not optional.** On the 12.1° ramp in the test house the
robot pitches nose-up, lengthening every return. Without correction that reads
as a continuous cliff and the robot refuses to use the ramp at all. There is a
regression test for exactly this
(`test_ramp_does_not_produce_phantom_steps`).

Classification, all thresholds in `locomotion.yaml`:

| `dz` | Class | Behaviour |
|---|---|---|
| < −0.060 | `cliff` | Hazard → costmap, refuse approach |
| −0.060 … 0.035 | `flat` | Drive over (door sills are 18–30 mm) |
| 0.035 … 0.150 | `climbable_riser` | **Not** a costmap obstacle — this is the route upstairs |
| > 0.150 | `blocked` | Hazard → costmap |

That third row matters: marking a riser as an obstacle would make Nav2 plan
around the one feature the platform was built to use.

---

## 4. Odometry, and knowing when not to trust it

Three sources, three different failure modes:

| Source | Good at | Fails when |
|---|---|---|
| Wheel odometry | Short-term scale | Any cluster tumbles |
| PLICP scan match | Heading, drift-free over a room | Stairs (scanner points at the ceiling); bare corridors (no along-axis features) |
| IMU | Always available | Always drifting |

`robot_localization` cannot be reconfigured at runtime, so `odom_supervisor`
applies the regime where it actually can be — in the covariances the filter
consumes:

```
scan matcher ──> /odom_scan_raw ──> [odom_supervisor] ──> /odom_scan ──> EKF
                                     inflates covariance
                                     when degenerate or climbing
```

Corridor degeneracy is detected from the scan itself: bin returns by angle,
weight by 1/r (near structure constrains a match far more than a distant wall),
and if nearly all the structure is perpendicular to one axis, the along-axis
estimate is unreliable.

The supervisor publishes `/odometry/health` carrying
`precise_goals_allowed`. The MCP layer honours it: `dock_precisely` refuses
when the pose underneath it is not precise. **Saying "I do not know where I am
well enough for that" is a feature.**

### Transform ownership

```
map  → odom        slam_toolbox (mapping) or amcl (localising)
odom → base_link   ekf_filter_node, and nothing else
```

The v1 `start_mapping.launch.py` had the scan matcher *and* slam_toolbox both
writing into this chain. The scan matcher now runs `publish_tf:=false`.

---

## 5. Multi-floor

Floor is **not** read from the EKF. The filter runs in `two_d_mode`, so its `z`
is identically zero, and turning that off just to get a floor index buys a
badly drifting height from double-integrated accelerometer noise.

Instead the floor is a discrete counter advanced by completed climb events.
`climb_fsm` reports each flight's IMU-integrated rise; a flight that gained
roughly one storey moves the counter one floor. Floors only change when the
robot climbs, so an event counter is both cheaper and more reliable than a
continuous estimate.

On arrival, `floor_manager`:
1. loads that floor's map into `nav2_map_server`,
2. clears both costmaps,
3. **re-seeds AMCL at the head of the flight** — the filter has been dead
   reckoning on an IMU for the whole climb, and the one thing we know for
   certain is that the robot is standing at the top of the staircase it just
   climbed.

If a floor has never been mapped, the manager refuses rather than serving an
empty grid. An empty grid and "the robot is lost" look identical to a planner
and very different to a person.

---

## 6. Precision

Nav2 on a skid-steer reliably delivers ±20 cm. Tightening the goal checker does
not help, because the error is in the global pose itself, not the checker.

`precise_docking` stops using the global frame for the last stretch. It servos
on the **live laser scan** against a local geometric feature, so map error drops
out of the loop entirely:

- **surface** — total-least-squares line fit, then drive to a standoff normal
  to it. TLS rather than OLS because a robot squared up to a flat surface sees
  points that all share an x coordinate, which is precisely the case where
  ordinary least squares has infinite slope.
- **gap** — find the two edges of an opening and centre on it. This is what
  makes a 0.90 m doorway passable without clipping the frame.

Tolerance ±2 cm, ±2°.

---

## 7. Memory: ledger and projection

```
   ROS topics ──> memory_node ──> NATS JetStream (hash-chained, append-only)
                                        │
                                        ├──> projector ──> Postgres + pgvector
                                        │                   (rebuildable)
   agent tool calls ────────────────────┘
```

**The ledger is the source of truth.** Each record carries the SHA-256 of its
predecessor, so altering or deleting a past event breaks every hash after it and
`--audit` reports exactly where. Re-sealing a tampered record fixes its own hash
but not its successor's `prev_hash` — the *chain*, not the hash, is the
protection.

**Postgres is a projection.** Drop it, replay the stream, get the same state
back. That is what makes it safe to change the object-fusion rules and reproject
rather than living with whatever was written the first time.

### Object fusion

A VLA looking at one chair from four angles produces four detections. Storing
four chairs makes the map useless; merging everything with the same label makes
two dining chairs into one. The rule: a detection joins an existing object when
it shares a label and floor and lies within a merge radius that **shrinks as the
object becomes better observed**.

```
radius = clamp(0.90 / sqrt(observations), 0.25, 0.90)
```

Positional spread is retained as `position_sigma`. A large sigma is the signal
that a merge was probably wrong.

### Episodic recall

v1's `find_similar_task` returned the most recent successful task regardless of
what was asked — so the planner was shown an unrelated example and encouraged to
imitate it. v2 does a real vector search and **keeps failures**. Knowing that
the last attempt at this errand ended because the study door was shut is worth
more than a success at a different task.

### Degrading gracefully

Every memory dependency is optional at runtime. Without `nats-py` the ledger
becomes a no-op; without `sentence-transformers` the embedder falls back to
deterministic hashing. Losing memory costs the robot its recall, not its ability
to drive.

---

## 8. Perception

NVIDIA **Cosmos Reason 2** behind an OpenAI-compatible endpoint. It returns
object localisation — 2D boxes with labels — alongside its reasoning, which is
exactly what the grounding step needs.

With no depth camera, grounding is bearing-plus-range:

1. Box centre → bearing, via the pinhole model (tangent form; a linear map is
   several degrees out at the edge of a wide lens).
2. Bearing → LiDAR range, via the **median** of a wedge. Not the mean: a wedge
   clipping an open doorway picks up a few 9 m returns from the next room, and a
   mean is dragged more than a metre by them.
3. → world coordinates.

Every estimate reports its own `grounding_quality`:

- `good` — LiDAR agreed
- `coarse` — ranged, but returns disagreed
- `bearing_only` — the object is off the scan plane (a light switch, a mug on a
  table), so only its direction is known

An object that cannot be ranged is still stored. "There is a light switch on
this wall" is useful even without a coordinate.

Frames are gated on movement *and* a time budget: a parked robot costs nothing.

---

## 9. The MCP layer

### Why tools instead of JSON

```
v1:  prompt → one JSON blob → repair_json() → execute open-loop
v2:  prompt → tool call → real outcome → next tool call → ...
```

Three concrete problems solved:

1. **The model never learned whether anything worked.** The v1 plan was emitted
   before the first wheel turned. Now `navigate_to_room` returns
   `{"success": false, "reason": "a step too tall for the wheel clusters to
   mount is in the way"}` and the model can do something about it.

2. **Parsing was a guess.** `repair_json` turning a malformed brace into a valid
   plan means executing something the model may not have meant. Tool calls are
   schema-validated before anything moves; a response that does not parse is
   read as text, never repaired.

3. **The prompt carried the whole capability surface** and grew with every
   feature. MCP tools are described and versioned by the server, so what the
   robot can do and what the model believes it can do cannot drift apart.

### Safety invariants the server enforces

Regardless of what is asked:
- a precise dock is refused while localisation quality says otherwise
- a climb is refused unless the robot faces a mountable riser
- a cross-floor goal is planned as a route, never sent as one Nav2 goal into the
  wrong floor's map

### Where things run

```
Local machine                          Cloud GPU (Nebius / GCP / NVIDIA catalogue)
├── Gazebo + ROS 2 + Nav2              ├── Cosmos Reason 2   (VLA)
├── MCP server  (robot as tools)       └── Nemotron          (planner)
├── MCP agent   (tool loop)                    ▲
├── NATS + Postgres                            │ outbound HTTPS only
└── ───────────────────────────────────────────┘
```

Only outbound HTTPS leaves the machine. Nothing is exposed inbound, so no
tunnel is needed — a real simplification over the v1 autossh arrangement.

---

## 10. Verification status

**Be aware of this section before trusting anything above.**

### Verified here

- **255 unit tests**, all passing, none requiring ROS/Gazebo/NATS/Postgres:

  | Suite | Tests | Covers |
  |---|---|---|
  | `r2d2_locomotion` | 71 | Skid-steer kinematics, mode-dependent joint commands, carrier phase hold, riser contact detection, arc odometry, ToF geometry, climb envelope |
  | `r2d2_navigation` | 49 | TLS line fitting, doorway detection, docking sign conventions, cross-floor routing, descent marking |
  | `r2d2_memory` | 28 | Hash-chain tamper detection, merge radius, position fusion, embeddings |
  | `r2d2_perception` | 44 | Pixel→bearing, median ranging, world projection, VLA response parsing |
  | `r2d2_mcp` | 42 | Tool-call parsing, agent loop, failure paths, dry run |
  | `r2d2_sim` | 21 | Wall-gap splitting, world geometry, and a flood-fill proving every room is actually reachable |

- `scripts/analyse_climb.py` — quasi-static analysis of reach, tread fit, gait
  match, tipping, torque, climb duration and ride height. **This found two real
  bugs**: the zero-velocity carrier hold, and — following from it — a mount
  trigger that could never have fired. The same analytical pass found a third:
  descent was entirely unimplemented while the route planner emitted descend
  legs. All three are described above. Two warnings stand
  deliberately (see below).
- All Python compiles; all XML/YAML/SDF parses; every `setup.py` entry point
  resolves to a real module; every referenced script exists.
- Topic-wiring audit confirms `/cmd_vel` and `/locomotion/mode` each have
  exactly one writer.
- `generate_house.py` output matches the committed world byte for byte.
- `scripts/check_params.py` — three checks in one. It cross-checks every node's
  `declare_parameters` against the YAML section that configures it, in both
  directions (a misspelled ROS 2 parameter fails silently, so this is the only
  place it can be caught); it compares the 12 constants that are necessarily
  duplicated between `robot_params.yaml` and the node parameter files; and it
  verifies that derived values still follow from the platform —
  `tof_mount_height` from the axle height, `tof_spot_ahead` from the mount
  geometry, `axle_height` from the carrier straddle position. Two of the bugs
  below were duplicated constants drifting apart, so this is now enforced.
- `scripts/check_clearances.py` — checks the robot actually fits the house:
  footprint radius against the body diagonal, doorway lane width, that margin
  against realistic path-following pose error, whether a zero-cost lane survives
  the costmap inflation, and hallway turning circle. It found that a 0.45 m
  inflation radius met in the middle of every 0.90 m doorway.
- `scripts/check_ci.py` — runs the whole workflow locally, so a broken CI step
  is found before a push rather than after one.
- All of the above runs in CI (`.github/workflows/tests.yml`) on every push,
  across Python 3.10 (Humble) and 3.12 (Jazzy).

### Bugs this tooling found

None of these were visible by reading the code; each needed two sources
compared, or a number computed.

| Bug | Consequence | Found by |
|---|---|---|
| Carriers held at zero *velocity*, not phase | Chassis rides anywhere in a 57 mm band, past the 35 mm ToF step threshold → phantom stairs on flat floor | `analyse_climb.py`, ride height |
| Mount trigger waited on body pitch | Phase-locked carriers cannot tip; approach times out → **never climbs a single step** | Following the fix above |
| Descent unimplemented | Robot goes upstairs and never comes back down, while the route planner emits descend legs | Reading the FSM against the route planner |
| "Foot of stairs" pose on the first step | Climb starts already standing on the flight | Generated vs. hand-typed coordinates |
| 0.20 m of approach clearance | A 0.36 m robot cannot square up to the flight | Same comparison |
| Contact detector 4× slow | Would have missed its own confirmation window | Tracing the detector by hand |
| Inflation covered the whole doorway | No zero-cost lane through any door in the house: the controller crawls at every threshold and the planner detours around doors | `check_clearances.py` |
| A doorway placed off its wall lengthened it | A door mistyped at x=99 produced a 98 m wall across the map instead of a 9 m one, silently | Falsifying the reachability test |

### Known warnings, accepted deliberately

| Check | Finding | Why it stands |
|---|---|---|
| Gait match | One 120° tumble advances 0.199 m against a 0.318 m step pitch (ratio 0.63) | Matching exactly needs `cluster_circumradius` = 0.183 m, which costs 60% more peak torque and a much taller robot. See the note below on what this ratio does and does not tell you |
| Torque | Peak 4.7 N·m per cluster when two clusters carry the lift | A real BOM constraint, now recorded as `min_cluster_torque` in `robot_params.yaml`. A typical hobby gearmotor (2–4 N·m) will stall on the first riser |

### NOT verified

Nothing in this container has ROS 2, Gazebo, NATS, Postgres or a GPU, so:

- **The robot has never been simulated.** The tri-star climb is the biggest
  open question: contact-rich tumbling is exactly what simulators get wrong.
  Expect to tune `max_step_size`, friction and the solver before it climbs
  cleanly. The physics block is already set for a 1 ms step with a stiff
  solver for this reason.
- **No launch file has been executed.** Node names, remappings and parameter
  plumbing are written carefully but not run.
- **Nav2 parameters are unvalidated** against your Nav2 version. Plugin names
  changed between Humble and Jazzy.
- **No model endpoint has been called.** The prompt and response handling are
  written against Cosmos Reason 2's documented output shape.
- **The Postgres schema has never been applied**, and the ivfflat `lists=100`
  is a starting guess.

### Falsifying the tests, not just the code

The abandoned simulator taught a habit worth keeping: a check that passes tells
you nothing until you have watched it fail. Every check added since has been
falsified deliberately before being trusted — drift injected into each mirrored
constant, the inflation radius pushed back to a value that should warn, doors
narrowed below the robot, walls deleted.

That discipline caught a mistake of mine immediately. Sealing the kitchen's
hallway door did **not** make the reachability test fail, which looked like a
broken test. It was a broken falsification: the kitchen has two doors, so the
route through the living room survived. Sealing both fires the assertion
correctly. The same exercise turned up a real bug in the wall splitter, since a
gap placed off its wall was extending the wall to reach it.

### An attempt that was abandoned, and why

The gait ratio of 0.63 is not a verdict. It is equally consistent with a cluster
that walks the flight awkwardly and one that wedges against the second riser and
stops, and the difference decides whether the platform works at all. So I tried
to settle it with a kinematic simulation: roll a rigid tri-star up an exact stair
profile and see what happens.

It was written twice and failed falsification both times. Before trusting any
result, the simulator was asked to climb risers well beyond the cluster's
0.165 m reach — a 0.30 m step, then a 0.50 m wall. Both versions cheerfully
reported success:

```
riser 0.150 (designed)      -> CLIMBS
riser 0.300 (far beyond)    -> CLIMBS     <- must be impossible
riser 0.500 (a wall)        -> CLIMBS     <- must be impossible
```

The first version resolved wheel contacts vertically only, so a wheel driven
into a riser face was simply lifted over it. The second added a pivot model but
resolved penetration by lifting the cluster out of it, which is the same bug
wearing a different hat. Both errors happened to flatter the design, which is
the signature of a model with an escape hatch in it.

Getting this right means writing a small rigid-body contact solver, and two
wrong attempts is poor evidence that a third would be right. **A subtly wrong
simulator reporting "it climbs" is considerably worse than no simulator**, since
it would give false confidence about the single riskiest part of the project.
So the code was deleted rather than committed.

The honest position is that the reach and tread checks are sound and necessary
but not sufficient, and whether the cluster tracks this particular staircase is
a question for Gazebo — which is the right tool, and which you are going to run
anyway. If you do revisit it, the falsification test above is the bar: a
simulator that cannot refuse a 0.50 m wall is not measuring anything.

### Suggested bring-up order

0. `python3 scripts/analyse_climb.py` — does the geometry still close? Run this
   first after changing any dimension; it is seconds, and it catches things a
   simulator will only show you as an unexplained failure to climb.
1. `ros2 launch r2d2_description display.launch.py` — is the robot the right
   shape?
2. `ros2 launch r2d2_sim sim.launch.py` — does it spawn and settle on its
   clusters?
3. Teleop on flat ground — does `/cmd_vel` drive it sensibly? Then
   `python3 scripts/calibrate_slip.py` to measure `yaw_slip_factor` rather than
   trusting the 1.25 estimate. Check the chassis holds a steady 0.1075 m: if it
   bobs, the carrier hold is not working and the ToF thresholds are void.
4. Drive at the staircase in tumbling mode — **this is where the tuning is**.
5. `slam.launch.py floor:=0`, save, repeat for floor 1.
6. `house_stack.launch.py mode:=amcl` — Nav2 on one floor.
7. `docker compose up -d`, then `python3 scripts/check_memory.py` to verify the
   ledger and pgvector store against real backends before wiring perception in.
8. `python3 -m r2d2_mcp.agent --dry-run "go to the kitchen"` — check the tool
   loop before letting it move anything.
