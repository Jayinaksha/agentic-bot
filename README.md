# R2-D2 Redux: An Embodied, Reasoning Robotic Assistant

Two stacks live in this repository.

**v2 (current)** — a tri-star stair-climbing rover that navigates a two-storey
house in Gazebo, with MCP tool calling in place of one-shot JSON planning, NVIDIA
open models for reasoning and vision, and an event-sourced memory on NATS
JetStream projected into Postgres/pgvector.
See **[docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md)**.

**v1 (original)** — the distributed robot/supercomputer/laptop system described
further down. Still present and still works; v2 does not delete it.

---

## v2 quick start

```bash
# 1. build
cd ~/agentic-bot && colcon build --symlink-install && source install/setup.bash

# 2. drive it around a two-storey house
ros2 launch r2d2_navigation house_stack.launch.py mode:=slam

# 3. map each floor, then navigate
ros2 run nav2_map_server map_saver_cli -f ~/r2d2_maps/house_f0
ros2 launch r2d2_navigation house_stack.launch.py mode:=amcl

# 4. memory backends
docker compose -f src/r2d2_memory/docker-compose.yml up -d

# 5. perception + memory bridge
export R2D2_NVIDIA_API_KEY=nvapi-...
ros2 launch r2d2_perception perception.launch.py

# 6. talk to it
python3 -m r2d2_mcp.agent "go upstairs and tell me what is on the desk"
python3 -m r2d2_mcp.agent --dry-run "..."      # plan without moving
```

### v2 packages

| Package | What it does |
|---|---|
| `r2d2_description` | Tri-star cluster rover: 4 clusters, 2D LiDAR, IMU, 4 ToF beams, camera, anti-tip tail |
| `r2d2_sim` | Parametric two-storey house world (rooms, stairs, ramp, sills) + Gazebo bring-up |
| `r2d2_locomotion` | `/cmd_vel` → 16 joints, rolling/tumbling transmission, terrain classification, stair-climb FSM |
| `r2d2_localization` | EKF fusing wheel + scan-match + IMU, regime gating, per-floor map management |
| `r2d2_navigation` | Nav2 (RPP + Smac 2D), cross-floor route planning, centimetre docking servo |
| `r2d2_perception` | Cosmos Reason 2 detections grounded into map coordinates via the LiDAR |
| `r2d2_memory` | Hash-chained JetStream ledger + pgvector projection |
| `r2d2_mcp` | The robot as MCP tools, and the agent that drives them |

### Sensors, and their cost

| Sensor | Count | Approx. cost | Used for |
|---|---|---|---|
| 2D LiDAR (RPLIDAR A1 class) | 1 | $99 | SLAM, scan matching, obstacles, ranging detections |
| 6-axis IMU (MPU6050/BNO085) | 1 | $5–25 | Attitude, climb detection, EKF, dead reckoning |
| Downward ToF (VL53L0X) | 4 | $12 | Steps, cliffs, stair edges |
| RGB camera | 1 | $15 | VLA grounding, semantic map |

No depth camera and no 3D LiDAR. Terrain understanding comes from fusing IMU
attitude with the four ToF beams — roughly 1/20th the cost and a small fraction
of the CPU of an RGB-D traversability pipeline.

### Configuration

```bash
# Planner and VLA. Both are OpenAI-compatible, so either the NVIDIA-hosted
# catalogue or your own NIM/vLLM on a cloud GPU works with the same code.
export R2D2_NVIDIA_API_KEY=nvapi-...
export R2D2_LLM_BASE_URL=https://integrate.api.nvidia.com/v1
export R2D2_LLM_MODEL=nvidia/nemotron-nano-12b-v2-vl
export R2D2_VLA_MODEL=nvidia/cosmos-reason2-8b

# Self-hosted instead:
# export R2D2_LLM_BASE_URL=http://10.0.0.5:8000/v1

# Memory (optional; the robot drives without it)
export R2D2_NATS_URL=nats://127.0.0.1:4222
export R2D2_PG_DSN=postgresql://r2d2:r2d2@127.0.0.1:5432/r2d2
export R2D2_MEMORY=on
```

### Tests and design checks

233 unit tests covering the geometry and control logic, none of which need ROS,
Gazebo, NATS, Postgres or a model endpoint:

```bash
python3 -m pytest src/r2d2_locomotion/test src/r2d2_navigation/test \
                  src/r2d2_memory/test src/r2d2_perception/test \
                  src/r2d2_mcp/test -q
```

Three helper scripts, all of which are worth running before the simulator:

```bash
python3 scripts/analyse_climb.py     # can this platform climb these stairs?
python3 scripts/calibrate_slip.py    # measure yaw_slip_factor (needs ROS)
python3 scripts/check_memory.py      # ledger + pgvector against real backends
```

`analyse_climb.py` settles reach, tread fit, gait match, tipping, torque and
ride height on paper, on the same `robot_params.yaml` the URDF and world
generator read. It has already caught two shipped bugs: a zero-velocity carrier
hold that let the chassis ride anywhere in a 57 mm band (well past the 35 mm
step the terrain monitor looks for), and — following from the fix — a
stair-mount trigger that waited for a body pitch that phase-locked carriers can
never produce, so the robot would have aborted every climb. The same pass found
that descent was unimplemented: the robot could go upstairs and never come back
down. Descent now exists but ships **off by default**, since it is
dead-reckoned over the last few centimetres and untested. Tests and checks
both run in CI on every push.

> **The v2 stack has not been run on hardware or in simulation.** The maths and
> control logic are unit-tested; the ROS graph, Gazebo physics and Nav2
> parameters are not. Start with the bring-up order in
> [docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md#suggested-bring-up-order).

---

# v1: the original distributed system

This project is the complete software architecture for an R2-D2-like autonomous robot that uses Large Language and Vision Models to understand and act upon natural language commands.

### Architecture Overview

The system is distributed across three main hardware components:

1.  **Robot (ESP32 + Raspberry Pi):** An ESP32 microcontroller runs micro-ROS firmware to drive the motors. A Raspberry Pi runs ROS2, streams all sensor data (camera, LIDAR, IMU) to the supercomputer brain via WebSocket.
2.  **Supercomputer (GPU server, e.g., IIT Patna Param Rudra HPC):** Runs the main "brain," including the data hub, VLM, semantic map, and the LLM planner.
3.  **Laptop:** Maintains an SSH tunnel to the supercomputer and bridges the planner's navigation commands to ROS2 Nav2.

**Data Flow:**
```
Robot (Sensors) ──> Brain Server (Hub) ──> VLM & Semantic Map ──> LLM Planner (Reasoning)
                                                                          │
Robot (Motors & Speaker) <── Laptop (ROS2 Nav2 Bridge) <────────────────┘
```

### Repository Structure

```
agentic-bot/
├── src/
│   ├── r2d2_brain/                  # Supercomputer-side AI components
│   │   ├── brain.py                 # Central WebSocket hub for all sensor data
│   │   ├── vlm.py                   # Vision-Language Model server (Qwen2-VL-7B-Instruct)
│   │   ├── sementic_map.py          # Semantic map server (LIDAR + VLM fusion)
│   │   ├── gpt_oss.py               # Main LLM planner v3.0 (GPT-OSS 20B)
│   │   ├── gpt oss_Archieve.py      # Archived LLM planner v2.0
│   │   └── view_map.py              # Utility to visualize saved semantic maps
│   ├── r2d2_comms/
│   │   └── r2d2_comms/              # ROS2 package
│   │       ├── r2d2_bridge.py       # Robot-side: streams sensor data to brain via WebSocket
│   │       ├── r2d2_reciever.py     # Laptop-side: ROS2 Nav2 bridge, exposes HTTP API
│   │       └── local_image_sender.py # Sends camera frames to VLM via SSH tunnel
│   └── command_client.py            # Interactive terminal UI for sending commands
├── esp32_drive/
│   └── esp32_drive.ino              # ESP32 micro-ROS differential drive firmware
├── god_led_suscriber/
│   └── god_led_suscriber.ino        # ESP32 micro-ROS LED test subscriber
├── start_brain_tunnel.sh            # autossh tunnel script (robot/laptop → supercomputer)
└── warehouse_bridge.yaml            # Gazebo simulation ↔ ROS2 bridge configuration
```

### Component Details

| Component | File | Runs On | Port(s) |
|---|---|---|---|
| Brain Hub | `src/r2d2_brain/brain.py` | Supercomputer | 9001 (sensors), 9002 (semantic map), 9003–9004 (control) |
| VLM Server | `src/r2d2_brain/vlm.py` | Supercomputer | 5000 |
| Semantic Map | `src/r2d2_brain/sementic_map.py` | Supercomputer | 8080 |
| LLM Planner | `src/r2d2_brain/gpt_oss.py` | Supercomputer | 7000 (WebSocket) |
| Sensor Bridge | `src/r2d2_comms/r2d2_comms/r2d2_bridge.py` | Robot (RPi) | — |
| Nav2 Bridge | `src/r2d2_comms/r2d2_comms/r2d2_reciever.py` | Laptop | 8888 (HTTP) |
| Image Sender | `src/r2d2_comms/r2d2_comms/local_image_sender.py` | Robot (RPi) | — |
| Command Client | `src/command_client.py` | Any | — |
| Drive Firmware | `esp32_drive/esp32_drive.ino` | ESP32 | — |

### Setup and Installation

**1. Hardware Requirements:**
-   A differential-drive robot with an ESP32 (motor controller), LIDAR, IMU, and camera (connected to Raspberry Pi).
-   A laptop for the SSH tunnel and ROS2 Nav2 bridge.
-   A powerful server with a high-VRAM GPU (e.g., NVIDIA A100 80GB) for the AI models.

**2. Software Requirements:**
-   Python 3.10+ with Conda/venv.
-   PyTorch, Transformers, Accelerate, Flask, WebSockets, NumPy, OpenCV.
-   ROS2 Jazzy on the robot (Raspberry Pi) and laptop.
-   `autossh` on the laptop for the persistent SSH tunnel.
-   Arduino IDE with micro-ROS library for the ESP32 firmware.

**3. Flash ESP32 Firmware:**

Flash `esp32_drive/esp32_drive.ino` to the ESP32 using the Arduino IDE. This firmware:
-   Subscribes to the `/cmd_vel` topic (geometry_msgs/TwistStamped) via micro-ROS over serial.
-   Converts linear/angular velocities to PWM values for a two-motor differential drive.
-   Tune `MAX_LINEAR_VEL` and `MAX_ANGULAR_VEL` in the sketch to match your robot's characteristics.

**4. Running the System (Order is Crucial):**

**Step 1 — On the Supercomputer** (in separate terminals):
```bash
# Terminal 1: VLM server (Qwen2-VL-7B-Instruct, port 5000)
python3 src/r2d2_brain/vlm.py

# Terminal 2: Brain WebSocket hub (ports 9001–9004)
python3 src/r2d2_brain/brain.py

# Terminal 3: Semantic map server (port 8080)
python3 src/r2d2_brain/sementic_map.py

# Terminal 4: LLM planner — GPT-OSS 20B (port 7000)
CUDA_VISIBLE_DEVICES=0 python3 src/r2d2_brain/gpt_oss.py
```

**Step 2 — On the Laptop:**
```bash
# Establish the autossh tunnel to the supercomputer
# Forwards ports 5000, 8080, 9001–9004 to the GPU compute node
bash start_brain_tunnel.sh

# In a separate terminal: run the ROS2 Nav2 bridge (port 8888)
ros2 run r2d2_comms r2d2_reciever
```

**Step 3 — On the Robot (Raspberry Pi):**
```bash
# Launch ROS2 drivers for your hardware (LIDAR, camera, IMU)
# Launch slam_toolbox and nav2

# Stream sensor data to the brain
ros2 run r2d2_comms r2d2_bridge
```

### How to Use

Interact with the system by running the command client on any machine that can reach the planner (port 7000):
```bash
python3 src/command_client.py
```

You can also type commands directly in the terminal running `gpt_oss.py`. Built-in commands:

| Command | Description |
|---|---|
| `observe` | Use the VLM to analyze what the robot currently sees |
| `context` | Show robot position, semantic map objects, and connection status |
| `locate <object>` | Estimate distance and position of a named object using LIDAR |
| `speak <text>` | Send a speech command to the robot |
| `scan_debug` | Show detailed LIDAR scan data |
| `ros2_status` | Check the ROS2 Nav2 bridge connection |
| `history` | Show previously successful navigation tasks |
| `help` | List all available commands |
| `exit` / `quit` | Exit the planner |

Any other input is treated as a natural language navigation instruction.

**Example Instructions:**
-   `what do you see?`
-   `find the chair`
-   `go to the table and tell me what is on it`

### Gazebo Simulation

To run with a simulated robot (Gazebo), use `warehouse_bridge.yaml` to bridge the simulator topics to ROS2:
```bash
ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=warehouse_bridge.yaml
```
This bridges LIDAR (`/scan`), camera (`/camera/image_raw`), IMU (`/imu/data`), and the clock from Gazebo to ROS2.