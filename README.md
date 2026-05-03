# R2-D2 Redux: An Embodied, Reasoning Robotic Assistant

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