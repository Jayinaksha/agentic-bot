# R2-D2 Redux: An Embodied, Reasoning Robotic Assistant

This project is the complete software architecture for an R2-D2-like autonomous robot that uses Large Language and Vision Models to understand and act upon natural language commands.

### Architecture Overview

The system is distributed across three main hardware components:
1.  **Robot (Raspberry Pi):** Runs ROS2 and streams sensor data.
2.  **Supercomputer (e.g., with A100 GPU):** Runs the main "brain," including the data hub, VLM, semantic map, and the LLM planner.
3.  **Laptop:** Bridges the planner's commands back to the ROS2 navigation stack.

**Data Flow:**
`Robot (Sensors)` -> `Brain Server (Hub)` -> `VLM & Semantic Map` -> `LLM Planner (Reasoning)` -> `Laptop (ROS2 Bridge)` -> `Robot (Motors & Speaker)`

### Components

This repository contains the core server-side and bridge code:
-   `gpt_oss_Archieve.py`: The main LLM planner. This is the user interface.
-   `vlm.py`: The Vision-Language Model server using Qwen2-VL.
-   `brain.py`: The central WebSocket server for ingesting all sensor data.
-   `sementic_map.py`: Builds the world model with sensor fusion.
-   `r2d2_bridge.py`: The ROS2 action bridge that runs on the laptop.
-   `r2d2_reciever.py`: The ROS2 client that runs on the robot (Raspberry Pi).

### Setup and Installation

**1. Hardware Requirements:**
-   A robot with a differential drive, LIDAR, IMU, and camera (e.g., Raspberry Pi).
-   A local laptop for the ROS2 bridge.
-   A powerful server with a high-VRAM GPU (e.g., NVIDIA A100 80GB) for the AI models.

**2. Software Requirements:**
-   Python 3.10+ with Conda/venv.
-   PyTorch, Transformers, Accelerate, Flask, WebSockets, NumPy, OpenCV.
-   ROS2 Jazzy on the robot and laptop.
-   A persistent SSH tunnel solution (e.g., `autossh`) for communication.

**3. Running the System (Order is Crucial):**

1.  **On the Robot (Raspberry Pi):**
    -   Launch the necessary ROS2 drivers for your hardware (LIDAR, camera, IMU).
    -   Launch `slam_toolbox` and `nav2`.
    -   Run the client script: `ros2 run your_package r2d2_reciever.py`

2.  **On the Laptop:**
    -   Establish the `autossh` tunnel to the supercomputer, forwarding all necessary ports (5000, 8080, 9001, 9002).
    -   Run the ROS2 bridge: `ros2 run your_package r2d2_bridge.py`

3.  **On the Supercomputer (in separate terminals):**
    -   **Terminal 1 (VLM):** `python3 vlm.py`
    -   **Terminal 2 (Brain Hub):** `python3 brain.py`
    -   **Terminal 3 (Semantic Map):** `python3 sementic_map.py`
    -   **Terminal 4 (Planner):** `CUDA_VISIBLE_DEVICES=0 python3 gpt_oss_Archieve.py`

### How to Use

Once all components are running, interact with the system via the terminal running `gpt_oss_Archieve.py`. Type natural language commands and press Enter.

**Example Commands:**
-   `what do you see?`
-   `find the chair`
-   `go to the table and tell me what is on it`