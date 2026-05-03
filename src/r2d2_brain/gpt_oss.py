#!/usr/bin/env python3
"""
R2D2 GPT-OSS Navigation Planner (v3.0)

This planner uses GPT-OSS 20B to generate high-level navigation plans.
It directly queries the VLM server for visual perception and outputs
navigation goals (x,y coordinates) and speech for the robot.

Features:
- Direct ROS2 integration via bridge server
- Object distance estimation using lidar data
- Robust JSON parsing and error handling
- Real-time navigation commands to robot
"""

import json
import logging
import time
import os
import requests
import torch
import sqlite3
import glob
import re
import math
import asyncio
import websockets
import threading
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any, Optional, Tuple, Union

# --- Configuration & Logging ---
BRAIN_SERVER_URL = "ws://localhost:9001"  # Brain server
SEMANTIC_MAP_URL = "http://localhost:8080/map"  # Semantic map server REST API
VLM_SERVER_URL = "http://localhost:5000/process_frame"  # VLM server
ROS2_BRIDGE_URL = "http://localhost:8888"  # ROS2 bridge server on laptop
GPT_MODEL_NAME = "openai/gpt-oss-20b"
GPU_DEVICE = "cuda:0"
MAX_NEW_TOKENS = 1024
TASK_DB_PATH = "navigation_tasks.sqlite"
SENSOR_LOG_DIR = "data"

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/gpt_oss_nav_planner.log")
    ]
)
log = logging.getLogger("nav_planner")

# --- Global State ---
model, tokenizer = None, None
last_observation = {}
last_camera_frame = None
last_lidar_data = None
brain_ws_connection = None
ros2_bridge_connected = False
sensor_data = {
    "camera": None,
    "lidar": None,
    "robot_position": {"x": 0.0, "y": 0.0, "theta": 0.0}
}
running = True

# --- ROS2 Bridge Communication ---
def setup_ssh_tunnel():
    """Setup SSH tunnel to laptop if needed"""
    try:
        # Check if tunnel already exists
        result = subprocess.run(
            ["ss", "-tln", "sport = :8888"], 
            capture_output=True, 
            text=True
        )
        
        if "8888" in result.stdout:
            log.info("SSH tunnel to laptop already exists")
            return True
            
        # Set up SSH tunnel
        log.info("Setting up SSH tunnel to laptop for ROS2 bridge...")
        # This runs in background and forwards supercomputer:8888 to laptop:8888
        # Replace user and laptop-ip with actual values
        subprocess.Popen(
            ["ssh", "-f", "-N", "-R", "8888:localhost:8888", "user@laptop-ip"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        log.info("SSH tunnel established")
        return True
    except Exception as e:
        log.error(f"Failed to setup SSH tunnel: {e}")
        return False

def check_ros2_bridge():
    """Check if ROS2 bridge is available"""
    global ros2_bridge_connected
    
    try:
        response = requests.get(f"{ROS2_BRIDGE_URL}/status", timeout=2)
        if response.status_code == 200:
            ros2_bridge_connected = True
            log.info("ROS2 bridge connected and available")
            return True
        else:
            ros2_bridge_connected = False
            log.warning(f"ROS2 bridge returned status code {response.status_code}")
            return False
    except requests.RequestException as e:
        ros2_bridge_connected = False
        log.warning(f"ROS2 bridge not available: {e}")
        return False

def send_navigation_goal(x: float, y: float, description: str, theta: float = 0.0) -> bool:
    """Send navigation goal to ROS2 bridge on laptop"""
    global ros2_bridge_connected
    
    # Check if bridge is connected
    if not ros2_bridge_connected and not check_ros2_bridge():
        log.warning("Cannot send navigation goal: ROS2 bridge not connected")
        return False
    
    try:
        log.info(f"Sending navigation goal to ROS2: ({x}, {y}, {theta}) - {description}")
        response = requests.post(
            f"{ROS2_BRIDGE_URL}/navigate",
            json={
                "x": x,
                "y": y,
                "theta": theta,
                "description": description
            },
            timeout=5
        )
        
        if response.status_code == 200:
            log.info(f"Successfully sent navigation goal to ROS2")
            return True
        else:
            log.error(f"Failed to send navigation goal: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        log.error(f"Error sending navigation goal: {e}")
        ros2_bridge_connected = False
        return False

def send_speech_command(text: str) -> bool:
    """Send speech command to ROS2 bridge on laptop"""
    global ros2_bridge_connected
    
    # Check if bridge is connected
    if not ros2_bridge_connected and not check_ros2_bridge():
        log.warning("Cannot send speech command: ROS2 bridge not connected")
        return False
    
    try:
        log.info(f"Sending speech command to ROS2: \"{text}\"")
        response = requests.post(
            f"{ROS2_BRIDGE_URL}/speak",
            json={"text": text},
            timeout=5
        )
        
        if response.status_code == 200:
            log.info(f"Successfully sent speech command to ROS2")
            return True
        else:
            log.error(f"Failed to send speech command: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        log.error(f"Error sending speech command: {e}")
        ros2_bridge_connected = False
        return False

# --- Database Management ---
def init_database():
    """Initialize the task database"""
    os.makedirs(os.path.dirname(TASK_DB_PATH) if os.path.dirname(TASK_DB_PATH) else '.', exist_ok=True)
    with sqlite3.connect(TASK_DB_PATH) as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS navigation_tasks (
            id TEXT PRIMARY KEY, 
            instruction TEXT,
            reasoning TEXT,
            nav_goals TEXT,
            speech TEXT,
            completed_at REAL,
            success BOOLEAN
        )''')
    log.info("Navigation task database initialized")

def save_successful_task(instruction: str, plan_data: Dict[str, Any]):
    """Save a successful navigation task to the database"""
    task_id = f"task_{int(time.time())}"
    reasoning = plan_data.get('reasoning', '')
    nav_goals = json.dumps(plan_data.get('navigation', []))
    speech = json.dumps(plan_data.get('speech', []))
    
    try:
        with sqlite3.connect(TASK_DB_PATH) as conn:
            conn.execute(
                "INSERT INTO navigation_tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
                (task_id, instruction, reasoning, nav_goals, speech, time.time(), True)
            )
        log.info(f"Saved successful navigation task {task_id}")
    except Exception as e:
        log.error(f"Error saving task to database: {e}")

def find_similar_task(instruction: str) -> Optional[Dict[str, Any]]:
    """Find a similar task in the database"""
    try:
        with sqlite3.connect(TASK_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT instruction, reasoning, nav_goals, speech FROM navigation_tasks WHERE success = 1 ORDER BY completed_at DESC LIMIT 1"
            ).fetchone()
        
        if row:
            log.info(f"Found similar task: '{row['instruction']}'")
            return {
                "instruction": row["instruction"],
                "reasoning": row["reasoning"],
                "navigation": json.loads(row["nav_goals"]),
                "speech": json.loads(row["speech"])
            }
    except Exception as e:
        log.error(f"Error finding similar task: {e}")
    
    return None

# --- WebSocket Connection to Brain Server ---
async def connect_to_brain_server():
    """Connect to the brain server and process sensor data"""
    global brain_ws_connection, sensor_data, running, last_lidar_data
    
    while running:
        try:
            log.info(f"Connecting to brain server at {BRAIN_SERVER_URL}...")
            async with websockets.connect(BRAIN_SERVER_URL) as websocket:
                brain_ws_connection = websocket
                log.info("Connected to brain server successfully")
                
                # Subscribe to relevant topics
                await websocket.send(json.dumps({
                    "type": "subscribe",
                    "topics": ["camera_data", "scan_data"]
                }))
                
                # Process incoming messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        data_type = data.get("type", "")
                        
                        # Process by message type
                        if data_type == "camera_data":
                            # Store camera data
                            sensor_data["camera"] = data
                            
                            # Extract robot position if available
                            robot_pose = data.get("data", {}).get("robot_pose")
                            if robot_pose:
                                sensor_data["robot_position"] = robot_pose
                            
                            # Check for VLM results
                            if "vlm_results" in data and data["vlm_results"]:
                                log.info(f"Received camera data with VLM results: {data['vlm_results'].get('scene_inference', 'No scene description')}")
                                last_observation = data["vlm_results"]
                                last_observation["timestamp"] = time.time()
                        
                        elif data_type == "scan_data":
                            # Store scan data
                            sensor_data["lidar"] = data
                            last_lidar_data = data
                            
                            # Log successful receipt of scan data
                            scan_data = data.get("data", {}).get("scan", {})
                            ranges = scan_data.get("ranges", [])
                            if ranges:
                                valid_ranges = [r for r in ranges if r < 90.0]  # Filter out 99.0 values
                                log.info(f"Received scan data with {len(valid_ranges)}/{len(ranges)} valid readings")
                            
                            # Extract robot position if available
                            robot_pose = data.get("data", {}).get("robot_pose")
                            if robot_pose:
                                sensor_data["robot_position"] = robot_pose
                    
                    except json.JSONDecodeError:
                        log.warning("Received invalid JSON from brain server")
                    except Exception as e:
                        log.error(f"Error processing message from brain server: {e}")
        
        except (websockets.exceptions.ConnectionClosed,
                websockets.exceptions.ConnectionClosedError,
                ConnectionRefusedError) as e:
            brain_ws_connection = None
            log.warning(f"Brain server connection lost: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            brain_ws_connection = None
            log.error(f"Unexpected error in brain server connection: {e}")
            await asyncio.sleep(5)

# --- Data Fetching ---
def get_world_context() -> Dict[str, Any]:
    """Get the current world context from the semantic map server"""
    context = {"robot_pose": sensor_data["robot_position"], "objects": [], "error": None}
    try:
        response = requests.get(SEMANTIC_MAP_URL, timeout=2)
        response.raise_for_status()
        context.update(response.json())
        log.info(f"Fetched world context: {len(context.get('objects', []))} objects mapped")
    except requests.RequestException as e:
        log.error(f"Could not connect to Semantic Map Server: {e}")
        context["error"] = f"Could not retrieve map state: {str(e)}"
    return context

def get_latest_camera_frame() -> Optional[str]:
    """Find the most recent camera frame in the sensor logs"""
    global last_camera_frame, sensor_data
    
    # First check if we have a camera frame in current sensor data
    if sensor_data["camera"] and "data" in sensor_data["camera"]:
        image_base64 = sensor_data["camera"]["data"].get("image_base64")
        if image_base64:
            log.info("Using camera frame from current sensor data")
            last_camera_frame = image_base64
            return image_base64
    
    # If not, try to get from log files
    try:
        # Find the most recent log file
        list_of_files = glob.glob(os.path.join(SENSOR_LOG_DIR, 'sensor_data_*.jsonl'))
        if not list_of_files:
            log.warning("No sensor log files found")
            return last_camera_frame  # Return the last known frame if no logs are found
        
        latest_file = max(list_of_files, key=os.path.getctime)
        
        with open(latest_file, 'rb') as f:
            # Read the last ~200KB to find a recent camera frame
            f.seek(max(0, f.seek(0, os.SEEK_END) - 204800))
            lines = f.readlines()
            
            # Look through the most recent entries first
            for line in reversed(lines):
                try:
                    data = json.loads(line.decode('utf-8'))
                    if data.get("type") == "camera_data":
                        image_base64 = data.get("data", {}).get("image_base64")
                        if image_base64:
                            log.info(f"Found camera frame in {os.path.basename(latest_file)}")
                            last_camera_frame = image_base64  # Cache this frame
                            return image_base64
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
        
        log.warning("No camera frame found in recent logs")
        return last_camera_frame  # Return last known frame if none found
    
    except Exception as e:
        log.error(f"Error getting camera frame: {e}")
        return last_camera_frame

def estimate_object_distance(object_name: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Estimate distance to an object seen in camera using lidar data
    
    Args:
        object_name: Name of the object to find distance to
    
    Returns:
        Tuple of (distance, angle, x, y) or (None, None, None, None) if not found
    """
    global last_observation, last_lidar_data, sensor_data
    
    # Step 1: Find the object in VLM results
    if not last_observation or "objects" not in last_observation:
        log.warning("No VLM observation available for distance estimation")
        return None, None, None, None
    
    vlm_objects = last_observation.get("objects", [])
    target_object = None
    
    for obj in vlm_objects:
        if obj.get("name", "").lower() == object_name.lower():
            target_object = obj
            break
    
    if not target_object:
        log.warning(f"Object '{object_name}' not found in VLM results")
        return None, None, None, None
    
    # Step 2: Get the object's position in the image
    box_2d = target_object.get("box_2d", None)
    if not box_2d or len(box_2d) != 4:
        log.warning(f"No bounding box available for object '{object_name}'")
        return None, None, None, None
    
    # Calculate center and bottom center of object in image (normalized 0-1)
    x_min, y_min, x_max, y_max = box_2d
    center_x = (x_min + x_max) / 2
    bottom_center_x = center_x  # X-coordinate of bottom center point
    
    # Debug logging
    log.info(f"Object '{object_name}' bounding box: ({x_min:.2f}, {y_min:.2f}, {x_max:.2f}, {y_max:.2f})")
    log.info(f"Object center in image: {center_x:.4f}")
    
    # Determine the image width (default to 640 if not available)
    image_width = 640
    image_height = 480
    if sensor_data["camera"] and "data" in sensor_data["camera"]:
        camera_info = sensor_data["camera"]["data"].get("camera_info", {})
        if camera_info:
            image_width = camera_info.get("width", 640)
            image_height = camera_info.get("height", 480)
    
    # Step 3: Map image position to lidar angle
    # Camera FOV is typically around 60-70 degrees
    camera_fov_degrees = 65
    camera_fov = math.radians(camera_fov_degrees)
    
    # Convert image x-position to angle relative to camera center
    # center_x=0 -> -camera_fov/2, center_x=1 -> +camera_fov/2
    relative_angle = (center_x - 0.5) * camera_fov
    
    # Get robot orientation from sensor data
    robot_theta = sensor_data["robot_position"].get("theta", 0)
    
    # Calculate global angle (relative to world frame)
    global_angle = robot_theta + relative_angle
    
    # Step 4: Use lidar data to find the distance
    if last_lidar_data:
        # Extract scan data
        scan_data = last_lidar_data.get("data", {}).get("scan", {})
        ranges = scan_data.get("ranges", [])
        angle_min = scan_data.get("angle_min", -math.pi)
        angle_max = scan_data.get("angle_max", math.pi)
        angle_increment = scan_data.get("angle_increment", 0.01)
        
        # Calculate number of angle bins
        num_angles = len(ranges) if ranges else 0
        
        if num_angles > 0:
            # Log scan data stats
            valid_ranges = [r for r in ranges if r < 90.0]  # Filter out 99.0 values
            log.info(f"LIDAR data: {len(valid_ranges)}/{len(ranges)} valid readings")
            
            # Find the index in the LiDAR scan closest to our object's angle
            # We need to adjust the angle to match LiDAR coordinate system
            # LiDAR 0° is typically forward, then angles increase counterclockwise
            
            # Convert relative_angle to LiDAR angle space
            lidar_angle = relative_angle
            
            # Find index in LiDAR data for this angle
            angle_diff = [abs(angle_min + i * angle_increment - lidar_angle) for i in range(num_angles)]
            best_idx = angle_diff.index(min(angle_diff))
            
            # Get the corresponding range
            distance = ranges[best_idx]
            
            # Check if it's a valid reading (not 99.0)
            if distance < 90.0:  # Valid range
                log.info(f"Found valid LIDAR distance for '{object_name}': {distance:.2f}m at angle index {best_idx}")
                
                # Calculate x,y coordinates
                robot_x = sensor_data["robot_position"].get("x", 0)
                robot_y = sensor_data["robot_position"].get("y", 0)
                
                x = robot_x + distance * math.cos(global_angle)
                y = robot_y + distance * math.sin(global_angle)
                
                return distance, global_angle, x, y
            else:
                log.info(f"LIDAR reading at angle index {best_idx} is invalid (value={distance}). Using visual estimation.")
    else:
        log.warning("No LIDAR data available. Using visual estimation.")
    
    # Step 5: If no valid LIDAR data, estimate based on object size and position in image
    # Objects at the bottom of the image are typically closer
    object_height = y_max - y_min
    object_width = x_max - x_min
    
    # Calculate metrics
    normalized_height = object_height / image_height
    normalized_width = object_width / image_width
    object_area_ratio = (object_width * object_height) / (image_width * image_height)
    bottom_position = y_max / image_height  # 1.0 means at bottom of image
    
    # Base distance estimate using multiple factors
    
    # 1. Position-based estimate (objects at the bottom are closer)
    # bottom_position = 1.0 (bottom of screen) -> close (1.0m)
    # bottom_position = 0.5 (middle of screen) -> medium (3.0m)
    # bottom_position = 0.0 (top of screen) -> far (5.0m)
    position_based = 1.0 + 4.0 * (1.0 - bottom_position)
    
    # 2. Size-based estimate (larger objects are closer)
    # Normalize object size 0.0-1.0 and invert (smaller = farther)
    size_factor = min(1.0, 10.0 * object_area_ratio)  # Cap at 1.0
    size_based = 1.0 + 9.0 * (1.0 - size_factor)
    
    # Combine estimates with weighting
    # Position is a stronger indicator than size
    distance = 0.7 * position_based + 0.3 * size_based
    
    # Log the estimation process
    log.info(f"Visual distance estimate for '{object_name}': position_based={position_based:.2f}m, " +
             f"size_based={size_based:.2f}m, final={distance:.2f}m")
    
    # Ensure distance is reasonable (between 0.5m and 10.0m)
    distance = max(0.5, min(10.0, distance))
    
    # Calculate x,y coordinates
    robot_x = sensor_data["robot_position"].get("x", 0)
    robot_y = sensor_data["robot_position"].get("y", 0)
    
    x = robot_x + distance * math.cos(global_angle)
    y = robot_y + distance * math.sin(global_angle)
    
    return distance, global_angle, x, y

def query_vlm_server(image_base64: str) -> Dict[str, Any]:
    """Query the VLM server with an image"""
    try:
        log.info("Sending image to VLM server...")
        response = requests.post(
            VLM_SERVER_URL,
            json={"image": image_base64},
            timeout=30  # 30 second timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            log.info(f"VLM server response: {result.get('scene_inference', '')[:100]}...")
            return result
        else:
            log.error(f"VLM server error: {response.status_code} - {response.text}")
            return {"error": f"VLM server error: {response.status_code}"}
    
    except requests.RequestException as e:
        log.error(f"Error querying VLM server: {e}")
        return {"error": f"Could not connect to VLM server: {str(e)}"}

# --- LLM Model Management ---
def load_gpt_model():
    """Load the GPT-OSS model"""
    global model, tokenizer
    
    try:
        log.info(f"Loading GPT-OSS model: {GPT_MODEL_NAME} on {GPU_DEVICE}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL_NAME)
        
        # Load model with configuration for efficiency
        model = AutoModelForCausalLM.from_pretrained(
            GPT_MODEL_NAME,
            device_map=GPU_DEVICE,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
        )
        
        # Set to evaluation mode
        model.eval()
        
        log.info("GPT-OSS model loaded successfully")
        return True
    
    except Exception as e:
        log.error(f"Error loading GPT-OSS model: {e}")
        import traceback
        traceback.print_exc()
        return False

def repair_json(broken_json: str) -> str:
    """Attempt to repair broken JSON"""
    # Common errors include missing quotes around property names
    try:
        # Try to replace unquoted property names
        fixed = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', broken_json)
        # Test if it works
        json.loads(fixed)
        log.info("Successfully repaired JSON")
        return fixed
    except:
        # If that fails, use a more structured approach
        try:
            # Create a fallback JSON with minimal structure
            return json.dumps({
                "reasoning": "Failed to parse GPT response properly. The model may have generated malformed JSON.",
                "navigation": [{"x": 0.0, "y": 0.0, "description": "Remain at current position due to parsing error"}],
                "speech": ["I'm having trouble understanding what to do. Could you clarify?"]
            })
        except:
            return ""

def extract_json(response_text: str) -> str:
    """Extract and fix JSON from model response"""
    # First try to find JSON within markdown code blocks
    if "```json" in response_text:
        try:
            start = response_text.find("```json") + len("```json")
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
            # Try to parse as is
            try:
                json.loads(json_str)
                return json_str
            except:
                # If that fails, attempt repair
                return repair_json(json_str)
        except Exception:
            pass
    
    # Try to find JSON using curly braces
    start = response_text.find('{')
    end = response_text.rfind('}') + 1
    if start != -1 and end > start:
        json_str = response_text[start:end].strip()
        try:
            json.loads(json_str)
            return json_str
        except:
            return repair_json(json_str)
    
    return ""

def build_system_prompt():
    """Build the system prompt for GPT-OSS"""
    return """You are the navigation planning system for an R2D2-like robot. Your role is to generate navigation goals and speech.

Your output MUST be a valid JSON object with this structure:
{
  "reasoning": "Your detailed reasoning process",
  "navigation": [
    {
      "x": 2.5,
      "y": 1.3,
      "description": "Navigate to the table"
    }
  ],
  "speech": [
    "I'll go to the table now.",
    "I've reached the table. I can see several objects on it."
  ]
}

Available actions:
1. Navigation: Generate (x,y) coordinates for the robot to move to
2. Speech: Create text for the robot to speak

For navigation goals:
- Use the semantic map to find object locations
- Provide accurate (x,y) coordinates in meters
- Include descriptive text for each goal

Your response must start with '{' and end with '}'. Do not include any text before or after the JSON.
"""

def build_user_prompt(instruction: str, world_context: Dict[str, Any], vlm_results: Dict[str, Any]):
    """Build the user prompt with instruction and current state"""
    # Format world context
    context_str = "WORLD CONTEXT:\n"
    
    # Robot position
    robot_pose = world_context.get("robot_pose", {})
    if robot_pose:
        context_str += f"Robot position: x={robot_pose.get('x', 0):.2f}, y={robot_pose.get('y', 0):.2f}, θ={robot_pose.get('theta', 0):.2f} radians\n"
    else:
        context_str += "Robot position: Unknown\n"
    
    # Map objects
    objects = world_context.get("objects", [])
    if objects:
        context_str += f"Map contains {len(objects)} objects:\n"
        # Group objects by type
        object_types = {}
        for obj in objects[:15]:  # Limit to first 15 objects
            obj_type = obj.get("name", "unknown")
            if obj_type not in object_types:
                object_types[obj_type] = []
            
            pos = obj.get("position", {})
            object_types[obj_type].append(f"({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f})")
        
        # List objects by type
        for obj_type, positions in object_types.items():
            context_str += f"- {len(positions)} {obj_type}(s): {', '.join(positions[:3])}"
            if len(positions) > 3:
                context_str += ", ..."
            context_str += "\n"
    else:
        context_str += "No objects in map\n"
    
    # Format VLM results
    vision_str = "VISUAL PERCEPTION:\n"
    
    if "error" in vlm_results:
        vision_str += f"Error: {vlm_results['error']}\n"
    else:
        # Scene description
        scene_inference = vlm_results.get("scene_inference", "No scene description available")
        vision_str += f"Scene: {scene_inference}\n"
        
        # Detected objects
        objects = vlm_results.get("objects", [])
        if objects:
            vision_str += f"Detected {len(objects)} objects:\n"
            for i, obj in enumerate(objects[:5]):  # Limit to 5 objects
                name = obj.get("name", "Unknown")
                confidence = obj.get("confidence", 0)
                description = obj.get("description", "")
                
                # Add distance estimate if available
                distance, angle, x, y = estimate_object_distance(name)
                distance_str = ""
                if distance is not None:
                    distance_str = f", estimated distance: {distance:.2f}m"
                
                vision_str += f"- {name} ({confidence:.2f}){distance_str}: {description}\n"
            
            if len(objects) > 5:
                vision_str += f"... and {len(objects) - 5} more objects\n"
        else:
            vision_str += "No objects detected\n"
    
    # Combine everything into the user prompt
    return f"""
{context_str}

{vision_str}

INSTRUCTION: {instruction}

Generate a navigation plan with specific (x,y) coordinates and speech responses to fulfill this instruction.
"""

def generate_plan(instruction: str) -> Dict[str, Any]:
    """Generate a navigation plan using GPT-OSS"""
    global last_observation
    
    # Get world context from semantic map
    world_context = get_world_context()
    
    # Get visual perception
    # First check if we have a recent observation
    if last_observation and time.time() - last_observation.get("timestamp", 0) < 30:
        vlm_results = last_observation
        log.info("Using recent VLM observation")
    else:
        # Get the latest camera frame
        image_base64 = get_latest_camera_frame()
        if image_base64:
            # Query the VLM server
            vlm_results = query_vlm_server(image_base64)
            vlm_results["timestamp"] = time.time()
            last_observation = vlm_results
        else:
            vlm_results = {"error": "No camera frame available", "timestamp": time.time()}
            last_observation = vlm_results
    
    # Check for similar tasks in database
    similar_task = find_similar_task(instruction)
    if similar_task:
        log.info(f"Using similar task as a reference: {similar_task['instruction']}")
    
    # Build prompts
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(instruction, world_context, vlm_results)
    
    # If we have a similar task, add it to the prompt
    if similar_task:
        example_json = json.dumps({
            "reasoning": similar_task["reasoning"],
            "navigation": similar_task["navigation"],
            "speech": similar_task["speech"]
        }, indent=2)
        user_prompt = f"Here's a similar example:\n```json\n{example_json}\n```\n\n{user_prompt}"
    
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    try:
        log.info("Generating navigation plan with GPT-OSS...")
        
        # Prepare input
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the response
        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        log.info(f"GPT-OSS Response:\n{response_text[:200]}...")
        
        # Extract JSON from response
        json_str = extract_json(response_text)
        if not json_str:
            log.error("No JSON found in response")
            return {"error": "Failed to extract JSON from model response"}
        
        # Parse the JSON
        try:
            plan_data = json.loads(json_str)
            return plan_data
        except json.JSONDecodeError as e:
            log.error(f"JSON parse error: {e}")
            # Final fallback
            return {
                "error": f"JSON parse error: {e}",
                "reasoning": "Failed to generate a valid plan",
                "navigation": [{"x": 0.0, "y": 0.0, "description": "Stay in place due to error"}],
                "speech": ["I'm sorry, I encountered an error while planning. Could you try a different instruction?"]
            }
    
    except Exception as e:
        log.error(f"Error generating plan: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error generating plan: {str(e)}"}

def observe_environment():
    """Force an observation of the environment using VLM"""
    global last_observation
    
    print("Observing environment with camera...")
    
    # Get the latest camera frame
    image_base64 = get_latest_camera_frame()
    if not image_base64:
        print("❌ Could not find a camera frame to analyze")
        return False
    
    # Query the VLM server
    print("Sending image to VLM server for analysis...")
    vlm_results = query_vlm_server(image_base64)
    
    if "error" in vlm_results:
        print(f"❌ VLM error: {vlm_results['error']}")
        return False
    
    # Update last observation
    vlm_results["timestamp"] = time.time()
    last_observation = vlm_results
    
    # Print the results
    print(f"✅ Scene: {vlm_results.get('scene_inference', 'No description')}")
    
    objects = vlm_results.get("objects", [])
    if objects:
        print(f"Detected {len(objects)} objects:")
        for i, obj in enumerate(objects[:5]):  # Limit to 5 objects
            name = obj.get("name", "Unknown")
            description = obj.get("description", "")
            
            # Add distance estimate if available
            distance, angle, x, y = estimate_object_distance(name)
            distance_str = ""
            if distance is not None:
                distance_str = f", distance: {distance:.2f}m"
                if x is not None and y is not None:
                    distance_str += f", position: ({x:.2f}, {y:.2f})"
            
            print(f"  - {name}{distance_str}: {description}")
        
        if len(objects) > 5:
            print(f"  ... and {len(objects) - 5} more objects")
    else:
        print("No objects detected")
    
    return True

def execute_navigation_plan(plan_data: Dict[str, Any], instruction: str):
    """Execute a navigation plan"""
    if not plan_data:
        print("❌ No plan data provided")
        return False
    
    if "error" in plan_data:
        print(f"❌ Error: {plan_data.get('error', 'Invalid plan')}")
        return False
    
    print("\n" + "="*50)
    print(" EXECUTING NAVIGATION PLAN ".center(50, "="))
    print("="*50 + "\n")
    
    # Print reasoning
    reasoning = plan_data.get("reasoning", "No reasoning provided")
    print(f"🤖 Reasoning: {reasoning}\n")
    
    # Execute navigation goals
    nav_goals = plan_data.get("navigation", [])
    speech_commands = plan_data.get("speech", [])
    
    if not nav_goals and not speech_commands:
        print("❌ Plan contains no navigation goals or speech commands")
        return False
    
    success = True
    speech_index = 0
    
    # Execute each navigation goal
    for i, goal in enumerate(nav_goals):
        if not isinstance(goal, dict) or "x" not in goal or "y" not in goal:
            print(f"❌ Invalid navigation goal: {goal}")
            success = False
            continue
        
        x = goal.get("x", 0)
        y = goal.get("y", 0)
        theta = goal.get("theta", 0)  # Optional rotation
        description = goal.get("description", f"Location ({x}, {y})")
        
        print(f"\n--- Navigation Goal {i+1}/{len(nav_goals)}: {description} ---")
        print(f"✅ Sending coordinates: ({x}, {y}) to navigation system")
        
        # Send navigation goal to ROS2
        if ros2_bridge_connected or check_ros2_bridge():
            send_success = send_navigation_goal(x, y, description, theta)
            if not send_success:
                print("⚠️ Failed to send navigation goal to robot. Continuing in simulation mode.")
        else:
            print("⚠️ ROS2 bridge not connected. Running in simulation mode only.")
        
        # Update robot position (in simulation)
        sensor_data["robot_position"]["x"] = x
        sensor_data["robot_position"]["y"] = y
        
        # Say something when we're moving
        if speech_index < len(speech_commands):
            speech = speech_commands[speech_index]
            print(f"🔊 Speaking: \"{speech}\"")
            
            # Send speech command to ROS2
            if ros2_bridge_connected or check_ros2_bridge():
                send_speech_command(speech)
            
            speech_index += 1
            
            # Give time for speech and movement
            time.sleep(2)
    
    # Say any remaining speech commands
    while speech_index < len(speech_commands):
        speech = speech_commands[speech_index]
        print(f"\n--- Speech Command ---")
        print(f"🔊 Speaking: \"{speech}\"")
        
        # Send speech command to ROS2
        if ros2_bridge_connected or check_ros2_bridge():
            send_speech_command(speech)
        
        speech_index += 1
        time.sleep(1)  # Brief pause between speech commands
    
    print("\n" + "="*50)
    
    if success:
        print("✅ Navigation plan executed successfully")
        # Save successful task
        save_successful_task(instruction, plan_data)
    else:
        print("❌ Navigation plan execution had errors")
    
    return success

def debug_scan_data():
    """Display detailed debug information for LiDAR scan data"""
    if last_lidar_data:
        scan_data = last_lidar_data.get("data", {}).get("scan", {})
        ranges = scan_data.get("ranges", [])
        angle_min = scan_data.get("angle_min", 0)
        angle_max = scan_data.get("angle_max", 0)
        angle_increment = scan_data.get("angle_increment", 0)
        
        # Calculate stats
        valid_ranges = [r for r in ranges if r < 90.0]
        min_range = min(valid_ranges) if valid_ranges else "N/A"
        max_range = max(valid_ranges) if valid_ranges else "N/A"
        avg_range = sum(valid_ranges)/len(valid_ranges) if valid_ranges else "N/A"
        
        print("\nLiDAR Scan Debug:")
        print(f"  Total readings: {len(ranges)}")
        print(f"  Valid readings: {len(valid_ranges)}")
        print(f"  Angle range: {math.degrees(angle_min):.1f}° to {math.degrees(angle_max):.1f}°")
        print(f"  Angle increment: {math.degrees(angle_increment):.3f}°")
        print(f"  Distance range: {min_range} to {max_range} meters")
        print(f"  Average distance: {avg_range} meters")
        print(f"  Robot position: x={sensor_data['robot_position'].get('x', 0):.2f}, " +
              f"y={sensor_data['robot_position'].get('y', 0):.2f}, " +
              f"θ={sensor_data['robot_position'].get('theta', 0):.2f}")
        
        # Print a simple visualization of the scan
        print("\nScan visualization (top-down view, robot at center):")
        print("     " + "-"*40)
        for i in range(8):
            angle_idx = int((i * len(ranges)) / 8)
            angle = angle_min + angle_idx * angle_increment
            distance = ranges[angle_idx]
            symbol = "•" if distance < 90.0 else " "
            direction = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖"][i]
            print(f"{direction} {math.degrees(angle):6.1f}°: {distance:5.2f}m {symbol*min(40, int(distance*2))}")
        print("     " + "-"*40)
        
        # Show a few sample ranges at key angles
        print("\nSample readings (at 45° intervals):")
        for i in range(0, len(ranges), max(1, len(ranges) // 8)):
            angle = angle_min + i * angle_increment
            distance = ranges[i]
            valid = "✓" if distance < 90.0 else "✗"
            print(f"  Index {i:4d}, Angle {math.degrees(angle):6.1f}°: {distance:5.2f}m {valid}")
    else:
        print("No LiDAR data available")

def check_ros2_bridge_status():
    """Check the status of the ROS2 bridge and display info"""
    if check_ros2_bridge():
        print("✅ ROS2 bridge connected")
        try:
            response = requests.get(f"{ROS2_BRIDGE_URL}/status", timeout=2)
            if response.status_code == 200:
                status_data = response.json()
                print(f"  Status: {status_data.get('status', 'unknown')}")
                uptime = status_data.get('uptime', 0)
                print(f"  Uptime: {uptime:.1f} seconds")
                print(f"  Last command: {time.ctime(status_data.get('last_command_time', 0))}")
        except:
            print("  Could not fetch detailed status")
    else:
        print("❌ ROS2 bridge not connected")
        print("  Make sure the ROS2 bridge is running on the laptop")
        print("  and that SSH tunneling is set up correctly")

def print_help():
    """Print help information"""
    print("\nAvailable commands:")
    print("  help                - Show this help message")
    print("  exit, quit          - Exit the program")
    print("  observe             - Use VLM to observe the environment")
    print("  context             - Show the current world context")
    print("  history             - Show previous successful tasks")
    print("  locate <object>     - Estimate distance and position of an object")
    print("  scan_debug          - Show detailed information about LiDAR scan data")
    print("  ros2_status         - Check the status of the ROS2 bridge connection")
    print("  speak <text>        - Send a speech command to the robot")
    print("  <text>              - Process as an instruction for the robot")
    print()

def print_context():
    """Print the current context"""
    print("\nCurrent Context:")
    
    # Print robot position
    pos = sensor_data["robot_position"]
    print(f"Robot Position: x={pos.get('x', 0):.2f}, y={pos.get('y', 0):.2f}, θ={pos.get('theta', 0):.2f}")
    
    # Print brain server connection status
    if brain_ws_connection:
        print("Brain Server: Connected")
    else:
        print("Brain Server: Disconnected")
    
    # Print ROS2 bridge status
    if ros2_bridge_connected:
        print("ROS2 Bridge: Connected")
    else:
        print("ROS2 Bridge: Disconnected")
    
    # Print semantic map
    world_context = get_world_context()
    objects = world_context.get("objects", [])
    if objects:
        print(f"Semantic Map: {len(objects)} objects")
        object_types = {}
        for obj in objects:
            obj_type = obj.get("name", "unknown")
            if obj_type not in object_types:
                object_types[obj_type] = 0
            object_types[obj_type] += 1
        
        for obj_type, count in object_types.items():
            print(f"  - {count} {obj_type}(s)")
    else:
        print("Semantic Map: No objects")
    
    # Print last observation
    if last_observation and "scene_inference" in last_observation:
        print(f"\nLast Visual Observation: {last_observation.get('scene_inference', 'No description')}")
        objects = last_observation.get("objects", [])
        if objects:
            print(f"Detected Objects: {len(objects)}")
            for i, obj in enumerate(objects[:3]):  # Limit to 3 objects
                print(f"  - {obj.get('name', 'Unknown')}: {obj.get('description', '')}")
            if len(objects) > 3:
                print(f"  ... and {len(objects) - 3} more")
    else:
        print("\nNo Visual Observation Available")
    
    # Print last lidar data
    if last_lidar_data:
        scan_data = last_lidar_data.get("data", {}).get("scan", {})
        ranges = scan_data.get("ranges", [])
        if ranges:
            valid_ranges = [r for r in ranges if r < 90.0]
            if valid_ranges:
                min_range = min(valid_ranges)
                max_range = max(valid_ranges)
                avg_range = sum(valid_ranges)/len(valid_ranges)
                print(f"\nLidar: {len(valid_ranges)} valid readings, range {min_range:.2f}m to {max_range:.2f}m, avg {avg_range:.2f}m")
            else:
                print("\nLidar: No valid readings")
    else:
        print("\nNo Lidar Data Available")
    
    print()

# --- Command Thread ---
def command_thread():
    """Thread for handling user commands"""
    global running
    
    print("\nR2D2 GPT-OSS Navigation Planner")
    print("Type 'help' for available commands, 'exit' to quit\n")
    
    while running:
        try:
            command = input("R2D2> ")
            
            if not command:
                continue
            
            if command.lower() in ["exit", "quit"]:
                running = False
                print("Shutting down...")
                break
            
            elif command.lower() == "help":
                print_help()
            
            elif command.lower() == "observe":
                observe_environment()
            
            elif command.lower() == "context":
                print_context()
            
            elif command.lower() == "scan_debug":
                debug_scan_data()
            
            elif command.lower() == "ros2_status":
                check_ros2_bridge_status()
            
            elif command.lower().startswith("speak "):
                text = command[len("speak "):].strip()
                if text:
                    print(f"🔊 Speaking: \"{text}\"")
                    if ros2_bridge_connected or check_ros2_bridge():
                        if send_speech_command(text):
                            print("✅ Speech command sent to robot")
                        else:
                            print("❌ Failed to send speech command")
                    else:
                        print("❌ ROS2 bridge not connected")
                else:
                    print("Please specify text to speak")
            
            elif command.lower().startswith("locate "):
                object_name = command[len("locate "):].strip()
                if object_name:
                    distance, angle, x, y = estimate_object_distance(object_name)
                    if distance is not None:
                        print(f"\nEstimated location of '{object_name}':")
                        print(f"  Distance: {distance:.2f} meters")
                        print(f"  Angle: {math.degrees(angle):.1f} degrees")
                        print(f"  Position: ({x:.2f}, {y:.2f})")
                    else:
                        print(f"\nCould not locate '{object_name}'. Make sure the object is in view.")
                else:
                    print("Please specify an object to locate")
            
            elif command.lower() == "history":
                try:
                    with sqlite3.connect(TASK_DB_PATH) as conn:
                        conn.row_factory = sqlite3.Row
                        rows = conn.execute(
                            "SELECT id, instruction, completed_at FROM navigation_tasks WHERE success = 1 ORDER BY completed_at DESC LIMIT 5"
                        ).fetchall()
                    
                    if rows:
                        print("\nRecent Successful Tasks:")
                        for row in rows:
                            task_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["completed_at"]))
                            print(f"  [{task_time}] {row['instruction']}")
                    else:
                        print("\nNo successful tasks found in history")
                except Exception as e:
                    print(f"Error getting history: {e}")
            
            else:
                # Process as an instruction
                plan = generate_plan(command)
                execute_navigation_plan(plan, command)
        
        except KeyboardInterrupt:
            running = False
            print("\nExiting...")
            break
        
        except Exception as e:
            print(f"Error: {e}")
            log.error(f"Unexpected error in command thread: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Main entry point"""
    global running, ros2_bridge_connected
    
    print("\n" + "="*70)
    print(" R2D2 GPT-OSS NAVIGATION PLANNER v3.0 ".center(70, "="))
    print("="*70 + "\n")
    
    # Initialize database
    init_database()
    
    # Make sure data directory exists
    os.makedirs(SENSOR_LOG_DIR, exist_ok=True)
    
    # Setup SSH tunnel
    setup_ssh_tunnel()
    
    # Check ROS2 bridge
    ros2_bridge_connected = check_ros2_bridge()
    if ros2_bridge_connected:
        print("✅ ROS2 bridge connected")
    else:
        print("⚠️ ROS2 bridge not connected - running in simulation mode only")
    
    # Load GPT-OSS model
    if not load_gpt_model():
        print("❌ Failed to load GPT-OSS model")
        return
    
    print("✅ Navigation planner initialized")
    
    # Start command thread
    cmd_thread = threading.Thread(target=command_thread)
    cmd_thread.daemon = True
    cmd_thread.start()
    
    # Start websocket connection to brain server
    brain_task = asyncio.create_task(connect_to_brain_server())
    
    # Wait for command thread to finish
    while running and cmd_thread.is_alive():
        await asyncio.sleep(0.1)
    
    # Cleanup
    running = False
    brain_task.cancel()
    try:
        await brain_task
    except asyncio.CancelledError:
        pass
    
    print("Navigation planner shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        running = False
        print("\nExiting...")