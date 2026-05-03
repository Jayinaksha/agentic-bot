#!/usr/bin/env python3
"""
R2D2 Semantic Map Server

This server creates a semantic map by combining:
1. LIDAR scan data for geometric mapping
2. VLM object recognition for semantic understanding

It provides:
- A 2D occupancy grid with semantic labels
- Object tracking and positioning
- Map visualization and storage
- Web interface for viewing the map
"""

import numpy as np
import json
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import math
import cv2
from flask import Flask, jsonify, request, send_file
import os
import signal
import asyncio
import websockets
import logging
import base64
import io
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("semantic_map")

# --- Configuration ---
MAP_RESOLUTION = 0.05  # meters per cell
MAP_WIDTH = 20.0  # meters
MAP_HEIGHT = 20.0  # meters
MAP_ORIGIN_X = -MAP_WIDTH/2  # center the map
MAP_ORIGIN_Y = -MAP_HEIGHT/2
BRAIN_SERVER_URI = "ws://localhost:9002"  # Brain server WebSocket URI
REST_PORT = 8080  # Port for REST API
SAVE_INTERVAL = 60  # Save map every 60 seconds
DATA_DIR = "map_data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- Semantic Classes ---
SEMANTIC_CLASSES = {
    "unknown": 0,
    "free": 1,
    "occupied": 2,
    "chair": 3,
    "table": 4,
    "door": 5,
    "wall": 6,
    "person": 7,
    "obstacle": 8
}

# Colors for visualization
SEMANTIC_COLORS = {
    0: [0.7, 0.7, 0.7],  # unknown - gray
    1: [1.0, 1.0, 1.0],  # free - white
    2: [0.0, 0.0, 0.0],  # occupied - black
    3: [0.8, 0.2, 0.2],  # chair - red
    4: [0.2, 0.6, 0.8],  # table - blue
    5: [0.8, 0.8, 0.2],  # door - yellow
    6: [0.4, 0.4, 0.4],  # wall - dark gray
    7: [0.9, 0.6, 0.2],  # person - orange
    8: [0.5, 0.0, 0.5],  # obstacle - purple
}

class SemanticMap:
    """Maintains a 2D occupancy grid with semantic information"""
    
    def __init__(self):
        # Initialize map size based on configuration
        self.resolution = MAP_RESOLUTION
        self.width = int(MAP_WIDTH / self.resolution)
        self.height = int(MAP_HEIGHT / self.resolution)
        self.origin_x = MAP_ORIGIN_X
        self.origin_y = MAP_ORIGIN_Y
        
        # Initialize map arrays
        # Occupancy grid: 0=unknown, 1=free, 2=occupied
        self.occupancy_grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.occupancy_grid.fill(SEMANTIC_CLASSES["unknown"])
        
        # Semantic grid: stores semantic class IDs
        self.semantic_grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.semantic_grid.fill(SEMANTIC_CLASSES["unknown"])
        
        # Object instances with positions
        self.objects = {}
        
        # Robot pose
        self.robot_pose = {"x": 0.0, "y": 0.0, "theta": 0.0, "frame": "map"}
        
        # Visualization figure
        self.fig = None
        self.ax = None
        
        # Map update timestamp
        self.last_update = time.time()
        
        # Lock for thread safety
        self.map_lock = threading.Lock()
        
        logger.info(f"Initialized semantic map: {self.width}x{self.height} cells at {self.resolution}m resolution")

    def world_to_map(self, x, y):
        """Convert world coordinates to map coordinates"""
        i = int((y - self.origin_y) / self.resolution)
        j = int((x - self.origin_x) / self.resolution)
        
        # Ensure within bounds
        i = max(0, min(i, self.height - 1))
        j = max(0, min(j, self.width - 1))
        
        return i, j
    
    def map_to_world(self, i, j):
        """Convert map coordinates to world coordinates"""
        y = i * self.resolution + self.origin_y
        x = j * self.resolution + self.origin_x
        return x, y

    def update_from_lidar(self, scan_data, robot_pose):
        """Update map with lidar scan data"""
        with self.map_lock:
            self.robot_pose = robot_pose
            
            # Extract scan parameters
            angle_min = scan_data.get("angle_min", -3.14159)
            angle_max = scan_data.get("angle_max", 3.14159)
            angle_increment = scan_data.get("angle_increment", 0.01)
            ranges = scan_data.get("ranges", [])
            
            if not ranges:
                logger.warning("No ranges in scan data")
                return
            
            # Robot position in map coordinates
            robot_i, robot_j = self.world_to_map(robot_pose["x"], robot_pose["y"])
            robot_theta = robot_pose["theta"]
            
            # Mark cells along each ray
            for idx, r in enumerate(ranges):
                # Skip invalid measurements
                if r <= 0.1 or r >= 10.0:  # Minimum and maximum range
                    continue
                
                # Calculate angle of this ray
                angle = angle_min + idx * angle_increment
                # Calculate global angle
                global_angle = robot_theta + angle
                
                # Calculate endpoint in world coordinates
                endpoint_x = robot_pose["x"] + r * math.cos(global_angle)
                endpoint_y = robot_pose["y"] + r * math.sin(global_angle)
                
                # Convert to map coordinates
                endpoint_i, endpoint_j = self.world_to_map(endpoint_x, endpoint_y)
                
                # Use Bresenham's algorithm to trace the ray
                cells = self.bresenham(robot_i, robot_j, endpoint_i, endpoint_j)
                
                # Mark cells along ray as free, except the endpoint
                for i, j in cells[:-1]:
                    # Only mark as free if currently unknown
                    if self.occupancy_grid[i, j] == SEMANTIC_CLASSES["unknown"]:
                        self.occupancy_grid[i, j] = SEMANTIC_CLASSES["free"]
                
                # Mark endpoint as occupied
                if cells:
                    i, j = cells[-1]
                    self.occupancy_grid[i, j] = SEMANTIC_CLASSES["occupied"]
            
            self.last_update = time.time()
    
    def update_from_vlm(self, camera_data, vlm_results):
        """Update semantic information from VLM results"""
        with self.map_lock:
            if not vlm_results or "objects" not in vlm_results:
                return
            
            # Extract robot pose
            robot_pose = camera_data.get("robot_pose", self.robot_pose)
            self.robot_pose = robot_pose
            
            # Get camera info for projection
            camera_info = camera_data.get("camera_info", {})
            width = camera_info.get("width", 320)
            height = camera_info.get("height", 240)
            
            # Parse objects detected by VLM
            objects = vlm_results.get("objects", [])
            
            # Scene description for map annotation
            scene_inference = vlm_results.get("scene_inference", "")
            
            # Robot position and orientation
            robot_x = robot_pose["x"]
            robot_y = robot_pose["y"]
            robot_theta = robot_pose["theta"]
            
            for obj in objects:
                name = obj.get("name", "unknown")
                description = obj.get("description", "")
                box_2d = obj.get("box_2d", [0, 0, 0, 0])
                importance = obj.get("importance", 3)  # Default medium importance
                
                # Skip if no valid box coordinates
                if len(box_2d) != 4 or sum(box_2d) == 0:
                    continue
                
                # Get semantic class for this object
                semantic_class = SEMANTIC_CLASSES.get(name.lower(), SEMANTIC_CLASSES["obstacle"])
                
                # Calculate object distance and angle
                # This is a simplistic approach - in a real system, you'd do proper 3D projection
                box_center_x = (box_2d[0] + box_2d[2]) / 2
                box_center_y = (box_2d[1] + box_2d[3]) / 2
                box_width = box_2d[2] - box_2d[0]
                box_height = box_2d[3] - box_2d[1]
                
                # Estimate distance based on box size (very approximate)
                # This would be replaced with actual depth data or proper monocular depth estimation
                object_size_factor = 0.5  # Depends on expected object size
                distance = object_size_factor * width / box_width
                
                # Limit distance to reasonable range
                distance = min(max(distance, 0.5), 5.0)
                
                # Calculate angle from center of image
                angle_factor = 0.75  # Field of view factor
                angle = angle_factor * (box_center_x - width/2) / (width/2)
                
                # Calculate world coordinates
                global_angle = robot_theta + angle
                object_x = robot_x + distance * math.cos(global_angle)
                object_y = robot_y + distance * math.sin(global_angle)
                
                # Convert to map coordinates
                object_i, object_j = self.world_to_map(object_x, object_y)
                
                # Create an object footprint (size based on importance)
                footprint_size = max(1, min(5, importance))
                for di in range(-footprint_size, footprint_size + 1):
                    for dj in range(-footprint_size, footprint_size + 1):
                        i, j = object_i + di, object_j + dj
                        if 0 <= i < self.height and 0 <= j < self.width:
                            self.semantic_grid[i, j] = semantic_class
                
                # Store object instance
                object_id = f"{name}_{int(time.time()*1000) % 10000}"
                self.objects[object_id] = {
                    "id": object_id,
                    "name": name,
                    "description": description,
                    "position": {"x": object_x, "y": object_y},
                    "detected_time": time.time(),
                    "semantic_class": semantic_class,
                    "importance": importance
                }
            
            self.last_update = time.time()
    
    def bresenham(self, i1, j1, i2, j2):
        """Bresenham's line algorithm for ray tracing"""
        cells = []
        di = abs(i2 - i1)
        dj = abs(j2 - j1)
        si = 1 if i1 < i2 else -1
        sj = 1 if j1 < j2 else -1
        err = di - dj
        
        i, j = i1, j1
        
        while True:
            # Check bounds
            if 0 <= i < self.height and 0 <= j < self.width:
                cells.append((i, j))
            
            if i == i2 and j == j2:
                break
                
            e2 = 2 * err
            if e2 > -dj:
                err -= dj
                i += si
            if e2 < di:
                err += di
                j += sj
                
        return cells
    
    def save_map(self, filename=None):
        """Save the map to a file"""
        if filename is None:
            filename = os.path.join(DATA_DIR, f"semantic_map_{int(time.time())}.npz")
            
        with self.map_lock:
            np.savez(filename, 
                     occupancy_grid=self.occupancy_grid,
                     semantic_grid=self.semantic_grid,
                     objects=json.dumps(self.objects),
                     resolution=self.resolution,
                     origin_x=self.origin_x,
                     origin_y=self.origin_y,
                     width=self.width,
                     height=self.height)
            logger.info(f"Map saved to {filename}")
            return filename
    
    def load_map(self, filename):
        """Load the map from a file"""
        if not os.path.exists(filename):
            logger.warning(f"Map file {filename} does not exist")
            return False
            
        with self.map_lock:
            try:
                data = np.load(filename, allow_pickle=True)
                self.occupancy_grid = data["occupancy_grid"]
                self.semantic_grid = data["semantic_grid"]
                self.objects = json.loads(data["objects"])
                self.resolution = float(data["resolution"])
                self.origin_x = float(data["origin_x"])
                self.origin_y = float(data["origin_y"])
                self.width = int(data["width"])
                self.height = int(data["height"])
                logger.info(f"Map loaded from {filename}")
                return True
            except Exception as e:
                logger.error(f"Error loading map: {e}")
                return False
    
    def visualize(self, save_path=None):
        """Visualize the map and return the figure or save to a file"""
        with self.map_lock:
            # Create blended map for visualization
            blended_map = np.copy(self.occupancy_grid)
            
            # Where semantic labels exist, use them instead
            semantic_mask = (self.semantic_grid > SEMANTIC_CLASSES["occupied"])
            blended_map[semantic_mask] = self.semantic_grid[semantic_mask]
            
            # Create color map
            colors = [SEMANTIC_COLORS[i] for i in range(max(SEMANTIC_CLASSES.values()) + 1)]
            cmap = ListedColormap(colors)
            
            # Create or update figure
            if self.fig is None or self.ax is None:
                self.fig, self.ax = plt.subplots(figsize=(10, 10))
            else:
                self.ax.clear()
            
            # Plot the map
            self.ax.imshow(blended_map, cmap=cmap, origin='lower')
            
            # Plot robot position
            robot_i, robot_j = self.world_to_map(self.robot_pose["x"], self.robot_pose["y"])
            robot_angle = self.robot_pose["theta"]
            
            # Draw robot as triangle
            triangle_size = 10
            dx = triangle_size * math.cos(robot_angle)
            dy = triangle_size * math.sin(robot_angle)
            self.ax.arrow(robot_j, robot_i, dx, dy, head_width=5, head_length=5, 
                         fc='green', ec='green', width=2)
            
            # Draw objects
            for obj_id, obj in self.objects.items():
                obj_x, obj_y = obj["position"]["x"], obj["position"]["y"]
                obj_i, obj_j = self.world_to_map(obj_x, obj_y)
                
                # Create patch based on object type
                if obj["name"].lower() in ["chair", "table", "door"]:
                    patch = patches.Rectangle(
                        (obj_j-2, obj_i-2), 4, 4, 
                        linewidth=1, edgecolor='black', 
                        facecolor=colors[obj["semantic_class"]], alpha=0.7
                    )
                    self.ax.add_patch(patch)
                else:
                    self.ax.scatter(obj_j, obj_i, s=30, c=[colors[obj["semantic_class"]]], 
                                   edgecolor='black', linewidth=1, alpha=0.7)
                
                # Add text label
                self.ax.text(obj_j, obj_i-3, obj["name"], color='black', 
                            fontsize=8, ha='center', backgroundcolor='white', alpha=0.7)
            
            # Add grid
            self.ax.grid(True, alpha=0.3)
            
            # Add title with timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_update))
            self.ax.set_title(f"R2D2 Semantic Map - {timestamp}")
            
            # Add axis labels
            self.ax.set_xlabel("X (map cells)")
            self.ax.set_ylabel("Y (map cells)")
            
            # Add legend
            legend_elements = []
            for name, value in SEMANTIC_CLASSES.items():
                if value <= max(SEMANTIC_CLASSES.values()):
                    legend_elements.append(patches.Patch(color=colors[value], label=name))
            self.ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Map visualization saved to {save_path}")
                return save_path
            else:
                plt.tight_layout()
                return self.fig
    
    def get_map_data(self):
        """Get map data as a dictionary for JSON serialization"""
        with self.map_lock:
            # Create simplified representation for transmission
            return {
                "resolution": self.resolution,
                "origin_x": self.origin_x,
                "origin_y": self.origin_y,
                "width": self.width,
                "height": self.height,
                "robot_pose": self.robot_pose,
                "objects": list(self.objects.values()),
                "last_update": self.last_update
            }

class SemanticMapServer:
    """Server for building and serving the semantic map"""
    
    def __init__(self):
        self.semantic_map = SemanticMap()
        self.last_save_time = time.time()
        self.running = True
        
        # Initialize Flask app for REST API
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Start the REST API in a separate thread
        self.rest_thread = threading.Thread(target=self._run_rest_api)
        self.rest_thread.daemon = True
        self.rest_thread.start()
        
        logger.info("Semantic Map Server initialized")
    
    def setup_routes(self):
        """Set up Flask routes for the REST API"""
        @self.app.route('/map', methods=['GET'])
        def get_map():
            """Get the current map data"""
            return jsonify(self.semantic_map.get_map_data())
        
        @self.app.route('/map/visualization', methods=['GET'])
        def get_visualization():
            """Get a visualization of the map"""
            # Create visualization and save to a temporary file
            temp_file = os.path.join(DATA_DIR, "temp_map_viz.png")
            self.semantic_map.visualize(save_path=temp_file)
            
            # Return the file
            return send_file(temp_file, mimetype='image/png')
        
        @self.app.route('/objects', methods=['GET'])
        def get_objects():
            """Get all detected objects"""
            return jsonify(list(self.semantic_map.objects.values()))
        
        @self.app.route('/robot', methods=['GET'])
        def get_robot_pose():
            """Get current robot pose"""
            return jsonify(self.semantic_map.robot_pose)
        
        @self.app.route('/save', methods=['POST'])
        def save_map():
            """Save the current map"""
            filename = self.semantic_map.save_map()
            return jsonify({"status": "success", "filename": filename})
        
        @self.app.route('/load', methods=['POST'])
        def load_map():
            """Load a map from file"""
            if not request.json or 'filename' not in request.json:
                return jsonify({"status": "error", "message": "Filename required"}), 400
                
            filename = request.json['filename']
            success = self.semantic_map.load_map(filename)
            
            if success:
                return jsonify({"status": "success"})
            else:
                return jsonify({"status": "error", "message": "Failed to load map"}), 400
    
    def _run_rest_api(self):
        """Run the Flask REST API"""
        self.app.run(host='0.0.0.0', port=REST_PORT)
    
    async def process_lidar_data(self, data):
        """Process LIDAR scan data"""
        # Extract scan data and robot pose
        scan_data = data.get("data", {}).get("scan", {})
        robot_pose = data.get("data", {}).get("robot_pose", {"x": 0.0, "y": 0.0, "theta": 0.0})
        
        # Update semantic map
        self.semantic_map.update_from_lidar(scan_data, robot_pose)
        
        # Save map periodically
        current_time = time.time()
        if current_time - self.last_save_time > SAVE_INTERVAL:
            self.semantic_map.save_map()
            self.last_save_time = current_time
    
    async def process_camera_data(self, data):
        """Process camera data with VLM results"""
        # Extract camera data
        camera_data = data.get("data", {})
        vlm_results = data.get("vlm_results")
        
        if vlm_results:
            # Update semantic map with VLM results
            self.semantic_map.update_from_vlm(camera_data, vlm_results)
    
    async def connect_to_brain_server(self):
        """Connect to the brain server and process data"""
        while self.running:
            try:
                async with websockets.connect(BRAIN_SERVER_URI) as websocket:
                    logger.info(f"Connected to brain server at {BRAIN_SERVER_URI}")
                    
                    # Subscribe to relevant topics
                    await websocket.send(json.dumps({
                        "type": "subscribe",
                        "topics": ["scan_data", "camera_data"]
                    }))
                    
                    # Process messages
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            data_type = data.get("type", "")
                            
                            if data_type == "scan_data":
                                await self.process_lidar_data(data)
                            elif data_type == "camera_data" and "vlm_results" in data:
                                await self.process_camera_data(data)
                        
                        except json.JSONDecodeError:
                            logger.warning("Received invalid JSON")
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            import traceback
                            traceback.print_exc()
            
            except (websockets.exceptions.ConnectionClosed, 
                    websockets.exceptions.ConnectionClosedError,
                    ConnectionRefusedError) as e:
                logger.warning(f"Connection to brain server lost: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await asyncio.sleep(5)
    
    async def run(self):
        """Run the semantic map server"""
        # Try to load the most recent map if available
        map_files = [f for f in os.listdir(DATA_DIR) if f.startswith("semantic_map_") and f.endswith(".npz")]
        if map_files:
            # Sort by modification time, newest first
            most_recent = sorted(map_files, key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)), reverse=True)[0]
            self.semantic_map.load_map(os.path.join(DATA_DIR, most_recent))
        
        # Connect to the brain server
        await self.connect_to_brain_server()
    
    def stop(self):
        """Stop the semantic map server"""
        self.running = False
        # Save the map before stopping
        self.semantic_map.save_map()
        logger.info("Semantic Map Server stopped")

def handle_shutdown(sig=None, frame=None):
    """Handle shutdown signal gracefully"""
    logger.info("Shutdown signal received, cleaning up...")
    if 'server' in globals():
        server.stop()
    logger.info("Shutdown complete")
    os._exit(0)

async def main():
    """Main entry point"""
    global server
    
    print("\n" + "="*80)
    print(" R2D2 SEMANTIC MAP SERVER ".center(80, "="))
    print("="*80 + "\n")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Create and run the server
    server = SemanticMapServer()
    await server.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass