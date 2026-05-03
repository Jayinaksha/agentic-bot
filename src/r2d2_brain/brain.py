import asyncio
import websockets
import json
import logging
import numpy as np
import base64
import cv2
from datetime import datetime
import threading
import queue
import os
import time
import signal
from collections import defaultdict
import socket
import requests  # For making HTTP requests to VLM server

# --- Configuration ---
SAVE_IMAGES = False
TERMINAL_OUTPUT_LEVEL = "minimal"
LOG_BUFFER_SIZE = 100
FILE_WRITE_THREAD_COUNT = 2
MAX_QUEUE_SIZE = 5000
WS_MAX_SIZE = 10485760  # 10MB
WS_PING_INTERVAL = 40
WS_PING_TIMEOUT = 30    # Add timeout value

# VLM server configuration
VLM_SERVER_URL = "http://localhost:5000/process_frame"  # Update with actual server address
VLM_PROCESSING_ENABLED = True
VLM_PROCESSING_INTERVAL = 5  # Process every Nth camera frame (to avoid overload)
VLM_QUEUE_SIZE = 10  # Maximum number of frames to queue for VLM processing
VLM_TIMEOUT = 10  # Timeout in seconds for VLM server requests
VLM_MAX_RETRIES = 3  # Maximum retries for VLM server connection

# Terminal output sample rate
TERMINAL_OUTPUT_SAMPLE_RATE = {
    "camera_data": 10,      # Show 1 in 10 camera frames
    "scan_data": 10,        # Show 1 in 10 scan messages
    "imu_data": 10,         # Show 1 in 10 IMU messages
    "tf_data": 10,          # Show 1 in 10 TF messages
    "default": 1            # Show all other types
}

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("r2d2_brain_server.log")]
)
log = logging.getLogger(__name__)

# --- Data Storage ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_LOG_FILE = os.path.join(DATA_DIR, f"sensor_data_{int(time.time())}.jsonl")

# --- Global State ---
sensor_data = {"camera": None, "scan": None, "imu": {}, "robot_pose": None, "timestamp": 0, "tf": None}
file_write_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
vlm_queue = queue.Queue(maxsize=VLM_QUEUE_SIZE)  # Queue for frames to be processed by VLM
camera_data_dict = {}  # Dictionary to store camera data frames by ID for later VLM association
camera_data_dict_lock = threading.Lock()  # Lock for the camera data dictionary
log_buffer = []
log_buffer_lock = threading.Lock()
running = True
file_writers = []
vlm_thread = None
semantic_map_clients = set()  # NEW: Track connected semantic map clients
stats = {
    "start_time": time.time(),
    "messages_received": 0,
    "messages_by_type": defaultdict(int),
    "bytes_written": 0,
    "buffer_high_water": 0,
    "queue_high_water": 0,
    "vlm_processed": 0,
    "vlm_errors": 0,
    "vlm_retries": 0
}
connections = {"sensor_clients": 0, "semantic_map_clients": 0, "messages_received": 0}  # Updated to track semantic map clients
msg_counters = defaultdict(int)
vlm_frame_counter = 0  # Counter for VLM processed frames

def write_to_log():
    """Background thread for writing logs to file"""
    global running, stats
    
    try:
        with open(DATA_LOG_FILE, 'a', buffering=1) as f:  # Use line buffering
            log.info(f"File writer started: {DATA_LOG_FILE}")
            
            while running:
                try:
                    # Try to get an item with a short timeout
                    try:
                        item = file_write_queue.get(timeout=0.5)
                        
                        # Ensure item is properly formatted as JSON
                        if isinstance(item, dict):
                            json_line = json.dumps(item)
                            f.write(json_line + '\n')
                            f.flush()  # Force flush after each write
                            stats["bytes_written"] += len(json_line) + 1
                        else:
                            log.warning(f"Skipping non-dict item: {type(item)}")
                        
                        file_write_queue.task_done()
                    
                    except queue.Empty:
                        # If queue is empty, just continue
                        continue
                
                except Exception as e:
                    log.error(f"Error in file writer: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.1)  # Avoid tight loop on error
    
    except Exception as e:
        log.error(f"Fatal error in file writer: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        log.info("File writer stopping")

def vlm_processor():
    """Background thread for processing camera frames with VLM"""
    global running, stats, vlm_frame_counter, camera_data_dict
    
    log.info("VLM processor thread started")
    
    while running:
        try:
            # Get camera frame from queue
            try:
                camera_data = vlm_queue.get(timeout=0.5)
                vlm_frame_counter += 1
                
                # Extract the base64 image
                image_base64 = camera_data.get('data', {}).get('image_base64', None)
                if not image_base64:
                    log.warning("Camera frame missing image_base64 data")
                    vlm_queue.task_done()
                    continue
                
                # Create metadata for the VLM request
                frame_id = camera_data.get('frame', vlm_frame_counter)
                timestamp = camera_data.get('data', {}).get('timestamp', time.time())
                
                # Send to VLM server with retry logic
                vlm_server_retries = 0
                processing_start_time = time.time()
                
                while vlm_server_retries <= VLM_MAX_RETRIES:
                    try:
                        # Send to VLM server
                        log.info(f"Sending frame {frame_id} to VLM server (attempt {vlm_server_retries+1})")
                        response = requests.post(
                            VLM_SERVER_URL,
                            json={"image": image_base64},
                            timeout=VLM_TIMEOUT
                        )
                        
                        if response.status_code == 200:
                            # Process successful response
                            vlm_result = response.json()
                            processing_time = time.time() - processing_start_time
                            
                            # Create enriched camera data with VLM results embedded
                            with camera_data_dict_lock:
                                if frame_id in camera_data_dict:
                                    # Get the original camera data 
                                    original_camera_data = camera_data_dict[frame_id]
                                    
                                    # Create a combined entry with the VLM results
                                    combined_data = original_camera_data.copy()
                                    
                                    # Add VLM results under the vlm_results key
                                    combined_data['vlm_results'] = {
                                        'processed_time': time.time(),
                                        'processing_duration': processing_time,
                                        'objects': vlm_result.get('objects', []),
                                        'scene_inference': vlm_result.get('scene_inference', ''),
                                        'navigation_advice': vlm_result.get('navigation_advice', '')
                                    }
                                    
                                    # Add the combined data to the file write queue
                                    file_write_queue.put(combined_data)
                                    
                                    # NEW: Forward to semantic map clients
                                    if semantic_map_clients:
                                        asyncio.create_task(forward_to_semantic_map(combined_data))
                                    
                                    # Remove the processed frame from the dictionary to free memory
                                    del camera_data_dict[frame_id]
                                    
                                    # Update stats
                                    stats["vlm_processed"] += 1
                                    log.info(f"VLM processing complete for frame {frame_id} in {processing_time:.2f}s: {vlm_result.get('scene_inference', '')}")
                                else:
                                    log.warning(f"Original camera data for frame {frame_id} not found. VLM results discarded.")
                            
                            break  # Success - exit retry loop
                        else:
                            # Server error
                            log.error(f"VLM server error: {response.status_code} - {response.text}")
                            vlm_server_retries += 1
                            stats["vlm_retries"] += 1
                            
                            if vlm_server_retries > VLM_MAX_RETRIES:
                                stats["vlm_errors"] += 1
                                log.error(f"Max retries reached for frame {frame_id}")
                            else:
                                time.sleep(1)  # Wait before retry
                    
                    except requests.RequestException as e:
                        vlm_server_retries += 1
                        stats["vlm_retries"] += 1
                        
                        if vlm_server_retries > VLM_MAX_RETRIES:
                            log.error(f"Failed to connect to VLM server after {VLM_MAX_RETRIES} attempts: {e}")
                            stats["vlm_errors"] += 1
                        else:
                            log.warning(f"VLM server connection attempt {vlm_server_retries} failed, retrying...")
                            time.sleep(1)  # Wait before retry
                
                # Mark queue task as done
                vlm_queue.task_done()
            
            except queue.Empty:
                # If queue is empty, just continue
                continue
        
        except Exception as e:
            log.error(f"Error in VLM processor: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)  # Avoid tight loop on error
    
    log.info("VLM processor thread stopping")

def flush_log_buffer():
    """Flush the log buffer to the file writing queue"""
    global log_buffer, stats
    
    with log_buffer_lock:
        if not log_buffer:
            return  # Nothing to flush
            
        # Track buffer stats
        if len(log_buffer) > stats["buffer_high_water"]:
            stats["buffer_high_water"] = len(log_buffer)
        
        # Track queue stats
        queue_size = file_write_queue.qsize()
        if queue_size > stats["queue_high_water"]:
            stats["queue_high_water"] = queue_size
        
        # Add all buffered items to the queue
        items_queued = 0
        for item in log_buffer:
            try:
                # Use blocking put with timeout to ensure items get queued
                file_write_queue.put(item, block=True, timeout=0.1)
                items_queued += 1
            except queue.Full:
                log.warning(f"File write queue full, dropped {len(log_buffer) - items_queued} items")
                break
        
        log.debug(f"Flushed {items_queued} items from buffer to queue")
        log_buffer.clear()

def print_to_terminal(data_obj):
    """Print data to terminal based on output level"""
    if TERMINAL_OUTPUT_LEVEL == "none":
        return
    
    data_type = data_obj.get('type', 'unknown')
    
    if TERMINAL_OUTPUT_LEVEL == "minimal":
        timestamp = data_obj.get('timestamp', time.time())
        print(f"[{datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]}] {data_type} - msg #{stats['messages_by_type'][data_type]}")
        
        # For camera data with VLM results, show more details
        if data_type == "camera_data" and 'vlm_results' in data_obj:
            vlm_results = data_obj.get('vlm_results')
            if vlm_results is not None:
                scene = vlm_results.get('scene_inference', 'No scene inference')
                print(f"  Scene: {scene}")
            else:
                print(f"  Scene: No VLM processing yet")
        return
    
    # For full output mode
    print(f"\n===== {data_type.upper()} =====")
    print(json.dumps(data_obj, indent=2))
    print("-" * 40)

def write_to_data_log(data_obj):
    """Non-blocking write to data log through buffer and queue"""
    global log_buffer, stats, msg_counters
    
    # Add timestamp if not present
    if 'timestamp' not in data_obj:
        data_obj['timestamp'] = time.time()
    
    # Update stats
    stats["messages_received"] += 1
    data_type = data_obj.get('type', 'unknown')
    stats["messages_by_type"][data_type] += 1
    
    # For camera data, we'll store a copy in the camera_data_dict for VLM processing
    if data_type == "camera_data":
        frame_id = data_obj.get('frame')
        if frame_id is not None:
            with camera_data_dict_lock:
                # Store a copy of the camera data with placeholder for VLM results
                camera_data_dict[frame_id] = data_obj
                
                # Limit the size of the dictionary to avoid memory issues
                # Remove oldest entries if we have more than 100
                if len(camera_data_dict) > 100:
                    oldest_key = min(camera_data_dict.keys())
                    del camera_data_dict[oldest_key]
    
    # Add to log buffer
    with log_buffer_lock:
        log_buffer.append(data_obj)
        
        # Flush buffer if it reaches the threshold
        if len(log_buffer) >= LOG_BUFFER_SIZE:
            flush_log_buffer()
    
    # Print to terminal based on sampling rate
    msg_counters[data_type] += 1
    sample_rate = TERMINAL_OUTPUT_SAMPLE_RATE.get(data_type, TERMINAL_OUTPUT_SAMPLE_RATE['default'])
    if msg_counters[data_type] % sample_rate == 0:
        print_to_terminal(data_obj)

def print_stats():
    """Print performance statistics"""
    runtime = time.time() - stats["start_time"]
    msgs_per_sec = stats["messages_received"] / runtime if runtime > 0 else 0
    mb_written = stats["bytes_written"] / (1024 * 1024)
    
    log.info(f"Stats: {stats['messages_received']} msgs ({msgs_per_sec:.1f}/sec), {mb_written:.2f} MB written")
    log.info(f"VLM: {stats['vlm_processed']} frames processed, {stats['vlm_errors']} errors, {stats['vlm_retries']} retries")
    log.info(f"Connections: Sensor={connections['sensor_clients']}, SemanticMap={connections['semantic_map_clients']}")  # Updated
    
    type_counts = sorted(stats["messages_by_type"].items(), key=lambda x: x[1], reverse=True)
    log.info("Top message types: " + ", ".join(f"{t}: {c}" for t, c in type_counts[:3]))
    
    # Report on camera data dictionary size
    with camera_data_dict_lock:
        log.info(f"Pending camera frames for VLM processing: {len(camera_data_dict)}")
    
    # Queue status
    log.info(f"Queues: VLM={vlm_queue.qsize()}/{VLM_QUEUE_SIZE}, FileWrite={file_write_queue.qsize()}/{MAX_QUEUE_SIZE}")

# NEW: Helper function to forward data to semantic map clients
async def forward_to_semantic_map(data_obj):
    """Forward data to semantic map clients"""
    if not semantic_map_clients:
        return
        
    try:
        message = json.dumps(data_obj)
        await asyncio.gather(*[client.send(message) for client in semantic_map_clients])
    except Exception as e:
        log.error(f"Error sending to semantic map clients: {e}")

class R2D2BrainProcessor:
    """Processes sensor data and runs inference"""
    
    def __init__(self):
        self.data_counter = 0
        self.last_processed = {"camera": 0, "scan": 0, "imu": 0}
        self.camera_frame_counter = 0
        
        # Start stats thread
        threading.Thread(target=self._stats_thread, daemon=True).start()
        log.info("R2D2 Brain Processor initialized")
    
    def _stats_thread(self):
        while running:
            time.sleep(60)
            print_stats()
    
    def process_sensor_data(self, data_type, data):
        """Main entry point for all sensor data"""
        self.data_counter += 1
        connections["messages_received"] += 1
        
        # Add metadata and timestamp
        data['processed_time'] = time.time()
        data['frame_id'] = self.data_counter
        
        # Process by type
        if data_type == "camera_data":
            self._process_camera_data(data)
        elif data_type == "scan_data":
            self._process_scan_data(data)
        elif "imu" in data_type:
            self._process_imu_data(data_type, data)
        elif data_type == "tf_data":
            self._process_tf_data(data)
        elif data_type == "heartbeat":
            # Just log heartbeats at debug level
            log.debug(f"Received heartbeat")
            return  # Don't log heartbeats to data file
        else:
            log.info(f"Received {data_type} data")
            write_to_data_log({'type': data_type, 'frame': self.data_counter, 'data': data})
    
    def _process_camera_data(self, data):
        """Process camera data without vision inference"""
        try:
            # Update global state
            sensor_data["camera"] = data
            sensor_data["robot_pose"] = data.get("robot_pose", {})
            sensor_data["timestamp"] = data.get("timestamp", time.time())
            self.last_processed["camera"] = time.time()
            self.camera_frame_counter += 1
            
            # Log the full camera data with image_base64
            camera_log_data = {
                'type': 'camera_data',
                'frame': self.data_counter,
                'data': data,  # Store the complete data including image_base64
                'vlm_results': None  # Placeholder for VLM results, will be filled later
            }
            
            # Write the camera data to the log
            write_to_data_log(camera_log_data)
            
            # Add to VLM queue if enabled and meets interval criteria
            if VLM_PROCESSING_ENABLED and self.camera_frame_counter % VLM_PROCESSING_INTERVAL == 0:
                try:
                    # Only enqueue if we have the image data
                    if 'image_base64' in data:
                        vlm_queue.put(camera_log_data, block=False)
                        log.debug(f"Queued camera frame {self.data_counter} for VLM processing")
                    else:
                        log.warning(f"Camera frame {self.data_counter} missing image_base64 data")
                except queue.Full:
                    log.warning("VLM queue full, skipping frame")
            
        except Exception as e:
            log.error(f"Error processing camera data: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_scan_data(self, data):
        """Process scan data"""
        try:
            # Update global state
            sensor_data["scan"] = data
            sensor_data["robot_pose"] = data.get("robot_pose", {})
            self.last_processed["scan"] = time.time()
            
            # Log data
            scan_data_obj = {'type': 'scan_data', 'frame': self.data_counter, 'data': data}
            write_to_data_log(scan_data_obj)
            
            # NEW: Forward to semantic map if clients are connected
            if semantic_map_clients:
                asyncio.create_task(forward_to_semantic_map(scan_data_obj))
            
        except Exception as e:
            log.error(f"Error processing scan data: {e}")
    
    def _process_imu_data(self, data_type, data):
        """Process IMU data"""
        try:
            # Store in sensor_data
            imu_topic = data_type.replace("_data", "")
            sensor_data["imu"][imu_topic] = data
            self.last_processed["imu"] = time.time()
            
            # Log data
            write_to_data_log({'type': data_type, 'frame': self.data_counter, 'data': data})
            
        except Exception as e:
            log.error(f"Error processing IMU data: {e}")
    
    def _process_tf_data(self, data):
        """Process transform frame data"""
        try:
            # Update global state
            sensor_data["tf"] = data
            
            # Log data (minimal version to save space)
            write_to_data_log({
                'type': 'tf_data',
                'frame': self.data_counter,
                'data': data
            })
            
        except Exception as e:
            log.error(f"Error processing TF data: {e}")

async def sensor_handler(websocket):
    """Handles incoming sensor data from the client"""
    connections["sensor_clients"] += 1
    client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    log.info(f"[Sensor Server] Client connected: {client_id}")
    
    # Set TCP_NODELAY for better performance
    try:
        socket_obj = websocket.transport.get_extra_info('socket')
        if socket_obj:
            socket_obj.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except:
        pass
    
    try:
        async for message in websocket:
            try:
                # Parse and process the data
                data = json.loads(message)
                data_type = data.get("type", "unknown")
                processor.process_sensor_data(data_type, data)
            except json.JSONDecodeError:
                log.error("Received invalid JSON")
            except Exception as e:
                log.error(f"Error processing message: {e}")
    except Exception as e:
        log.warning(f"Sensor client disconnected: {e}")
    finally:
        connections["sensor_clients"] -= 1
        log.info(f"Sensor client disconnected: {client_id}")

# NEW: Semantic map handler
async def semantic_map_handler(websocket):
    """Handles connections from the semantic map server"""
    client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    log.info(f"[Semantic Map Server] Connected: {client_id}")
    
    # Add client to set of semantic map clients
    semantic_map_clients.add(websocket)
    connections["semantic_map_clients"] += 1
    
    try:
        async for message in websocket:
            try:
                # Parse the request
                data = json.loads(message)
                request_type = data.get("type", "")
                
                # Handle subscription requests
                if request_type == "subscribe":
                    topics = data.get("topics", [])
                    log.info(f"Semantic map server subscribed to topics: {topics}")
                
            except json.JSONDecodeError:
                log.error("Received invalid JSON from semantic map server")
            except Exception as e:
                log.error(f"Error processing message from semantic map server: {e}")
    except Exception as e:
        log.warning(f"Semantic map server disconnected: {e}")
    finally:
        semantic_map_clients.remove(websocket)
        connections["semantic_map_clients"] -= 1
        log.info(f"Semantic map server disconnected: {client_id}")

def handle_shutdown(sig=None, frame=None):
    """Handle shutdown signal gracefully"""
    global running
    
    log.info("Shutdown signal received, cleaning up...")
    running = False
    
    try:
        # Flush any remaining buffered data
        with log_buffer_lock:
            log.info(f"Flushing {len(log_buffer)} buffered items")
            for item in log_buffer:
                try:
                    file_write_queue.put(item, block=True, timeout=0.1)
                except queue.Full:
                    log.warning("Queue full during shutdown, some data may be lost")
                    break
            log_buffer.clear()
        
        # Wait for queue to drain
        try:
            log.info(f"Waiting for {file_write_queue.qsize()} queued items to be written...")
            file_write_queue.join()  # Wait for all tasks to be processed
        except:
            log.warning("Error waiting for queue to drain")
        
        # Print final stats
        print_stats()
        
        # Give the file writers a chance to finish
        log.info("Waiting for file writers to finish...")
        time.sleep(2)
        
        log.info("Shutdown complete")
    except Exception as e:
        log.error(f"Error during shutdown: {e}")
    finally:
        # Force exit
        os._exit(0)

async def main():
    """Start the server and wait for shutdown"""
    global processor, vlm_thread
    
    print("\n" + "="*80)
    print(" R2D2 BRAIN SERVER WITH INTEGRATED VLM ".center(80, "="))
    print("="*80 + "\n")
    
    # Register shutdown handler
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Start file writer threads
    for _ in range(FILE_WRITE_THREAD_COUNT):
        thread = threading.Thread(target=write_to_log, daemon=True)
        thread.start()
        file_writers.append(thread)
    
    # Start VLM processor thread if enabled
    if VLM_PROCESSING_ENABLED:
        vlm_thread = threading.Thread(target=vlm_processor, daemon=True)
        vlm_thread.start()
        log.info("VLM processor thread started")
    
    # Create processor
    processor = R2D2BrainProcessor()

    # Start WebSocket server for sensor data
    sensor_server = await websockets.serve(
        sensor_handler, 
        "0.0.0.0", 
        9001, 
        max_size=WS_MAX_SIZE,
        ping_interval=WS_PING_INTERVAL,
        ping_timeout=WS_PING_TIMEOUT
    )
    
    # NEW: Start WebSocket server for semantic map connections
    semantic_map_server = await websockets.serve(
        semantic_map_handler,
        "0.0.0.0",
        9002,
        max_size=WS_MAX_SIZE,
        ping_interval=WS_PING_INTERVAL,
        ping_timeout=WS_PING_TIMEOUT
    )
    
    log.info("R2D2 Brain Server running on port 9001 (sensor data)")
    log.info("Semantic Map Server running on port 9002")
    
    # Periodic buffer flush
    async def periodic_flush():
        while running:
            await asyncio.sleep(1)
            flush_log_buffer()
    
    # Start periodic flush task
    flush_task = asyncio.create_task(periodic_flush())
    
    # Wait for shutdown signal
    while running:
        await asyncio.sleep(0.5)
    
    # Cleanup
    flush_task.cancel()
    
    # Closing servers
    sensor_server.close()
    semantic_map_server.close()
    await asyncio.gather(
        sensor_server.wait_closed(),
        semantic_map_server.wait_closed()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        exit(0)