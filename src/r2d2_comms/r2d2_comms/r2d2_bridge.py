import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import asyncio
import websockets
import json
import base64
import threading
import tf2_ros
from sensor_msgs.msg import CompressedImage, LaserScan, Imu, MagneticField, Temperature
from geometry_msgs.msg import PoseStamped, Twist
from tf2_msgs.msg import TFMessage
import tf_transformations
import numpy as np
import sys
import logging
import time
import socket
import cv2
import traceback

# --- Configuration ---
URI_SENSOR_STREAM = "ws://0.0.0.0:9001"
URI_CONTROL_STREAM = "ws://0.0.0.0:9003"
WS_PING_INTERVAL = 45     # Higher ping interval (seconds)
WS_PING_TIMEOUT = 30      # Longer ping timeout (seconds)
WS_MAX_SIZE = 10485760    # 10MB max message size
QUEUE_SIZE = 10           # Queue size for data buffers
TF_PUBLISH_RATE = 5.0     # Publish TF data at max 5 Hz
IMU_PUBLISH_RATE = 10.0   # Publish IMU data at max 10 Hz
CAMERA_PUBLISH_RATE = 2.0 # Publish camera at 2 Hz
IMAGE_RESIZE_FACTOR = 0.5 # Resize images to 50% before sending
IMAGE_QUALITY = 75        # JPEG quality (0-100)
CIRCUIT_BREAKER_ERRORS = 10 # Errors before circuit breaker trips
CIRCUIT_RESET_TIMEOUT = 30  # Seconds before resetting circuit breaker

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)

class NetworkQualityMonitor:
    """Monitors network quality and adjusts publishing rates"""
    def __init__(self):
        self.quality = 1.0  # 0.0 to 1.0 (1.0 is best)
        self.send_times = []  # Recent send times
        self.send_errors = 0
        self.last_reset = time.time()
        self.consecutive_timeouts = 0
        self.circuit_open = False
        self.circuit_reset_time = 0
        
    def record_send_time(self, duration):
        """Record a successful send operation"""
        self.send_times.append(duration)
        if len(self.send_times) > 20:
            self.send_times.pop(0)
        self._update_quality()
        
    def record_error(self, is_timeout=False):
        """Record a send error"""
        self.send_errors += 1
        if is_timeout:
            self.consecutive_timeouts += 1
            if self.consecutive_timeouts >= CIRCUIT_BREAKER_ERRORS:
                self.trip_circuit_breaker()
        else:
            self.consecutive_timeouts = 0
        self._update_quality()
        
    def record_success(self):
        """Record a successful operation"""
        self.consecutive_timeouts = 0
        
    def trip_circuit_breaker(self):
        """Trip the circuit breaker to prevent further network operations"""
        self.circuit_open = True
        self.circuit_reset_time = time.time() + CIRCUIT_RESET_TIMEOUT
        log.warning(f"Network circuit breaker tripped! Will reset in {CIRCUIT_RESET_TIMEOUT}s")
        
    def check_circuit_breaker(self):
        """Check if circuit breaker is open"""
        if self.circuit_open:
            current_time = time.time()
            if current_time > self.circuit_reset_time:
                self.circuit_open = False
                self.consecutive_timeouts = 0
                log.info("Network circuit breaker reset")
                return False
            return True
        return False
    
    def _update_quality(self):
        """Update network quality metric"""
        # Reset periodically to allow recovery
        current_time = time.time()
        if current_time - self.last_reset > 60:  # Reset every minute
            self.send_errors = max(0, self.send_errors - 5)  # Decay errors
            self.last_reset = current_time
            
        # Calculate quality from send times
        if self.send_times:
            avg_time = sum(self.send_times) / len(self.send_times)
            time_factor = 1.0 / (1.0 + avg_time * 10)  # Lower times = better quality
        else:
            time_factor = 0.5  # Default mid-range
            
        # Factor in errors
        error_factor = 1.0 / (1.0 + self.send_errors * 0.2)
        
        # Combine factors (weighted)
        self.quality = max(0.1, min(1.0, time_factor * 0.7 + error_factor * 0.3))
        
    def get_camera_interval(self):
        """Get adaptive camera interval based on network quality"""
        # Scale from 0.5s (at quality=1.0) to 2.0s (at quality=0.1)
        return 0.5 + (1.0 - self.quality) * 1.5

class R2D2BridgeClientNode(Node):
    """Bridge client connecting ROS2 to WebSocket server"""
    
    def __init__(self):
        super().__init__('r2d2_bridge_client')
        
        # TF buffer for robot pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Camera info
        self.cam_info = {
            'k': [615.7, 0.0, 320.5, 0.0, 615.7, 240.5, 0.0, 0.0, 1.0],
            'd': [0.0, 0.0, 0.0, 0.0, 0.0],
            'width': 640, 'height': 480
        }
        
        # Resized camera info
        scale = IMAGE_RESIZE_FACTOR
        self.cam_info_resized = {
            'k': [self.cam_info['k'][0]*scale, 0.0, self.cam_info['k'][2]*scale, 
                  0.0, self.cam_info['k'][4]*scale, self.cam_info['k'][5]*scale, 0.0, 0.0, 1.0],
            'd': self.cam_info['d'],
            'width': int(self.cam_info['width']*scale), 
            'height': int(self.cam_info['height']*scale)
        }

        # Connection status
        self.connection_status = {
            "sensor_connected": False,
            "control_connected": False,
            "last_sensor_connect": 0,
            "last_control_connect": 0,
            "sensor_errors": 0,
            "control_errors": 0,
            "messages_sent": 0,
            "messages_dropped": 0,
            "network_quality": 1.0
        }
        
        # Network quality monitor
        self.network_monitor = NetworkQualityMonitor()
        
        # Status timer
        self.create_timer(10.0, self.print_status)
        
        # Rate limiting for high-frequency data
        self.last_tf_publish_time = 0
        self.last_imu_publish_time = 0
        self.last_camera_publish_time = 0

        # --- ROS Subscriptions ---
        self.get_logger().info('Setting up sensor subscriptions...')
        
        # Camera subscription
        self.cam_sub = self.create_subscription(
            CompressedImage, 
            '/camera1/image_compressed', 
            self.camera_callback, 
            10
        )
        
        # Scan subscription
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback, 
            10
        )
        
        # IMU subscriptions
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10
        )
        
        self.bno055_imu_sub = self.create_subscription(
            Imu, '/bno055/imu', self.bno055_imu_callback, 10
        )
        
        self.bno055_mag_sub = self.create_subscription(
            MagneticField, '/bno055/mag', self.bno055_mag_callback, 10
        )
        
        self.bno055_temp_sub = self.create_subscription(
            Temperature, '/bno055/temp', self.bno055_temp_callback, 10
        )
        
        # TF subscription with throttling
        self.tf_sub = self.create_subscription(
            TFMessage, '/tf', self.tf_callback, 10
        )

        # --- Publishers ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        self.goal_pose_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # --- Data Queues ---
        self.camera_queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        self.scan_queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        self.imu_queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        self.tf_queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        self.other_sensors_queue = asyncio.Queue(maxsize=QUEUE_SIZE*2)
        
        # Robot pose tracking
        self.last_robot_pose = {"x": 0.0, "y": 0.0, "theta": 0.0, "frame": "fallback"}
        self.create_timer(0.1, self.update_robot_pose)
        
        # Running flag
        self.running = True
        
        # Start asyncio thread
        self.asyncio_thread = threading.Thread(target=self.run_asyncio_loop, daemon=True)
        self.asyncio_thread.start()
        
        self.get_logger().info('R2D2 Bridge Client initialized')

    def print_status(self):
        """Print connection status"""
        now = time.time()
        sensor_age = now - self.connection_status["last_sensor_connect"]
        control_age = now - self.connection_status["last_control_connect"]
        
        self.get_logger().info(
            f"Status: Sensor={self.connection_status['sensor_connected']} "
            f"({sensor_age:.1f}s), Control={self.connection_status['control_connected']} "
            f"({control_age:.1f}s), Quality={self.network_monitor.quality:.2f}, "
            f"Msgs: {self.connection_status['messages_sent']}, "
            f"Dropped: {self.connection_status['messages_dropped']}"
        )
        
        # Update camera rate based on network quality
        camera_interval = self.network_monitor.get_camera_interval()
        camera_fps = 1.0 / camera_interval
        self.get_logger().info(f"Network adaptive rates - Camera: {camera_fps:.1f} fps")

    def run_asyncio_loop(self):
        """Run asyncio event loop in separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Create main tasks
        main_task = asyncio.gather(
            self.run_sensor_publisher(),
           # self.run_control_listener()
        )
        
        try:
            loop.run_until_complete(main_task)
        except asyncio.CancelledError:
            pass
        finally:
            loop.close()

    def update_robot_pose(self):
        """Update robot pose from TF"""
        try:
            # Try map frame first
            try:
                trans = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
                frame = 'map'
            except tf2_ros.TransformException:
                # Fall back to odom frame
                trans = self.tf_buffer.lookup_transform('odom', 'base_footprint', rclpy.time.Time())
                frame = 'odom'
            
            # Get yaw from quaternion
            q = (trans.transform.rotation.x, trans.transform.rotation.y, 
                 trans.transform.rotation.z, trans.transform.rotation.w)
            _, _, yaw = tf_transformations.euler_from_quaternion(q)
            
            # Update pose
            self.last_robot_pose = {
                "x": float(trans.transform.translation.x),
                "y": float(trans.transform.translation.y),
                "theta": float(yaw),
                "frame": frame
            }
        except Exception as e:
            # Use existing pose if lookup fails
            pass

    def convert_numpy_types(self, obj):
        """Convert NumPy types to Python native types"""
        if isinstance(obj, dict):
            return {k: self.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_types(item) for item in obj)
        elif isinstance(obj, np.ndarray):
            return [self.convert_numpy_types(item) for item in obj.tolist()]
        elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            try:
                return float(obj) if '.' in str(obj) else int(obj)
            except:
                return str(obj)

    def add_to_queue(self, queue, data, data_type):
        """Add data to a queue, handling queue full condition"""
        try:
            if queue.full():
                queue.get_nowait()  # Remove oldest item
                self.connection_status["messages_dropped"] += 1
            queue.put_nowait(data)
            self.get_logger().debug(f'{data_type} queued')
        except Exception as e:
            self.get_logger().error(f"Queue error: {e}")

    def camera_callback(self, msg: CompressedImage):
        """Process camera data with rate limiting and resizing"""
        # Apply rate limiting based on network quality
        current_time = time.time()
        camera_interval = self.network_monitor.get_camera_interval()
        
        if current_time - self.last_camera_publish_time < camera_interval:
            return  # Skip this message
            
        self.last_camera_publish_time = current_time
        
        try:
            # Decode compressed image
            np_arr = np.frombuffer(bytes(msg.data), np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is None:
                self.get_logger().error("Failed to decode camera image")
                return
                
            # Resize to reduce bandwidth
            if IMAGE_RESIZE_FACTOR != 1.0:
                height, width = img.shape[:2]
                new_height = int(height * IMAGE_RESIZE_FACTOR)
                new_width = int(width * IMAGE_RESIZE_FACTOR)
                img_resized = cv2.resize(img, (new_width, new_height))
            else:
                img_resized = img
            
            # Re-encode with specified quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), IMAGE_QUALITY]
            _, buffer = cv2.imencode('.jpg', img_resized, encode_param)
            
            # Convert to base64
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Create data object
            data = {
                "type": "camera_data",
                "timestamp": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9,
                "image_base64": image_base64,
                "camera_info": self.cam_info_resized,
                "robot_pose": self.last_robot_pose,
            }
            
            # Convert NumPy types
            data = self.convert_numpy_types(data)
            
            # Add to queue
            self.add_to_queue(self.camera_queue, data, "Camera data")
            self.get_logger().info('Camera data queued', throttle_duration_sec=2.0)
            
        except Exception as e:
            self.get_logger().error(f"Error processing camera: {e}")
            traceback.print_exc()

    def scan_callback(self, msg: LaserScan):
        """Process scan data"""
        # Convert scan ranges to simple Python list
        ranges_list = []
        for r in msg.ranges:
            if np.isnan(r):
                ranges_list.append(0.0)  # Convert NaN to 0.0
            elif np.isinf(r) and r > 0:
                ranges_list.append(99.0)  # Convert +Inf to 99.0
            elif np.isinf(r) and r < 0:
                ranges_list.append(0.0)   # Convert -Inf to 0.0
            else:
                ranges_list.append(float(r))
        
        data = {
            "type": "scan_data",
            "timestamp": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9,
            "robot_pose": self.last_robot_pose,
            "scan": {
                "angle_min": float(msg.angle_min),
                "angle_max": float(msg.angle_max),
                "angle_increment": float(msg.angle_increment),
                "ranges": ranges_list
            }
        }
        
        # Add to queue
        self.add_to_queue(self.scan_queue, data, "Scan data")
        self.get_logger().info('Scan data queued', throttle_duration_sec=2.0)

    def imu_callback(self, msg: Imu):
        """Process IMU data with rate limiting"""
        # Apply rate limiting
        current_time = time.time()
        if current_time - self.last_imu_publish_time < 1.0/IMU_PUBLISH_RATE:
            return  # Skip this message
        
        self.last_imu_publish_time = current_time
        
        # Create IMU data
        data = {
            "type": "imu_data",
            "timestamp": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9,
            "robot_pose": self.last_robot_pose,
            "angular_velocity": {
                "x": float(msg.angular_velocity.x),
                "y": float(msg.angular_velocity.y),
                "z": float(msg.angular_velocity.z)
            },
            "linear_acceleration": {
                "x": float(msg.linear_acceleration.x),
                "y": float(msg.linear_acceleration.y),
                "z": float(msg.linear_acceleration.z)
            },
            "orientation": {
                "x": float(msg.orientation.x),
                "y": float(msg.orientation.y),
                "z": float(msg.orientation.z),
                "w": float(msg.orientation.w)
            }
        }
        
        # Add to queue
        self.add_to_queue(self.imu_queue, data, "IMU data")

    def bno055_imu_callback(self, msg: Imu):
        """Process BNO055 IMU data with rate limiting"""
        # Apply rate limiting
        current_time = time.time()
        if current_time - self.last_imu_publish_time < 1.0/IMU_PUBLISH_RATE:
            return  # Skip this message
        
        # Create data bundle
        data = {
            "type": "bno055_imu_data",
            "timestamp": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9,
            "robot_pose": self.last_robot_pose,
            "angular_velocity": {
                "x": float(msg.angular_velocity.x),
                "y": float(msg.angular_velocity.y),
                "z": float(msg.angular_velocity.z)
            },
            "linear_acceleration": {
                "x": float(msg.linear_acceleration.x),
                "y": float(msg.linear_acceleration.y),
                "z": float(msg.linear_acceleration.z)
            },
            "orientation": {
                "x": float(msg.orientation.x),
                "y": float(msg.orientation.y),
                "z": float(msg.orientation.z),
                "w": float(msg.orientation.w)
            }
        }
        
        # Add to queue
        self.add_to_queue(self.other_sensors_queue, data, "BNO055 IMU data")

    def bno055_mag_callback(self, msg: MagneticField):
        """Process BNO055 magnetometer data with rate limiting"""
        # Apply rate limiting
        current_time = time.time()
        if current_time - self.last_imu_publish_time < 1.0/IMU_PUBLISH_RATE:
            return  # Skip this message
        
        data = {
            "type": "bno055_mag_data",
            "timestamp": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9,
            "robot_pose": self.last_robot_pose,
            "magnetic_field": {
                "x": float(msg.magnetic_field.x),
                "y": float(msg.magnetic_field.y),
                "z": float(msg.magnetic_field.z)
            }
        }
        
        # Add to queue
        self.add_to_queue(self.other_sensors_queue, data, "BNO055 mag data")

    def bno055_temp_callback(self, msg: Temperature):
        """Process BNO055 temperature data"""
        data = {
            "type": "bno055_temp_data",
            "timestamp": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9,
            "robot_pose": self.last_robot_pose,
            "temperature": float(msg.temperature)
        }
        
        # Add to queue
        self.add_to_queue(self.other_sensors_queue, data, "BNO055 temp data")

    def tf_callback(self, msg: TFMessage):
        """Process transform frames data with rate limiting"""
        # Apply rate limiting
        current_time = time.time()
        if current_time - self.last_tf_publish_time < 1.0/TF_PUBLISH_RATE:
            return  # Skip this message
        
        self.last_tf_publish_time = current_time
        
        # Convert to simplified format
        transforms = []
        for transform in msg.transforms:
            transforms.append({
                "header": {
                    "frame_id": transform.header.frame_id,
                    "stamp": float(transform.header.stamp.sec) + float(transform.header.stamp.nanosec) / 1e9
                },
                "child_frame_id": transform.child_frame_id,
                "transform": {
                    "translation": {
                        "x": float(transform.transform.translation.x),
                        "y": float(transform.transform.translation.y),
                        "z": float(transform.transform.translation.z)
                    },
                    "rotation": {
                        "x": float(transform.transform.rotation.x),
                        "y": float(transform.transform.rotation.y),
                        "z": float(transform.transform.rotation.z),
                        "w": float(transform.transform.rotation.w)
                    }
                }
            })
        
        data = {
            "type": "tf_data",
            "timestamp": float(msg.transforms[0].header.stamp.sec) + float(msg.transforms[0].header.stamp.nanosec) / 1e9 if msg.transforms else time.time(),
            "robot_pose": self.last_robot_pose,
            "transforms": transforms
        }
        
        # Add to queue
        self.add_to_queue(self.tf_queue, data, "TF data")
        self.get_logger().info(f'TF data queued ({len(transforms)} transforms)', throttle_duration_sec=5.0)

    async def _flush_queues(self):
        """Clear all queues when reconnecting"""
        queues_cleared = 0
        
        # Clear all queues
        while not self.camera_queue.empty():
            await self.camera_queue.get()
            queues_cleared += 1
        
        while not self.scan_queue.empty():
            await self.scan_queue.get()
            queues_cleared += 1
            
        while not self.imu_queue.empty():
            await self.imu_queue.get()
            queues_cleared += 1
            
        while not self.tf_queue.empty():
            await self.tf_queue.get()
            queues_cleared += 1
            
        while not self.other_sensors_queue.empty():
            await self.other_sensors_queue.get()
            queues_cleared += 1
        
        if queues_cleared > 0:
            self.get_logger().warn(f"Cleared {queues_cleared} backlogged messages")

    async def _send_data_with_monitoring(self, websocket, data):
        """Send data with monitoring of network performance"""
        if self.network_monitor.check_circuit_breaker():
            self.get_logger().warning("Network circuit breaker open, not sending data")
            return False
            
        try:
            start_time = time.time()
            await websocket.send(json.dumps(data))
            send_time = time.time() - start_time
            
            # Record successful send
            self.network_monitor.record_send_time(send_time)
            self.network_monitor.record_success()
            self.connection_status["messages_sent"] += 1
            
            return True
        except websockets.exceptions.ConnectionClosedError as e:
            is_timeout = "timeout" in str(e).lower()
            self.network_monitor.record_error(is_timeout)
            raise
        except Exception as e:
            self.network_monitor.record_error(False)
            raise

    async def run_sensor_publisher(self):
        """Publish sensor data to WebSocket server"""
        self.get_logger().info(f'Connecting to sensor server: {URI_SENSOR_STREAM}')
        
        reconnect_delay = 1
        max_reconnect_delay = 15
        
        while self.running:
            try:
                self.get_logger().info('Attempting sensor connection...')
                async with websockets.connect(
                    URI_SENSOR_STREAM,
                    ping_interval=WS_PING_INTERVAL,
                    ping_timeout=WS_PING_TIMEOUT,
                    max_size=WS_MAX_SIZE,
                    close_timeout=5.0  # Don't hang on close
                ) as websocket:
                    # Connection successful
                    reconnect_delay = 1
                    self.get_logger().info('Sensor connection established')
                    self.connection_status["sensor_connected"] = True
                    self.connection_status["last_sensor_connect"] = time.time()
                    
                    # Enable TCP_NODELAY
                    try:
                        socket_obj = websocket.transport.get_extra_info('socket')
                        if socket_obj:
                            socket_obj.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                            self.get_logger().info('TCP_NODELAY enabled for sensor WebSocket')
                    except Exception as e:
                        self.get_logger().warn(f'Could not set TCP_NODELAY: {e}')
                    
                    # Clear backlog
                    await self._flush_queues()
                    
                    # Process queues
                    while True:
                        # Check if circuit breaker is open
                        if self.network_monitor.check_circuit_breaker():
                            self.get_logger().warning("Network circuit breaker open, pausing data transmission")
                            await asyncio.sleep(1.0)
                            continue
                        
                        # Create tasks for all queues with timeout
                        camera_task = asyncio.create_task(self.camera_queue.get())
                        scan_task = asyncio.create_task(self.scan_queue.get())
                        imu_task = asyncio.create_task(self.imu_queue.get())
                        tf_task = asyncio.create_task(self.tf_queue.get())
                        other_task = asyncio.create_task(self.other_sensors_queue.get())
                        
                        # Wait for any task to complete
                        done, pending = await asyncio.wait(
                            [camera_task, scan_task, imu_task, tf_task, other_task],
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=1.0
                        )
                        
                        # Cancel all pending tasks
                        for task in pending:
                            task.cancel()
                        
                        if not self.running:
                            break
                            
                        if not done:
                            continue
                            
                        # Process completed tasks
                        for task in done:
                            try:
                                data = task.result()
                                # Use the monitored send function
                                await self._send_data_with_monitoring(websocket, data)
                                self.get_logger().info(f"{data['type']} sent", throttle_duration_sec=2.0)
                            except Exception as e:
                                self.get_logger().error(f"Error sending data: {e}")
                                # If we get too many errors, circuit breaker will activate
                        
            except websockets.exceptions.ConnectionClosedError as e:
                self.connection_status["sensor_connected"] = False
                self.connection_status["sensor_errors"] += 1
                is_timeout = "timeout" in str(e).lower()
                if is_timeout:
                    self.network_monitor.record_error(True)  # Record as timeout
                
                self.get_logger().error(f'Connection closed: {repr(e)}. Reconnecting in {reconnect_delay}s...')
            except Exception as e:
                self.connection_status["sensor_connected"] = False
                self.connection_status["sensor_errors"] += 1
                self.get_logger().error(f'Sensor connection error: {repr(e)}. Reconnecting in {reconnect_delay}s...')
                traceback.print_exc()
            
            # Use exponential backoff for reconnection
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)

    async def _send_heartbeat(self, websocket):
        """Send periodic heartbeat messages"""
        try:
            while True:
                await asyncio.sleep(10)  # Every 10 seconds
                if not self.running:
                    break
                
                # Skip if circuit breaker is open
                if self.network_monitor.check_circuit_breaker():
                    continue
                    
                try:
                    # Simple heartbeat message
                    heartbeat = {
                        "type": "heartbeat",
                        "timestamp": time.time(),
                        "client_id": "r2d2_bridge"
                    }
                    await self._send_data_with_monitoring(websocket, heartbeat)
                    self.get_logger().debug('Heartbeat sent')
                except Exception as e:
                    self.get_logger().error(f'Heartbeat error: {e}')
                    break
        except asyncio.CancelledError:
            # Normal cancellation during cleanup
            pass

    async def run_control_listener(self):
        """Listen for control commands from the server"""
        self.get_logger().info(f'Connecting to control server: {URI_CONTROL_STREAM}')
        
        reconnect_delay = 1
        max_reconnect_delay = 15
        
        while self.running:
            try:
                async with websockets.connect(
                    URI_CONTROL_STREAM,
                    ping_interval=WS_PING_INTERVAL,
                    ping_timeout=WS_PING_TIMEOUT,
                    max_size=WS_MAX_SIZE,
                    close_timeout=5.0  # Don't hang on close
                ) as websocket:
                    self.get_logger().info('Control connection established')
                    self.connection_status["control_connected"] = True
                    self.connection_status["last_control_connect"] = time.time()
                    
                    # Reset reconnect delay
                    reconnect_delay = 1
                    
                    # Enable TCP_NODELAY
                    try:
                        socket_obj = websocket.transport.get_extra_info('socket')
                        if socket_obj:
                            socket_obj.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                            self.get_logger().info('TCP_NODELAY enabled for control WebSocket')
                    except Exception as e:
                        self.get_logger().warn(f'Could not set TCP_NODELAY: {e}')
                    
                    # Start heartbeat task
                    heartbeat_task = asyncio.create_task(self._send_heartbeat(websocket))
                    
                    try:
                        async for message in websocket:
                            try:
                                command = json.loads(message)
                                self.process_command(command)
                            except json.JSONDecodeError:
                                self.get_logger().error('Received invalid JSON command')
                            except Exception as e:
                                self.get_logger().error(f'Error processing command: {e}')
                                traceback.print_exc()
                    finally:
                        # Cancel heartbeat when connection ends
                        heartbeat_task.cancel()
                                
            except websockets.exceptions.ConnectionClosedError as e:
                self.connection_status["control_connected"] = False
                self.connection_status["control_errors"] += 1
                is_timeout = "timeout" in str(e).lower()
                if is_timeout:
                    self.network_monitor.record_error(True)
                    
                self.get_logger().error(f'Control connection closed: {repr(e)}. Reconnecting in {reconnect_delay}s...')
            except Exception as e:
                self.connection_status["control_connected"] = False
                self.connection_status["control_errors"] += 1
                self.get_logger().error(f'Control connection error: {repr(e)}. Reconnecting in {reconnect_delay}s...')
                traceback.print_exc()
            
            # Use exponential backoff for reconnection
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)

    def process_command(self, command):
        """Process commands received from the server with validation"""
        try:
            cmd_type = command.get('type')
            if not cmd_type:
                self.get_logger().warn("Received command without type field")
                return
                
            if cmd_type == 'brain_response':
                # Process motor commands
                if 'motor_commands' in command:
                    motor_cmd = command['motor_commands']
                    motor_type = motor_cmd.get('type', '')
                    
                    if motor_type == 'CMD_VEL':
                        # Validate and apply safety limits
                        linear_x = float(motor_cmd.get('linear_x', 0.0))
                        angular_z = float(motor_cmd.get('angular_z', 0.0))
                        
                        # Safety limits
                        MAX_LINEAR = 0.5  # m/s
                        MAX_ANGULAR = 1.0  # rad/s
                        
                        linear_x = max(-MAX_LINEAR, min(MAX_LINEAR, linear_x))
                        angular_z = max(-MAX_ANGULAR, min(MAX_ANGULAR, angular_z))
                        
                        # Create Twist message
                        msg = Twist()
                        msg.linear.x = linear_x
                        msg.angular.z = angular_z
                        self.cmd_vel_pub.publish(msg)
                        self.get_logger().info(f"Published CMD_VEL: lin={msg.linear.x}, ang={msg.angular.z}")
                    
                    elif motor_type == 'Diffdrive':
                        # Handle omni-wheel commands with validation
                        motor_matrix = motor_cmd.get('matrix', [])
                        if isinstance(motor_matrix, list) and len(motor_matrix) == 4:
                            self.get_logger().info(f"Received omni-wheel commands: {motor_matrix}")
                            # Implementation depends on your motor controller interface
                        else:
                            self.get_logger().warn(f"Invalid omni-wheel matrix format: {motor_matrix}")
                
                # Process audio response
                if 'audio_response' in command:
                    audio_data = command['audio_response']
                    text = audio_data.get('text', '')
                    emotion = audio_data.get('emotions', 'neutral')
                    self.get_logger().info(f"Audio response: '{text}' (emotion: {emotion})")
                    # Implementation depends on your audio system
            
            elif cmd_type == 'NAV_GOAL':
                # Set navigation goal with validation
                try:
                    pose = command['pose']
                    if not all(k in pose for k in ['x', 'y', 'theta']):
                        self.get_logger().error(f"Invalid NAV_GOAL pose format: {pose}")
                        return
                        
                    msg = PoseStamped()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = 'map'
                    msg.pose.position.x = float(pose['x'])
                    msg.pose.position.y = float(pose['y'])
                    
                    q = tf_transformations.quaternion_from_euler(0.0, 0.0, float(pose['theta']))
                    msg.pose.orientation.x = q[0]
                    msg.pose.orientation.y = q[1]
                    msg.pose.orientation.z = q[2]
                    msg.pose.orientation.w = q[3]
                    
                    self.goal_pose_pub.publish(msg)
                    self.get_logger().info(f"Published navigation goal: ({pose['x']}, {pose['y']}, {pose['theta']})")
                except Exception as e:
                    self.get_logger().error(f'Failed to parse NAV_GOAL: {e}')
                    traceback.print_exc()
            
            else:
                self.get_logger().warn(f"Received unknown command type: {cmd_type}")
        
        except Exception as e:
            self.get_logger().error(f"Error processing command: {e}")
            traceback.print_exc()

def main(args=None):
    rclpy.init(args=args)
    node = R2D2BridgeClientNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
     
    # Shutdown handler
    def clean_shutdown(signum=None, frame=None):
        node.get_logger().info('Shutting down bridge...')
        node.running = False
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        
    # Register signal handlers
    import signal
    signal.signal(signal.SIGINT, clean_shutdown)
    signal.signal(signal.SIGTERM, clean_shutdown)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        clean_shutdown()

if __name__ == '__main__':
    main()