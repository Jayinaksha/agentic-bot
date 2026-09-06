#!/usr/bin/env python3
"""
VLA perception node: NVIDIA Cosmos Reason 2 grounding the camera into the map.

Cosmos Reason 2 is an open reasoning VLM built for physical AI, and it does the
one thing this stack needs that a generic captioner does not: it returns object
localisation - 2D boxes with labels and an explanation - alongside its
reasoning about the scene. That is exactly the input grounding.py needs.

Where it runs
-------------
Not on the robot. The model is served behind an OpenAI-compatible endpoint,
either NVIDIA's hosted catalogue or your own NIM/vLLM deployment on a cloud GPU:

    R2D2_VLA_BASE_URL=https://integrate.api.nvidia.com/v1
    R2D2_VLA_MODEL=nvidia/cosmos-reason2-8b
    R2D2_NVIDIA_API_KEY=nvapi-...

A 2B variant (nvidia/cosmos-reason2-2b) exists and is worth trying first: this
node asks for object boxes and a one-line scene description, not open-ended
reasoning, and the smaller model is markedly cheaper per frame. Raise
min_interval before reaching for a bigger model - most of the value here comes
from looking carefully at a handful of places, not from looking continuously.

    # or self-hosted, e.g. a Nebius or GCP GPU VM running the NIM container
    R2D2_VLA_BASE_URL=http://10.0.0.5:8000/v1

Only outbound HTTPS is needed, so Gazebo and the whole ROS graph stay on the
local machine and nothing has to be exposed inbound.

Rate and cost
-------------
Frames are expensive, in latency and in tokens. This node therefore does not
stream: it processes a frame when

    * a periodic budget allows it (min_interval), AND
    * the robot has moved far enough to be looking at something new, OR
    * something explicitly asked it to look (/perception/observe)

A robot parked in a corridor should cost nothing. Most of the value of a VLA in
a house comes from looking carefully at a handful of places, not from watching
continuously.

Failure behaviour
-----------------
The endpoint is remote and will sometimes be slow or down. Requests run on a
worker thread with a timeout; on failure the node logs, backs off, and keeps
running. Perception degrading must never take navigation with it.
"""

from __future__ import annotations

import base64
import json
import math
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from std_msgs.msg import Int32, String

from r2d2_perception.grounding import (extract_json, ground_detection,
                                       parse_vla_detections)

# The prompt is written for a model that reasons about physical scenes. It asks
# for exactly the fields grounding.py consumes and nothing else: every extra
# field is tokens spent on something no consumer reads.
GROUNDING_PROMPT = """You are the vision system of a small indoor robot driving through a house.

Look at this image and report the objects that matter to a robot that has to
navigate around them, find things, and describe rooms to a person.

Return ONLY a JSON object, no prose and no markdown fence:

{
  "room": "your best guess at which room this is, or null",
  "scene": "one sentence describing what the robot is looking at",
  "objects": [
    {
      "label": "short noun, lower case, e.g. chair, doorway, staircase",
      "description": "brief physical description that distinguishes this one",
      "box_2d": [x_min, y_min, x_max, y_max],
      "confidence": 0.0 to 1.0,
      "traversability": "clear" | "obstacle" | "hazard"
    }
  ]
}

Rules:
- box_2d coordinates are fractions of the image width and height, in [0, 1].
- Report furniture, appliances, doorways, stairs and people. Ignore wall
  texture, flooring and lighting.
- Mark a staircase, a balcony edge or a step down as "hazard".
- If you are unsure what something is, give a lower confidence rather than
  omitting it.
"""


class VlaNode(Node):

    def __init__(self):
        super().__init__('vla_node')

        self.declare_parameters('', [
            ('base_url', os.environ.get('R2D2_VLA_BASE_URL',
                                        'https://integrate.api.nvidia.com/v1')),
            ('model', os.environ.get('R2D2_VLA_MODEL', 'nvidia/cosmos-reason2-8b')),
            ('api_key_env', 'R2D2_NVIDIA_API_KEY'),
            ('min_interval', 6.0),        # s between automatic frames
            ('min_travel', 0.8),          # m of movement to justify a new frame
            ('min_rotation', 0.6),        # rad of turning, likewise
            ('request_timeout', 45.0),
            ('max_tokens', 1024),
            ('temperature', 0.1),         # near-greedy: this is extraction work
            ('jpeg_quality', 80),
            ('image_max_width', 640),
            ('hfov', 1.089),
            ('camera_offset_x', 0.18),
            ('scan_range_max', 12.0),
            ('min_confidence', 0.35),
            ('backoff_max', 120.0),
        ])
        g = self.get_parameter
        self.base_url = g('base_url').value.rstrip('/')
        self.model = g('model').value
        self.api_key = os.environ.get(g('api_key_env').value, '')
        self.min_interval = g('min_interval').value
        self.min_travel = g('min_travel').value
        self.min_rotation = g('min_rotation').value
        self.timeout = g('request_timeout').value
        self.max_tokens = g('max_tokens').value
        self.temperature = g('temperature').value
        self.jpeg_quality = g('jpeg_quality').value
        self.image_max_width = g('image_max_width').value
        self.hfov = g('hfov').value
        self.camera_offset_x = g('camera_offset_x').value
        self.scan_range_max = g('scan_range_max').value
        self.min_confidence = g('min_confidence').value
        self.backoff_max = g('backoff_max').value

        if not self.api_key and 'integrate.api.nvidia.com' in self.base_url:
            self.get_logger().warn(
                'no API key found: set R2D2_NVIDIA_API_KEY for the hosted '
                'endpoint, or point base_url at your own NIM deployment.')

        self._frame = None
        self._frame_stamp = 0.0
        self._scan: Optional[LaserScan] = None
        self._pose = (0.0, 0.0, 0.0)
        self._floor = 0

        self._last_request = 0.0
        self._last_pose_at_request = (0.0, 0.0, 0.0)
        self._backoff = 0.0
        self._consecutive_failures = 0
        self._requests = 0
        self._observe_queue: "queue.Queue[str]" = queue.Queue(maxsize=4)

        sensor_qos = QoSProfile(depth=1,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(Image, '/camera/image_raw', self._on_image, sensor_qos)
        self.create_subscription(CompressedImage, '/camera/image_raw/compressed',
                                 self._on_compressed, sensor_qos)
        self.create_subscription(LaserScan, '/scan', self._on_scan, sensor_qos)
        self.create_subscription(Odometry, '/odometry/filtered', self._on_odom, 10)
        self.create_subscription(Int32, '/floor/current', self._on_floor, 10)
        self.create_subscription(String, '/perception/observe', self._on_observe, 10)

        self.detections_pub = self.create_publisher(String, '/perception/detections', 10)
        self.scene_pub = self.create_publisher(String, '/perception/scene', 10)

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self.create_timer(1.0, self._maybe_observe)
        self.get_logger().info(
            f'VLA node up: {self.model} at {self.base_url}, '
            f'auto frame every {self.min_interval:.0f} s when moving')

    # ---------------------------------------------------------------- inputs

    def _on_image(self, msg: Image):
        self._frame = ('raw', msg)
        self._frame_stamp = time.time()

    def _on_compressed(self, msg: CompressedImage):
        self._frame = ('compressed', msg)
        self._frame_stamp = time.time()

    def _on_scan(self, msg: LaserScan):
        self._scan = msg

    def _on_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def _on_floor(self, msg: Int32):
        self._floor = msg.data

    def _on_observe(self, msg: String):
        """Explicit request, from the MCP `look` tool. Bypasses the movement
        gate but not the queue: a burst of requests still runs one at a time."""
        try:
            self._observe_queue.put_nowait(msg.data or GROUNDING_PROMPT)
        except queue.Full:
            self.get_logger().warn('observe request dropped: worker is busy')

    # ------------------------------------------------------------- scheduling

    def _maybe_observe(self):
        """Decide whether an automatic frame is worth its cost."""
        if not self._observe_queue.empty():
            return                                 # explicit request pending
        now = time.time()
        if now < self._backoff:
            return
        if now - self._last_request < self.min_interval:
            return
        if self._frame is None:
            return

        dx = math.dist(self._pose[:2], self._last_pose_at_request[:2])
        dyaw = abs(_wrap(self._pose[2] - self._last_pose_at_request[2]))
        if dx < self.min_travel and dyaw < self.min_rotation:
            # Same viewpoint as last time: a fresh frame would cost a request
            # and tell us what we already know.
            return

        try:
            self._observe_queue.put_nowait(GROUNDING_PROMPT)
        except queue.Full:
            pass

    # ----------------------------------------------------------------- worker

    def _worker_loop(self):
        while True:
            prompt = self._observe_queue.get()
            try:
                self._process(prompt)
            except Exception as exc:              # noqa: BLE001 - never die
                self.get_logger().error(f'VLA processing failed: {exc}')
                self._note_failure()
            finally:
                self._observe_queue.task_done()

    def _process(self, prompt: str):
        frame = self._frame
        scan = self._scan
        if frame is None:
            self.get_logger().warn('no camera frame yet')
            return

        jpeg = self._encode(frame)
        if jpeg is None:
            return

        pose = self._pose
        floor = self._floor
        self._last_request = time.time()
        self._last_pose_at_request = pose
        self._requests += 1

        started = time.time()
        payload = self._call_vla(jpeg, prompt)
        elapsed = time.time() - started
        if payload is None:
            self._note_failure()
            return

        self._consecutive_failures = 0
        self._backoff = 0.0

        detections = parse_vla_detections(payload)
        grounded: List[Dict[str, Any]] = []
        for detection in detections:
            if detection.confidence < self.min_confidence:
                continue
            if scan is None:
                continue
            result = ground_detection(
                detection, scan.ranges, scan.angle_min, scan.angle_increment,
                robot_pose=pose, hfov=self.hfov,
                camera_offset_x=self.camera_offset_x,
                scan_range_max=min(self.scan_range_max, scan.range_max))
            grounded.append(result.as_dict())

        out = String()
        out.data = json.dumps({
            'stamp': time.time(),
            'floor': floor,
            'robot_pose': {'x': pose[0], 'y': pose[1], 'yaw': pose[2]},
            'source': self.model,
            'latency_s': round(elapsed, 2),
            'detections': grounded,
        })
        self.detections_pub.publish(out)

        scene = String()
        scene.data = json.dumps({
            'stamp': time.time(),
            'room': payload.get('room'),
            'scene': payload.get('scene'),
            'floor': floor,
            'object_count': len(grounded),
        })
        self.scene_pub.publish(scene)

        ranged = sum(1 for d in grounded if 'world_x' in d)
        self.get_logger().info(
            f'VLA frame {self._requests}: {len(grounded)} objects '
            f'({ranged} ranged) in {elapsed:.1f} s - '
            f'{payload.get("scene", "")[:70]}')

    def _note_failure(self):
        self._consecutive_failures += 1
        # Exponential backoff, capped. A dead endpoint should stop costing
        # requests quickly, and recover without a restart.
        delay = min(self.backoff_max, 5.0 * (2 ** (self._consecutive_failures - 1)))
        self._backoff = time.time() + delay
        self.get_logger().warn(
            f'VLA unavailable ({self._consecutive_failures} in a row); '
            f'backing off {delay:.0f} s. Navigation is unaffected.')

    # ------------------------------------------------------------- encoding

    def _encode(self, frame) -> Optional[str]:
        """Frame to a base64 JPEG, downscaled to keep the uplink cheap."""
        kind, msg = frame
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.get_logger().error('opencv-python is required to encode frames')
            return None

        if kind == 'compressed':
            buffer = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        else:
            channels = max(1, msg.step // max(msg.width, 1))
            array = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            try:
                image = array.reshape(msg.height, msg.width, channels)
            except ValueError:
                self.get_logger().warn(
                    f'unexpected image layout: {msg.encoding} '
                    f'{msg.width}x{msg.height} step {msg.step}')
                return None
            if msg.encoding in ('rgb8', 'rgba8'):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if image is None:
            return None
        if image.shape[1] > self.image_max_width:
            scale = self.image_max_width / image.shape[1]
            image = cv2.resize(image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode(
            '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return None
        return base64.b64encode(encoded.tobytes()).decode('ascii')

    # ---------------------------------------------------------------- request

    def _call_vla(self, jpeg_b64: str, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            import httpx
        except ImportError:
            self.get_logger().error('httpx is required to reach the VLA endpoint')
            return None

        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        # NIM for VLMs follows the OpenAI spec for images: a content list with
        # an image_url block carrying a data URI. Verified against NVIDIA's
        # Cosmos Reason 2 API reference rather than assumed - some inference
        # servers instead expect an <img> tag inline in the text, and the two
        # fail differently enough to be worth pinning down.
        # JPG, JPEG and PNG are the supported encodings; _encode emits JPEG.
        body = {
            'model': self.model,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url',
                     'image_url': {'url': f'data:image/jpeg;base64,{jpeg_b64}'}},
                ],
            }],
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f'{self.base_url}/chat/completions',
                                       headers=headers, json=body)
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content']
        except Exception as exc:                  # noqa: BLE001 - many failure modes
            self.get_logger().warn(f'VLA request failed: {exc}')
            return None

        return extract_json(content)


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def main(args=None):
    rclpy.init(args=args)
    node = VlaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
