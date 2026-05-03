#!/usr/bin/env python3
"""
R2D2 Navigation Bridge

Bridges between the GPT-OSS planner on the supercomputer and ROS2 Nav2.
Provides HTTP endpoints to receive navigation goals and speech commands.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import threading
from flask import Flask, request, jsonify
import tf_transformations
import math
import time
import os

class R2D2NavigationBridge(Node):
    def __init__(self):
        super().__init__('r2d2_navigation_bridge')
        
        # Create action client for Nav2
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().info('Waiting for Nav2 action server...')
        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().warn('Nav2 action server not available after 10 seconds')
        else:
            self.get_logger().info('Nav2 action server is available')
        
        # Create publisher for TTS
        self.tts_publisher = self.create_publisher(String, 'tts_text', 10)
        
        # Statuses
        self.current_goal = None
        self.current_status = "idle"
        self.last_command_time = time.time()
        self.start_time = time.time()
        
        # Start Flask server in a separate thread
        self.flask_app = Flask(__name__)
        self.setup_flask_routes()
        self.flask_thread = threading.Thread(target=self.run_flask_server)
        self.flask_thread.daemon = True
        self.flask_thread.start()
        
        self.get_logger().info('R2D2 Navigation Bridge ready on port 8888')
    
    def setup_flask_routes(self):
        """Set up Flask routes"""
        
        @self.flask_app.route('/navigate', methods=['POST'])
        def navigate():
            try:
                data = request.json
                x = float(data.get('x', 0.0))
                y = float(data.get('y', 0.0))
                theta = float(data.get('theta', 0.0))
                description = data.get('description', 'Navigation goal')
                
                self.get_logger().info(f'Received navigation request: {description} ({x}, {y}, {theta})')
                
                # Send to Nav2
                goal_future = self.send_goal(x, y, theta)
                self.current_status = "navigating"
                self.last_command_time = time.time()
                
                return jsonify({
                    'status': 'success', 
                    'message': f'Goal sent: {description}',
                    'coordinates': {'x': x, 'y': y, 'theta': theta}
                })
            except Exception as e:
                self.get_logger().error(f'Error processing navigation request: {e}')
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.flask_app.route('/speak', methods=['POST'])
        def speak():
            try:
                data = request.json
                text = data.get('text', '')
                
                self.get_logger().info(f'Received speech request: "{text}"')
                
                # Publish to TTS topic
                self.publish_text_to_speech(text)
                self.last_command_time = time.time()
                
                return jsonify({'status': 'success', 'message': f'Speech sent: {text}'})
            except Exception as e:
                self.get_logger().error(f'Error processing speech request: {e}')
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.flask_app.route('/status', methods=['GET'])
        def status():
            return jsonify({
                'status': self.current_status,
                'last_command_time': self.last_command_time,
                'uptime': time.time() - self.start_time
            })
    
    def run_flask_server(self):
        """Run Flask server in a separate thread"""
        self.flask_app.run(host='0.0.0.0', port=8888)
    
    def send_goal(self, x, y, theta=0.0):
        """Send a navigation goal to Nav2"""
        self.get_logger().info(f'Sending navigation goal: ({x}, {y}, {theta})')
        
        # Create goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        # Set position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        
        # Set orientation (quaternion)
        q = tf_transformations.quaternion_from_euler(0, 0, theta)
        goal_msg.pose.pose.orientation.x = q[0]
        goal_msg.pose.pose.orientation.y = q[1]
        goal_msg.pose.pose.orientation.z = q[2]
        goal_msg.pose.pose.orientation.w = q[3]
        
        # Send goal
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)
        
        return send_goal_future
    
    def goal_response_callback(self, future):
        """Callback for when Nav2 accepts or rejects the goal"""
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by Nav2')
            self.current_status = "rejected"
            return
        
        self.get_logger().info('Goal accepted by Nav2')
        self.current_status = "executing"
        
        # Get result future
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)
    
    def goal_result_callback(self, future):
        """Callback for when Nav2 completes the goal"""
        result = future.result().result
        
        if result:
            self.get_logger().info('Navigation goal completed successfully')
            self.current_status = "completed"
        else:
            self.get_logger().error('Navigation goal failed')
            self.current_status = "failed"
    
    def publish_text_to_speech(self, text):
        """Publish text to the TTS topic"""
        msg = String()
        msg.data = text
        self.tts_publisher.publish(msg)
        self.get_logger().info(f'Published TTS: "{text}"')

def main(args=None):
    rclpy.init(args=args)
    node = R2D2NavigationBridge()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()