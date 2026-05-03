import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage as RosCompressedImage
from std_msgs.msg import String as RosString
import requests
import base64
import json

class LocalImageSenderNode(Node):
    def __init__(self):
        super().__init__('local_image_sender_node')
        
        self.api_url = 'http://localhost:8080/process_frame'
        
        # Subscribe to your camera's final, correct compressed image topic
        self.subscription = self.create_subscription(
            RosCompressedImage,
            '/camera1/image_compressed',  # <-- FINAL CORRECT TOPIC
            self.image_callback,
            10)
            
        self.publisher = self.create_publisher(
            RosString, '/llm/scene_analysis', 10)
            
        self.frame_skip = 15 # Throttling for ~2 FPS
        self.frame_counter = 0
        
        self.get_logger().info('Local Image Sender Node started.')
        self.get_logger().info('Subscribing to /camera1/image_compressed')
        self.get_logger().info('Waiting for SSH tunnel to be active on localhost:8080...')

    def image_callback(self, msg: RosCompressedImage):
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            return

        self.get_logger().info(f'Sending frame #{self.frame_counter} to supercomputer...')
        try:
            image_bytes = bytes(msg.data)
            img_str = base64.b64encode(image_bytes).decode('utf-8')

            response = requests.post(self.api_url, json={'image': img_str}, timeout=20)
            
            if response.status_code == 200:
                json_data = response.json()
                ros_msg = RosString()
                ros_msg.data = json.dumps(json_data)
                self.publisher.publish(ros_msg)
                self.get_logger().info('Successfully received and published analysis.')
            else:
                self.get_logger().error(f'API Error: {response.status_code} - {response.text}')

        except requests.exceptions.RequestException as e:
            self.get_logger().error(f'Connection Error. Is the SSH tunnel running? Error: {e}')
        except Exception as e:
            self.get_logger().error(f'An unexpected error occurred: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = LocalImageSenderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()