import rclpy
from rclpy.node import Node
from std_msgs.msg import String as RosString
import json
import pprint # Import the pretty-print library

class JsonViewerNode(Node):
    def __init__(self):
        super().__init__('json_viewer_node')
        
        # Subscribe to the final analysis topic
        self.subscription = self.create_subscription(
            RosString,
            '/llm/scene_analysis',
            self.listener_callback,
            10)
            
        self.get_logger().info('JSON Viewer node started. Waiting for messages on /llm/scene_analysis...')
        
        # Initialize a pretty-printer
        self.pp = pprint.PrettyPrinter(indent=2)

    def listener_callback(self, msg: RosString):
        """
        This function is called every time a new message is received.
        """
        self.get_logger().info('--- New Message Received ---')
        try:
            # Parse the incoming JSON string into a Python dictionary
            json_data = json.loads(msg.data)
            
            # Pretty-print the entire dictionary to the console
            self.pp.pprint(json_data)
            
        except json.JSONDecodeError:
            self.get_logger().error('Received a message that was not valid JSON:')
            print(msg.data) # Print the raw data if it's not JSON
        except Exception as e:
            self.get_logger().error(f'An unexpected error occurred: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = JsonViewerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()