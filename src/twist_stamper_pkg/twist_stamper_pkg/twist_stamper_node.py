#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped

class TwistStamperNode(Node):
    """
    This node subscribes to a Twist message, adds a timestamp
    and frame_id, and republishes it as a TwistStamped message.
    """
    def __init__(self):
        super().__init__('twist_stamper')

        # --- Parameters ---
        # Declare and get the frame_id parameter.
        # Default is now 'base_footprint' as you requested.
        self.declare_parameter('frame_id', 'base_footprint')
        self.frame_id_ = self.get_parameter('frame_id').get_parameter_value().string_value

        # --- Publisher ---
        # Publishes the TwistStamped message.
        # Default topic name is 'cmd_vel_out' (remappable)
        self.publisher_ = self.create_publisher(
            TwistStamped, 
            'cmd_vel_out',  # Default output topic
            10)

        # --- Subscriber ---
        # Subscribes to the Twist message.
        # Default topic name is 'cmd_vel_in' (remappable)
        self.subscription_ = self.create_subscription(
            Twist,
            'cmd_vel_in', # Default input topic
            self.listener_callback,
            10)
        
        self.get_logger().info(
            f"Twist Stamper Node has started. \n"
            f"  Subscribing to:   '{self.subscription_.topic_name}' (Twist)\n"
            f"  Publishing to:    '{self.publisher_.topic_name}' (TwistStamped)\n"
            f"  Using frame_id:   '{self.frame_id_}'")

    def listener_callback(self, msg: Twist):
        """
        Callback for the Twist subscriber.
        """
        stamped_msg = TwistStamped()

        # Fill in the header
        stamped_msg.header.stamp = self.get_clock().now().to_msg()
        stamped_msg.header.frame_id = self.frame_id_

        # Copy the twist data
        stamped_msg.twist = msg

        # Publish the new message
        self.publisher_.publish(stamped_msg)

def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)
    
    # Create an instance of the node
    node = TwistStamperNode()
    
    try:
        # "Spin" the node, which keeps it alive and processing callbacks
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Handle Ctrl+C
        pass
    finally:
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()