#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>

class CompressedImageViewer : public rclcpp::Node {
public:
  CompressedImageViewer() : Node("compressed_image_viewer") {
    // Create publisher for decompressed images (optional)
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
      "/camera1/decompressed", 10);
    
    // Subscribe to compressed image topic
    subscription_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
      "/camera1/image_compressed", 10,
      std::bind(&CompressedImageViewer::callback, this, std::placeholders::_1));
    
    RCLCPP_INFO(this->get_logger(), "Compressed image viewer started");
    
    // Create display window
    cv::namedWindow("Decompressed Image", cv::WINDOW_NORMAL);
  }

private:
  void callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
    try {
      // Log the format
      RCLCPP_INFO(this->get_logger(), "Received image with format: %s", msg->format.c_str());
      
      // Convert vector<uint8_t> to Mat
      std::vector<uchar> buffer(msg->data.begin(), msg->data.end());
      
      // Decode the compressed image (regardless of the format string)
      cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
      
      if (image.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to decode compressed image");
        return;
      }
      
      // Display the image
      cv::imshow("Decompressed Image", image);
      cv::waitKey(1);
      
      // If you want to publish the decompressed image, you'll need to manually create the message
      // For simplicity, I'm omitting that part since it would normally use cv_bridge
      
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Error processing image: %s", e.what());
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CompressedImageViewer>();
  rclcpp::spin(node);
  cv::destroyAllWindows();
  rclcpp::shutdown();
  return 0;
}