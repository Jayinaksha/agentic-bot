cmake_minimum_required(VERSION 3.8)
project(compressed_image_viewer)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# Add executable - make sure this matches your actual file name
add_executable(compressed_image_viewer src/compressed_image_viewer.cpp)
target_include_directories(compressed_image_viewer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)
ament_target_dependencies(
  compressed_image_viewer
  rclcpp
  std_msgs
  sensor_msgs
)
target_link_libraries(compressed_image_viewer ${OpenCV_LIBRARIES})

# Install executable
install(TARGETS compressed_image_viewer
  DESTINATION lib/${PROJECT_NAME})

ament_package()