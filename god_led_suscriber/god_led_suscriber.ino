// This MUST be at the top, before any #include
#define RMW_UXRCE_TRANSPORT_SERIAL

#include <micro_ros_arduino.h>

#include <stdio.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>

// 1. Include the 'TwistStamped' message header
#include <geometry_msgs/msg/twist_stamped.h>

rcl_subscription_t subscriber;
// 2. Use the 'TwistStamped' message type
geometry_msgs__msg__TwistStamped msg;

rclc_executor_t executor;
rcl_allocator_t allocator;
rclc_support_t support;
rcl_node_t node;

// Use pin 2 for most ESP32 built-in LEDs
#define LED_PIN 2
#define mr_dir 3
#define mr_pwm 4
#define ml_dir 5
#define ml_pwm 6

#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){error_loop();}}
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){error_loop();}}


void error_loop(){
  while(1){
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    delay(100);
  }
}

// Callback for 'TwistStamped' messages
void subscription_callback(const void *msgin) {
  // 3. Cast to the correct message type
  const geometry_msgs__msg__TwistStamped * msg = (const geometry_msgs__msg__TwistStamped *)msgin;
  
  // 4. Access the data via the 'twist' field
  //    (msg->twist.linear.x)
  digitalWrite(LED_PIN, (msg->twist.linear.x == 0) ? LOW : HIGH);
}

void setup() {
  // This calls our stable transport function
  set_microros_transports();
  
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);  
  
  delay(2000);

  allocator = rcl_get_default_allocator();

   //create init_options
  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));

  // create node
  RCCHECK(rclc_node_init_default(&node, "micro_ros_arduino_subscriber_node", "", &support));

  // create subscriber
  RCCHECK(rclc_subscription_init_default(
    &subscriber,
    &node,
    // 5. Use the 'TwistStamped' message type support
    ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, TwistStamped),
    // 6. Subscribe to the 'cmd_vel' topic
    "cmd_vel"));

  // create executor
  RCCHECK(rclc_executor_init(&executor, &support.context, 1, &allocator));
  RCCHECK(rclc_executor_add_subscription(&executor, &subscriber, &msg, &subscription_callback, ON_NEW_DATA));
}

void loop() {
  delay(100);
  // Using RCCHECK in loop can be risky; RCSOFTCHECK is safer
  // but we will follow your example's pattern.
  RCCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100)));
}