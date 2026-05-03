// This MUST be at the top, before any #include
#define RMW_UXRCE_TRANSPORT_SERIAL

#include <micro_ros_arduino.h>

#include <stdio.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>

#include <geometry_msgs/msg/twist_stamped.h> // <--- Using TwistStamped

// --- Pin Definitions ---
#define LED_PIN 2     // Most ESP32s use GPIO 2 for the built-in LED
#define MR_DIR_PIN 3  // Right Motor Direction
#define MR_PWM_PIN 4  // Right Motor PWM
#define ML_DIR_PIN 5  // Left Motor Direction
#define ML_PWM_PIN 6  // Left Motor PWM

// --- Motor Driver Logic ---
// ⚠️ TUNE THIS: You might need to swap HIGH/LOW if your motors run backward
#define FORWARD_DIR HIGH
#define BACKWARD_DIR LOW

// --- ESP32 PWM (LEDC) Setup ---
#define PWM_FREQ 5000       // PWM Frequency (Hz)
#define PWM_RESOLUTION 8    // 8-bit resolution (0-255)
#define ML_PWM_CHANNEL 0    // Use PWM Channel 0 for Left Motor
#define MR_PWM_CHANNEL 1    // Use PWM Channel 1 for Right Motor
#define PWM_MAX 255         // Max PWM value (matches 8-bit resolution)

// --- Robot Tuning Parameters ---
// ⚠️ TUNE THIS: The max m/s your robot can go at 100% PWM
#define MAX_LINEAR_VEL 0.5
// ⚠️ TUNE THIS: The max rad/s your robot can turn
#define MAX_ANGULAR_VEL 1.0 

// --- ROS 2 Global Variables ---
rcl_subscription_t subscriber;
geometry_msgs__msg__TwistStamped msg; // <--- Use TwistStamped type
rclc_executor_t executor;
rcl_allocator_t allocator;
rclc_support_t support;
rcl_node_t node;

#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){error_loop();}}
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){}}

// --- Helper Functions ---

// 1. Error Loop
void error_loop(){
  while(1){
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    delay(100);
  }
}

// 3. Map float values
float fmap(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

// 4. Set individual motor speed
void setMotorSpeed(int dir_pin, int pwm_channel, int speed) {
  // Constrain speed to min/max
  int pwm_val = constrain(abs(speed), 0, PWM_MAX);

  if (speed > 0) {
    digitalWrite(dir_pin, FORWARD_DIR);
  } else {
    digitalWrite(dir_pin, BACKWARD_DIR);
  }
  
  ledcWrite(pwm_channel, pwm_val);
}

// --- ROS 2 Subscriber Callback ---
void subscription_callback(const void *msgin) {
  const geometry_msgs__msg__TwistStamped * msg = (const geometry_msgs__msg__TwistStamped *)msgin;

  // Extract linear.x and angular.z from the 'twist' field
  float linear_x = msg->twist.linear.x;
  float angular_z = msg->twist.angular.z;

  // Map velocities (m/s, rad/s) to PWM values (-255 to 255)
  int linear_pwm = (int)fmap(linear_x, -MAX_LINEAR_VEL, MAX_LINEAR_VEL, -PWM_MAX, PWM_MAX);
  int angular_pwm = (int)fmap(angular_z, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL, -PWM_MAX, PWM_MAX);

  // Differential Drive Mixing
  int left_pwm = linear_pwm - angular_pwm;
  int right_pwm = linear_pwm + angular_pwm;

  // Set motor speeds
  setMotorSpeed(ML_DIR_PIN, ML_PWM_CHANNEL, left_pwm);
  setMotorSpeed(MR_DIR_PIN, MR_PWM_CHANNEL, right_pwm);
}


// --- Main Setup ---
void setup() {
  set_microros_transports();

  // --- Initialize Pins ---
  pinMode(LED_PIN, OUTPUT);
  pinMode(ML_DIR_PIN, OUTPUT);
  pinMode(ML_PWM_PIN, OUTPUT);
  pinMode(MR_DIR_PIN, OUTPUT);
  pinMode(MR_PWM_PIN, OUTPUT);
  
  digitalWrite(LED_PIN, HIGH); // Turn LED on during setup
  
  // --- Setup PWM Channels ---
  ledcSetup(ML_PWM_CHANNEL, PWM_FREQ, PWM_RESOLUTION);
  ledcSetup(MR_PWM_CHANNEL, PWM_FREQ, PWM_RESOLUTION);
  
  // --- Attach Pins to Channels ---
  ledcAttachPin(ML_PWM_PIN, ML_PWM_CHANNEL);
  ledcAttachPin(MR_PWM_PIN, MR_PWM_CHANNEL);

  delay(2000); // Wait for setup

  // --- Initialize ROS 2 ---
  allocator = rcl_get_default_allocator();
  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
  RCCHECK(rclc_node_init_default(&node, "micro_ros_diff_drive_node", "", &support));

  // --- Initialize Subscriber ---
  RCCHECK(rclc_subscription_init_default(
    &subscriber,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, TwistStamped), // <--- TwistStamped type
    "cmd_vel")); // <--- Subscribing to "cmd_vel"

  // --- Initialize Executor ---
  RCCHECK(rclc_executor_init(&executor, &support.context, 1, &allocator));
  RCCHECK(rclc_executor_add_subscription(&executor, &subscriber, &msg, &subscription_callback, ON_NEW_DATA));

  digitalWrite(LED_PIN, LOW); // Turn LED off, setup complete
}

// --- Main Loop ---
void loop() {
  delay(10); // Small delay
  RCSOFTCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100)));
}