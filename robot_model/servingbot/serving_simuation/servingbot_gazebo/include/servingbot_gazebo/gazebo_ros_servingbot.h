/*******************************************************************************
* Copyright 2018 Zetabank CO., LTD.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/* Authors: Jaewan Choi */

#ifndef GAZEBO_ROS_SERVINGBOT_H_
#define GAZEBO_ROS_SERVINGBOT_H_

#include <ros/ros.h>
#include <ros/time.h>

#include <math.h>
#include <limits.h>

#include <std_msgs/String.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
#include <tf/tf.h>
#include <nav_msgs/Odometry.h>

#define DEG2RAD (M_PI / 180.0)
#define RAD2DEG (180.0 / M_PI)

#define CENTER 0
#define LEFT   1
#define RIGHT  2

#define LINEAR_VELOCITY  0.3
#define ANGULAR_VELOCITY 1.5

#define GET_SB_DIRECTION 0
#define SB_DRIVE_FORWARD 1
#define SB_RIGHT_TURN    2
#define SB_LEFT_TURN     3

class GazeboRosServingbot
{
 public:
  GazeboRosServingbot();
  ~GazeboRosServingbot();
  bool init();
  bool controlLoop();

 private:
  // ROS NodeHandle
  ros::NodeHandle nh_;
  ros::NodeHandle nh_priv_;

  // ROS Parameters
  bool is_debug_;

  // ROS Time

  // ROS Topic Publishers
  ros::Publisher cmd_vel_pub_;

  // ROS Topic Subscribers
  ros::Subscriber laser_scan_sub_;
  ros::Subscriber joint_state_sub_;

  double turning_radius_;
  double rotate_angle_;
  double front_distance_limit_;
  double side_distance_limit_;

  double direction_vector_[3] = {0.0, 0.0, 0.0};

  double right_joint_encoder_;
  double priv_right_joint_encoder_;

  // Function prototypes
  void updatecommandVelocity(double linear, double angular);
  void laserScanMsgCallBack(const sensor_msgs::LaserScan::ConstPtr &msg);
  void jointStateMsgCallBack(const sensor_msgs::JointState::ConstPtr &msg);
};
#endif // GAZEBO_ROS_SERVINGBOT_H_
