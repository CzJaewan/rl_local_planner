/*******************************************************************************
* Copyright 2018 Seoultech CO., LTD.
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

/* Authors: Geonhee Lee */

#include "servingbot_gazebo/gazebo_ros_servingbot.h"

GazeboRosServingbot::GazeboRosServingbot()
  : nh_priv_("~")
{
  //Init gazebo ros zetabank node
  ROS_INFO("Servingbot Simulation Node Init");
  ROS_ASSERT(init());
}

GazeboRosServingbot::~GazeboRosServingbot()
{
  updatecommandVelocity(0.0, 0.0);
  ros::shutdown();
}

/*******************************************************************************
* Init function
*******************************************************************************/
bool GazeboRosServingbot::init()
{
  // initialize ROS parameter
  nh_.param("is_debug", is_debug_, is_debug_);
  std::string robot_model = nh_.param<std::string>("servingbot_model", "");

  turning_radius_ = 0.1435;
  rotate_angle_ = 40.0 * DEG2RAD;
  front_distance_limit_ = 0.7;
  side_distance_limit_  = 0.6;

  ROS_INFO("robot_model : %s", robot_model.c_str());
  ROS_INFO("turning_radius_ : %lf", turning_radius_);
  ROS_INFO("front_distance_limit_ = %lf", front_distance_limit_);
  ROS_INFO("side_distance_limit_ = %lf", side_distance_limit_);

  // initialize variables
  right_joint_encoder_ = 0.0;
  priv_right_joint_encoder_ = 0.0;
  // initialize publishers
  cmd_vel_pub_   = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10);

  // initialize subscribers
  laser_scan_sub_  = nh_.subscribe("/scan", 10, &GazeboRosServingbot::laserScanMsgCallBack, this);
  joint_state_sub_ = nh_.subscribe("/joint_states", 10, &GazeboRosServingbot::jointStateMsgCallBack, this);

  return true;
}

void GazeboRosServingbot::jointStateMsgCallBack(const sensor_msgs::JointState::ConstPtr &msg)
{
  right_joint_encoder_ = msg->position.at(0);
}

void GazeboRosServingbot::laserScanMsgCallBack(const sensor_msgs::LaserScan::ConstPtr &msg)
{
  uint16_t scan_angle[3] = {0, 30, 330};

  for (int num = 0; num < 3; num++)
  {
    if (std::isinf(msg->ranges.at(scan_angle[num])))
    {
      direction_vector_[num] = msg->range_max;
    }
    else
    {
      direction_vector_[num] = msg->ranges.at(scan_angle[num]);
    }
  }
}

void GazeboRosServingbot::updatecommandVelocity(double linear, double angular)
{
  geometry_msgs::Twist cmd_vel;

  cmd_vel.linear.x  = linear;
  cmd_vel.angular.z = angular;

  cmd_vel_pub_.publish(cmd_vel);
}

/*******************************************************************************
* Control Loop function
*******************************************************************************/
bool GazeboRosServingbot::controlLoop()
{
  static uint8_t servingbot_state_num = 0;
  double wheel_radius = 0.033;
  double servingbot_rotation = 0.0;

  servingbot_rotation = (rotate_angle_ * turning_radius_ / wheel_radius);

  switch(servingbot_state_num)
  {
    case GET_SB_DIRECTION:
      if (direction_vector_[CENTER] > front_distance_limit_)
      {
        servingbot_state_num = SB_DRIVE_FORWARD;
      }

      if (direction_vector_[CENTER] < front_distance_limit_ || direction_vector_[LEFT] < side_distance_limit_)
      {
        priv_right_joint_encoder_ = right_joint_encoder_ - servingbot_rotation;
        servingbot_state_num = SB_RIGHT_TURN;
      }
      else if (direction_vector_[RIGHT] < side_distance_limit_)
      {
        priv_right_joint_encoder_ = right_joint_encoder_ + servingbot_rotation;
        servingbot_state_num = SB_LEFT_TURN;
      }
      break;

    case SB_DRIVE_FORWARD:
      updatecommandVelocity(LINEAR_VELOCITY, 0.0);
      servingbot_state_num = GET_SB_DIRECTION;
      break;

    case SB_RIGHT_TURN:
      if (priv_right_joint_encoder_ == 0.0)
      {
        servingbot_state_num = GET_SB_DIRECTION;
      }
      else
      {
        if (fabs(priv_right_joint_encoder_ - right_joint_encoder_) < 0.1)
        {
          servingbot_state_num = GET_SB_DIRECTION;
        }
        else
        {
          updatecommandVelocity(0.0, -1 * ANGULAR_VELOCITY);
        }
      }
      break;

    case SB_LEFT_TURN:
      if (priv_right_joint_encoder_ == 0.0)
      {
        servingbot_state_num = GET_SB_DIRECTION;
      }
      else
      {
        if (fabs(priv_right_joint_encoder_ - right_joint_encoder_) < 0.1)
        {
          servingbot_state_num = GET_SB_DIRECTION;
        }
        else
        {
          updatecommandVelocity(0.0, ANGULAR_VELOCITY);
        }
      }
      break;

    default:
      servingbot_state_num = GET_SB_DIRECTION;
      break;
  }

  return true;
}

/*******************************************************************************
* Main function
*******************************************************************************/
int main(int argc, char* argv[])
{
  ros::init(argc, argv, "gazebo_ros_servingbot");
  GazeboRosServingbot gazeboRosServingbot;

  ros::Rate loop_rate(125);

  while (ros::ok())
  {
    gazeboRosServingbot.controlLoop();
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
