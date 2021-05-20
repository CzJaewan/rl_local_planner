cmake_minimum_required(VERSION 3.5)
project(gazebo_rl_test)

## Compile as C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++11 -O3)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  costmap_2d
  dynamic_reconfigure
  geometry_msgs
  nav_msgs
  ackermann_msgs
  message_generation
  move_base
  roscpp
  rospy
  std_msgs
  tf
  visualization_msgs
)

catkin_package(
   INCLUDE_DIRS
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories( ${catkin_INCLUDE_DIRS})
include_directories(/usr/include)
link_directories(/usr/lib)

# Own Global planner for tracking desired trajectory
# Look a head Node
ADD_EXECUTABLE( Look_a_head_Node src/lookahead_point.cpp)
TARGET_LINK_LIBRARIES(Look_a_head_Node ${catkin_LIBRARIES})


