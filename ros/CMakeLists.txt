cmake_minimum_required(VERSION 3.0.2)
project(unicycle_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  tf
)

catkin_package()

install(PROGRAMS
  scripts/unicycle_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
