cmake_minimum_required(VERSION 2.8.3)
project(line_segment_detector)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  message_generation
  message_runtime
)

add_message_files(
  FILES
  LineSeg.msg
  LineSegArray.msg
)

generate_messages(
	DEPENDENCIES
	geometry_msgs
  std_msgs
	)
catkin_package(
	CATKIN_DEPENDS message_runtime)

include_directories(
  ${catkin_INCLUDE_DIRS}
)


catkin_install_python(PROGRAMS
  demo_ros.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#install(DIRECTORY
#  nodes/models
#  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)

