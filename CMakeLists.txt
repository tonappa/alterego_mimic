cmake_minimum_required(VERSION 3.0.2)
project(alterego_mimic)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs
)

# The node runs through the wrapper, with the Python of the conda env (see scripts/person_tracker_wrapper.sh)
install(PROGRAMS scripts/person_tracker_wrapper.sh scripts/person_tracker.py scripts/person_follower.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
