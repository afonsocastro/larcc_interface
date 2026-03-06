cmake_minimum_required(VERSION 3.0.2)
project(object_recognition_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  geometry_msgs
  sensor_msgs
  shape_msgs
)

add_message_files(
  FILES
  RecognizedObject.msg
  RecognizedObjectArray.msg
  # adiciona outros .msg necessários
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
  shape_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs geometry_msgs sensor_msgs shape_msgs
)

include_directories(${catkin_INCLUDE_DIRS})