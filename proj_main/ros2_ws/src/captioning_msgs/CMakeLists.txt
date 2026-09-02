cmake_minimum_required(VERSION 3.8)
project(captioning_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SoundSourceTrack.msg"
  "msg/SoundSourceTracks.msg"
  "msg/SeparatedAudio.msg"
  "msg/Face.msg"
  "msg/Faces.msg"
  "msg/Caption.msg"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

ament_package()
