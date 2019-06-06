cmake_minimum_required(VERSION 2.8.3)
project(circle_tracker)

set(OpenCV_DIR /usr/share/OpenCV/)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS 
  roscpp 
  rospy 
  std_msgs 
  sensor_msgs
  cv_bridge
  OpenCV)

## Declare ROS messages and services
#add_message_files(FILES Num.msg)
#add_service_files(FILES Trigger.srv)
#add_service_files(FILES SetInt.srv)
#add_service_files(FILES SetFloat.srv)

## Generate added messages and services
#generate_messages(DEPENDENCIES std_msgs)

## Declare a catkin package
catkin_package()


## Build talker and listener
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

#add_executable(talker src/talker.cpp)
#target_link_libraries(talker ${catkin_LIBRARIES})
#add_dependencies(talker)

add_executable(circle_tracker src/main_track.cpp)
add_executable(multi_circle_tracker src/main_multi_track.cpp)
target_link_libraries(circle_tracker ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries(multi_circle_tracker ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
#add_dependencies(main)
