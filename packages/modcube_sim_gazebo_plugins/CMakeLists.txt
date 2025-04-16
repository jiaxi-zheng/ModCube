cmake_minimum_required(VERSION 3.0.2)
project(modcube_sim_gazebo_plugins)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(
  catkin
  REQUIRED COMPONENTS
  gazebo_dev
  roscpp
  std_msgs
  geometry_msgs
  modcube_msgs
  tf2
  tf2_ros
)

catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    modcube_thruster_plugin
    modcube_navigation_state_plugin
  CATKIN_DEPENDS
    roscpp
    std_msgs
    geometry_msgs
    modcube_msgs
    tf2
    tf2_ros
)

include_directories(${PROJECT_SOURCE_DIR}/include ${catkin_INCLUDE_DIRS})

link_directories(${catkin_LIBRARY_DIRS})

add_library(modcube_thruster_plugin SHARED src/thruster_plugin.cpp)
target_link_libraries(modcube_thruster_plugin ${catkin_LIBRARIES})
add_dependencies(modcube_thruster_plugin ${catkin_EXPORTED_TARGETS})

add_library(modcube_navigation_state_plugin SHARED src/navigation_state_plugin.cpp)
target_link_libraries(modcube_navigation_state_plugin ${catkin_LIBRARIES})
add_dependencies(modcube_navigation_state_plugin ${catkin_EXPORTED_TARGETS})

install(TARGETS modcube_thruster_plugin modcube_navigation_state_plugin
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
