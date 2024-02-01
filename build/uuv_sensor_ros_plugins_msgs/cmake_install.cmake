# Install script for directory: /home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/uuv_simulator/uuv_sensor_plugins/uuv_sensor_ros_plugins_msgs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  
      if (NOT EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
        file(MAKE_DIRECTORY "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
      endif()
      if (NOT EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.catkin")
        file(WRITE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.catkin" "")
      endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install/_setup_util.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install" TYPE PROGRAM FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/uuv_sensor_ros_plugins_msgs/catkin_generated/installspace/_setup_util.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install/env.sh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install" TYPE PROGRAM FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/uuv_sensor_ros_plugins_msgs/catkin_generated/installspace/env.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install/setup.bash;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install/local_setup.bash")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install" TYPE FILE FILES
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/uuv_sensor_ros_plugins_msgs/catkin_generated/installspace/setup.bash"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/uuv_sensor_ros_plugins_msgs/catkin_generated/installspace/local_setup.bash"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install/setup.sh;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install/local_setup.sh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install" TYPE FILE FILES
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/uuv_sensor_ros_plugins_msgs/catkin_generated/installspace/setup.sh"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/uuv_sensor_ros_plugins_msgs/catkin_generated/installspace/local_setup.sh"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install/setup.zsh;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install/local_setup.zsh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install" TYPE FILE FILES
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/uuv_sensor_ros_plugins_msgs/catkin_generated/installspace/setup.zsh"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/uuv_sensor_ros_plugins_msgs/catkin_generated/installspace/local_setup.zsh"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install/.rosinstall")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install" TYPE FILE FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/uuv_sensor_ros_plugins_msgs/catkin_generated/installspace/.rosinstall")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/uuv_sensor_ros_plugins_msgs/msg" TYPE FILE FILES
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/uuv_simulator/uuv_sensor_plugins/uuv_sensor_ros_plugins_msgs/msg/DVL.msg"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/uuv_simulator/uuv_sensor_plugins/uuv_sensor_ros_plugins_msgs/msg/DVLBeam.msg"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/uuv_simulator/uuv_sensor_plugins/uuv_sensor_ros_plugins_msgs/msg/PositionWithCovariance.msg"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/uuv_simulator/uuv_sensor_plugins/uuv_sensor_ros_plugins_msgs/msg/PositionWithCovarianceStamped.msg"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/uuv_simulator/uuv_sensor_plugins/uuv_sensor_ros_plugins_msgs/msg/ChemicalParticleConcentration.msg"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/uuv_simulator/uuv_sensor_plugins/uuv_sensor_ros_plugins_msgs/msg/Salinity.msg"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/uuv_sensor_ros_plugins_msgs/srv" TYPE FILE FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/uuv_simulator/uuv_sensor_plugins/uuv_sensor_ros_plugins_msgs/srv/ChangeSensorState.srv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/uuv_sensor_ros_plugins_msgs/cmake" TYPE FILE FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/uuv_sensor_ros_plugins_msgs/catkin_generated/installspace/uuv_sensor_ros_plugins_msgs-msg-paths.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/devel/.private/uuv_sensor_ros_plugins_msgs/include/uuv_sensor_ros_plugins_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/devel/.private/uuv_sensor_ros_plugins_msgs/share/roseus/ros/uuv_sensor_ros_plugins_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/devel/.private/uuv_sensor_ros_plugins_msgs/share/common-lisp/ros/uuv_sensor_ros_plugins_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/devel/.private/uuv_sensor_ros_plugins_msgs/share/gennodejs/ros/uuv_sensor_ros_plugins_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python3" -m compileall "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/devel/.private/uuv_sensor_ros_plugins_msgs/lib/python3/dist-packages/uuv_sensor_ros_plugins_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages" TYPE DIRECTORY FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/devel/.private/uuv_sensor_ros_plugins_msgs/lib/python3/dist-packages/uuv_sensor_ros_plugins_msgs")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/uuv_sensor_ros_plugins_msgs/catkin_generated/installspace/uuv_sensor_ros_plugins_msgs.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/uuv_sensor_ros_plugins_msgs/cmake" TYPE FILE FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/uuv_sensor_ros_plugins_msgs/catkin_generated/installspace/uuv_sensor_ros_plugins_msgs-msg-extras.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/uuv_sensor_ros_plugins_msgs/cmake" TYPE FILE FILES
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/uuv_sensor_ros_plugins_msgs/catkin_generated/installspace/uuv_sensor_ros_plugins_msgsConfig.cmake"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/uuv_sensor_ros_plugins_msgs/catkin_generated/installspace/uuv_sensor_ros_plugins_msgsConfig-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/uuv_sensor_ros_plugins_msgs" TYPE FILE FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/uuv_simulator/uuv_sensor_plugins/uuv_sensor_ros_plugins_msgs/package.xml")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/uuv_sensor_ros_plugins_msgs/gtest/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/uuv_sensor_ros_plugins_msgs/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
