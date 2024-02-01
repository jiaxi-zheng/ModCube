# Install script for directory: /home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_common

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
  file(INSTALL DESTINATION "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install" TYPE PROGRAM FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/catkin_generated/installspace/_setup_util.py")
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
  file(INSTALL DESTINATION "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install" TYPE PROGRAM FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/catkin_generated/installspace/env.sh")
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
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/catkin_generated/installspace/setup.bash"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/catkin_generated/installspace/local_setup.bash"
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
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/catkin_generated/installspace/setup.sh"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/catkin_generated/installspace/local_setup.sh"
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
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/catkin_generated/installspace/setup.zsh"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/catkin_generated/installspace/local_setup.zsh"
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
  file(INSTALL DESTINATION "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/install" TYPE FILE FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/catkin_generated/installspace/.rosinstall")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/catkin_generated/safe_execute_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tauv_common/srv" TYPE FILE FILES
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_common/srv/GetThrusterCurve.srv"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_common/srv/ThrusterManagerInfo.srv"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_common/srv/SetThrusterManagerConfig.srv"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_common/srv/GetThrusterManagerConfig.srv"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tauv_common/cmake" TYPE FILE FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/catkin_generated/installspace/tauv_common-msg-paths.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/devel/.private/tauv_common/include/tauv_common")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/devel/.private/tauv_common/share/roseus/ros/tauv_common")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/devel/.private/tauv_common/share/common-lisp/ros/tauv_common")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/devel/.private/tauv_common/share/gennodejs/ros/tauv_common")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python3" -m compileall "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/devel/.private/tauv_common/lib/python3/dist-packages/tauv_common")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages" TYPE DIRECTORY FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/devel/.private/tauv_common/lib/python3/dist-packages/tauv_common")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/catkin_generated/installspace/tauv_common.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tauv_common/cmake" TYPE FILE FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/catkin_generated/installspace/tauv_common-msg-extras.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tauv_common/cmake" TYPE FILE FILES
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/catkin_generated/installspace/tauv_commonConfig.cmake"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/catkin_generated/installspace/tauv_commonConfig-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tauv_common" TYPE FILE FILES "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_common/package.xml")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tauv_common" TYPE DIRECTORY FILES
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_common/launch"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_common/config"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_common/include"
    "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_common/src/"
    FILES_MATCHING REGEX "/[^/]*\\.h$" REGEX "/[^/]*\\.hpp$" REGEX "/[^/]*\\~$" EXCLUDE)
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/gtest/cmake_install.cmake")
  include("/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/src/state_estimation_cpp/cmake_install.cmake")
  include("/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/src/tauv_alarms/cmake_install.cmake")
  include("/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/src/tauv_utils_cpp/cmake_install.cmake")
  include("/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/src/global_map_cpp/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/tauv_common/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
