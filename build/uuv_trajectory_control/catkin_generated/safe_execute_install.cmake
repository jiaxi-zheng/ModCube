execute_process(COMMAND "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/uuv_trajectory_control/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/build/uuv_trajectory_control/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
