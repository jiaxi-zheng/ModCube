# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "tauv_msgs: 28 messages, 15 services")

set(MSG_I_FLAGS "-Itauv_msgs:/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg;-Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Ivision_msgs:/opt/ros/noetic/share/vision_msgs/cmake/../msg;-Ijsk_recognition_msgs:/opt/ros/noetic/share/jsk_recognition_msgs/cmake/../msg;-Ipcl_msgs:/opt/ros/noetic/share/pcl_msgs/cmake/../msg;-Ijsk_footstep_msgs:/opt/ros/noetic/share/jsk_footstep_msgs/cmake/../msg;-Iactionlib_msgs:/opt/ros/noetic/share/actionlib_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(tauv_msgs_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmReport.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmReport.msg" ""
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmWithMessage.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmWithMessage.msg" ""
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerCommand.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerCommand.msg" ""
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerDebug.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerDebug.msg" "tauv_msgs/PIDTuning:tauv_msgs/PIDDebug"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg" "geometry_msgs/Point:std_msgs/Header"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetections.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetections.msg" "std_msgs/Header:geometry_msgs/Point:tauv_msgs/FeatureDetection"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FluidDepth.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FluidDepth.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsTuning.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsTuning.msg" ""
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParametersEstimate.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParametersEstimate.msg" ""
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParameterConfigUpdate.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParameterConfigUpdate.msg" ""
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/GateDetection.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/GateDetection.msg" "geometry_msgs/Point"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Message.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Message.msg" ""
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/SonarPulse.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/SonarPulse.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Servos.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Servos.msg" ""
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/RegisterMeasurement.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/RegisterMeasurement.msg" "geometry_msgs/Point:tauv_msgs/PoseGraphMeasurement:std_msgs/Header"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg" "geometry_msgs/Point:std_msgs/Header"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ReadableAlarmReport.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ReadableAlarmReport.msg" ""
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Thrust.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Thrust.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TrajPoint.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TrajPoint.msg" "geometry_msgs/Quaternion:geometry_msgs/Pose:geometry_msgs/Twist:geometry_msgs/Vector3:geometry_msgs/Point"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Battery.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Battery.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuSync.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuSync.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuData.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuData.msg" "std_msgs/Header:geometry_msgs/Vector3"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TeledyneDvlData.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TeledyneDvlData.msg" "std_msgs/String:std_msgs/Header:geometry_msgs/Vector3"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/NavigationState.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/NavigationState.msg" "std_msgs/Header:geometry_msgs/Vector3"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg" ""
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg" "tauv_msgs/PIDTuning"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDPlannerDebug.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDPlannerDebug.msg" "tauv_msgs/PIDTuning:tauv_msgs/PIDDebug"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PingDetection.msg" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PingDetection.msg" "std_msgs/Header:geometry_msgs/Vector3"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/FeatureDetectionsSync.srv" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/FeatureDetectionsSync.srv" "std_msgs/Header:geometry_msgs/Point:tauv_msgs/FeatureDetection:tauv_msgs/FeatureDetections"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SonarControl.srv" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SonarControl.srv" ""
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFind.srv" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFind.srv" "std_msgs/Header:geometry_msgs/Point:tauv_msgs/FeatureDetection"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindOne.srv" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindOne.srv" "std_msgs/Header:geometry_msgs/Point:tauv_msgs/FeatureDetection"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindClosest.srv" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindClosest.srv" "std_msgs/Header:geometry_msgs/Point:tauv_msgs/FeatureDetection"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetCameraInfo.srv" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetCameraInfo.srv" "sensor_msgs/RegionOfInterest:sensor_msgs/CameraInfo:std_msgs/Header"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SyncAlarms.srv" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SyncAlarms.srv" "tauv_msgs/AlarmWithMessage"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneDynamics.srv" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneDynamics.srv" "tauv_msgs/DynamicsTuning"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneController.srv" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneController.srv" "tauv_msgs/PIDTuning"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TunePIDPlanner.srv" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TunePIDPlanner.srv" "tauv_msgs/PIDTuning"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetTrajectory.srv" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetTrajectory.srv" "geometry_msgs/Quaternion:geometry_msgs/Pose:geometry_msgs/Twist:std_msgs/Header:geometry_msgs/Vector3:geometry_msgs/Point"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetTargetPose.srv" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetTargetPose.srv" "geometry_msgs/Pose:geometry_msgs/Quaternion:geometry_msgs/Point"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetPose.srv" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetPose.srv" "geometry_msgs/Vector3"
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/RunMission.srv" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/RunMission.srv" ""
)

get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/UpdateDynamicsParameterConfigs.srv" NAME_WE)
add_custom_target(_tauv_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "tauv_msgs" "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/UpdateDynamicsParameterConfigs.srv" "tauv_msgs/DynamicsParameterConfigUpdate"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmReport.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmWithMessage.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerCommand.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerDebug.msg"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetections.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FluidDepth.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsTuning.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParametersEstimate.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParameterConfigUpdate.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/GateDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Message.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/SonarPulse.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Servos.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/RegisterMeasurement.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ReadableAlarmReport.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Thrust.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TrajPoint.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Battery.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuSync.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuData.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TeledyneDvlData.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/NavigationState.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDPlannerDebug.msg"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PingDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)

### Generating Services
_generate_srv_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/FeatureDetectionsSync.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetections.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SonarControl.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFind.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindOne.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindClosest.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetCameraInfo.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/CameraInfo.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SyncAlarms.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmWithMessage.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneDynamics.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneController.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TunePIDPlanner.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetTrajectory.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetTargetPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/RunMission.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_cpp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/UpdateDynamicsParameterConfigs.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParameterConfigUpdate.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
)

### Generating Module File
_generate_module_cpp(tauv_msgs
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(tauv_msgs_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(tauv_msgs_generate_messages tauv_msgs_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmReport.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmWithMessage.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerCommand.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerDebug.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetections.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FluidDepth.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsTuning.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParametersEstimate.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParameterConfigUpdate.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/GateDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Message.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/SonarPulse.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Servos.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/RegisterMeasurement.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ReadableAlarmReport.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Thrust.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TrajPoint.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Battery.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuSync.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuData.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TeledyneDvlData.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/NavigationState.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDPlannerDebug.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PingDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/FeatureDetectionsSync.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SonarControl.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFind.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindOne.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindClosest.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetCameraInfo.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SyncAlarms.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneDynamics.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneController.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TunePIDPlanner.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetTrajectory.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetTargetPose.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetPose.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/RunMission.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/UpdateDynamicsParameterConfigs.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_cpp _tauv_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(tauv_msgs_gencpp)
add_dependencies(tauv_msgs_gencpp tauv_msgs_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_msgs_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmReport.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmWithMessage.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerCommand.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerDebug.msg"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetections.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FluidDepth.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsTuning.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParametersEstimate.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParameterConfigUpdate.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/GateDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Message.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/SonarPulse.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Servos.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/RegisterMeasurement.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ReadableAlarmReport.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Thrust.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TrajPoint.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Battery.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuSync.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuData.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TeledyneDvlData.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/NavigationState.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDPlannerDebug.msg"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_msg_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PingDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)

### Generating Services
_generate_srv_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/FeatureDetectionsSync.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetections.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_srv_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SonarControl.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_srv_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFind.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_srv_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindOne.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_srv_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindClosest.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_srv_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetCameraInfo.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/CameraInfo.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_srv_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SyncAlarms.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmWithMessage.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_srv_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneDynamics.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_srv_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneController.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_srv_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TunePIDPlanner.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_srv_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetTrajectory.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_srv_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetTargetPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_srv_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_srv_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/RunMission.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)
_generate_srv_eus(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/UpdateDynamicsParameterConfigs.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParameterConfigUpdate.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
)

### Generating Module File
_generate_module_eus(tauv_msgs
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(tauv_msgs_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(tauv_msgs_generate_messages tauv_msgs_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmReport.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmWithMessage.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerCommand.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerDebug.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetections.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FluidDepth.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsTuning.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParametersEstimate.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParameterConfigUpdate.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/GateDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Message.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/SonarPulse.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Servos.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/RegisterMeasurement.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ReadableAlarmReport.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Thrust.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TrajPoint.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Battery.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuSync.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuData.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TeledyneDvlData.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/NavigationState.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDPlannerDebug.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PingDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/FeatureDetectionsSync.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SonarControl.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFind.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindOne.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindClosest.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetCameraInfo.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SyncAlarms.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneDynamics.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneController.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TunePIDPlanner.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetTrajectory.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetTargetPose.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetPose.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/RunMission.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/UpdateDynamicsParameterConfigs.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_eus _tauv_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(tauv_msgs_geneus)
add_dependencies(tauv_msgs_geneus tauv_msgs_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_msgs_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmReport.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmWithMessage.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerCommand.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerDebug.msg"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetections.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FluidDepth.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsTuning.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParametersEstimate.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParameterConfigUpdate.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/GateDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Message.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/SonarPulse.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Servos.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/RegisterMeasurement.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ReadableAlarmReport.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Thrust.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TrajPoint.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Battery.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuSync.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuData.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TeledyneDvlData.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/NavigationState.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDPlannerDebug.msg"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_msg_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PingDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)

### Generating Services
_generate_srv_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/FeatureDetectionsSync.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetections.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SonarControl.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFind.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindOne.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindClosest.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetCameraInfo.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/CameraInfo.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SyncAlarms.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmWithMessage.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneDynamics.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneController.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TunePIDPlanner.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetTrajectory.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetTargetPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/RunMission.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)
_generate_srv_lisp(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/UpdateDynamicsParameterConfigs.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParameterConfigUpdate.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
)

### Generating Module File
_generate_module_lisp(tauv_msgs
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(tauv_msgs_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(tauv_msgs_generate_messages tauv_msgs_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmReport.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmWithMessage.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerCommand.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerDebug.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetections.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FluidDepth.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsTuning.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParametersEstimate.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParameterConfigUpdate.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/GateDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Message.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/SonarPulse.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Servos.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/RegisterMeasurement.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ReadableAlarmReport.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Thrust.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TrajPoint.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Battery.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuSync.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuData.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TeledyneDvlData.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/NavigationState.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDPlannerDebug.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PingDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/FeatureDetectionsSync.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SonarControl.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFind.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindOne.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindClosest.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetCameraInfo.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SyncAlarms.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneDynamics.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneController.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TunePIDPlanner.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetTrajectory.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetTargetPose.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetPose.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/RunMission.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/UpdateDynamicsParameterConfigs.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_lisp _tauv_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(tauv_msgs_genlisp)
add_dependencies(tauv_msgs_genlisp tauv_msgs_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_msgs_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmReport.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmWithMessage.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerCommand.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerDebug.msg"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetections.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FluidDepth.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsTuning.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParametersEstimate.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParameterConfigUpdate.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/GateDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Message.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/SonarPulse.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Servos.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/RegisterMeasurement.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ReadableAlarmReport.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Thrust.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TrajPoint.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Battery.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuSync.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuData.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TeledyneDvlData.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/NavigationState.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDPlannerDebug.msg"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_msg_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PingDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)

### Generating Services
_generate_srv_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/FeatureDetectionsSync.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetections.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_srv_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SonarControl.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_srv_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFind.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_srv_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindOne.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_srv_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindClosest.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_srv_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetCameraInfo.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/CameraInfo.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_srv_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SyncAlarms.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmWithMessage.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_srv_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneDynamics.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_srv_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneController.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_srv_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TunePIDPlanner.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_srv_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetTrajectory.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_srv_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetTargetPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_srv_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_srv_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/RunMission.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)
_generate_srv_nodejs(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/UpdateDynamicsParameterConfigs.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParameterConfigUpdate.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
)

### Generating Module File
_generate_module_nodejs(tauv_msgs
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(tauv_msgs_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(tauv_msgs_generate_messages tauv_msgs_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmReport.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmWithMessage.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerCommand.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerDebug.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetections.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FluidDepth.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsTuning.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParametersEstimate.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParameterConfigUpdate.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/GateDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Message.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/SonarPulse.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Servos.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/RegisterMeasurement.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ReadableAlarmReport.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Thrust.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TrajPoint.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Battery.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuSync.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuData.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TeledyneDvlData.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/NavigationState.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDPlannerDebug.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PingDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/FeatureDetectionsSync.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SonarControl.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFind.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindOne.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindClosest.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetCameraInfo.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SyncAlarms.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneDynamics.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneController.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TunePIDPlanner.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetTrajectory.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetTargetPose.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetPose.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/RunMission.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/UpdateDynamicsParameterConfigs.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_nodejs _tauv_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(tauv_msgs_gennodejs)
add_dependencies(tauv_msgs_gennodejs tauv_msgs_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_msgs_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmReport.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmWithMessage.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerCommand.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerDebug.msg"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetections.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FluidDepth.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsTuning.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParametersEstimate.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParameterConfigUpdate.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/GateDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Message.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/SonarPulse.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Servos.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/RegisterMeasurement.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ReadableAlarmReport.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Thrust.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TrajPoint.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Battery.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuSync.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuData.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TeledyneDvlData.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/String.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/NavigationState.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDPlannerDebug.msg"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_msg_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PingDetection.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)

### Generating Services
_generate_srv_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/FeatureDetectionsSync.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetections.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_srv_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SonarControl.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_srv_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFind.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_srv_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindOne.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_srv_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindClosest.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_srv_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetCameraInfo.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/RegionOfInterest.msg;/opt/ros/noetic/share/sensor_msgs/cmake/../msg/CameraInfo.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_srv_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SyncAlarms.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmWithMessage.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_srv_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneDynamics.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_srv_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneController.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_srv_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TunePIDPlanner.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_srv_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetTrajectory.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Twist.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_srv_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetTargetPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_srv_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_srv_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/RunMission.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)
_generate_srv_py(tauv_msgs
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/UpdateDynamicsParameterConfigs.srv"
  "${MSG_I_FLAGS}"
  "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParameterConfigUpdate.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
)

### Generating Module File
_generate_module_py(tauv_msgs
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(tauv_msgs_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(tauv_msgs_generate_messages tauv_msgs_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmReport.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/AlarmWithMessage.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerCommand.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ControllerDebug.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FeatureDetections.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/FluidDepth.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsTuning.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParametersEstimate.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/DynamicsParameterConfigUpdate.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/GateDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Message.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/SonarPulse.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Servos.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/RegisterMeasurement.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PoseGraphMeasurement.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/ReadableAlarmReport.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Thrust.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TrajPoint.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/Battery.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuSync.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/XsensImuData.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/TeledyneDvlData.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/NavigationState.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDTuning.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDDebug.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PIDPlannerDebug.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/msg/PingDetection.msg" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/FeatureDetectionsSync.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SonarControl.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFind.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindOne.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/MapFindClosest.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetCameraInfo.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SyncAlarms.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneDynamics.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TuneController.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/TunePIDPlanner.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/GetTrajectory.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetTargetPose.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/SetPose.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/RunMission.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/jiaxi/Projects/aCube/planner/TAUV-ROS-Packages/src/packages/tauv_msgs/srv/UpdateDynamicsParameterConfigs.srv" NAME_WE)
add_dependencies(tauv_msgs_generate_messages_py _tauv_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(tauv_msgs_genpy)
add_dependencies(tauv_msgs_genpy tauv_msgs_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS tauv_msgs_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/tauv_msgs
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(tauv_msgs_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(tauv_msgs_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(tauv_msgs_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET vision_msgs_generate_messages_cpp)
  add_dependencies(tauv_msgs_generate_messages_cpp vision_msgs_generate_messages_cpp)
endif()
if(TARGET jsk_recognition_msgs_generate_messages_cpp)
  add_dependencies(tauv_msgs_generate_messages_cpp jsk_recognition_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/tauv_msgs
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(tauv_msgs_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(tauv_msgs_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(tauv_msgs_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET vision_msgs_generate_messages_eus)
  add_dependencies(tauv_msgs_generate_messages_eus vision_msgs_generate_messages_eus)
endif()
if(TARGET jsk_recognition_msgs_generate_messages_eus)
  add_dependencies(tauv_msgs_generate_messages_eus jsk_recognition_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/tauv_msgs
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(tauv_msgs_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(tauv_msgs_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(tauv_msgs_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET vision_msgs_generate_messages_lisp)
  add_dependencies(tauv_msgs_generate_messages_lisp vision_msgs_generate_messages_lisp)
endif()
if(TARGET jsk_recognition_msgs_generate_messages_lisp)
  add_dependencies(tauv_msgs_generate_messages_lisp jsk_recognition_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/tauv_msgs
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(tauv_msgs_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(tauv_msgs_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(tauv_msgs_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET vision_msgs_generate_messages_nodejs)
  add_dependencies(tauv_msgs_generate_messages_nodejs vision_msgs_generate_messages_nodejs)
endif()
if(TARGET jsk_recognition_msgs_generate_messages_nodejs)
  add_dependencies(tauv_msgs_generate_messages_nodejs jsk_recognition_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/tauv_msgs
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(tauv_msgs_generate_messages_py geometry_msgs_generate_messages_py)
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(tauv_msgs_generate_messages_py sensor_msgs_generate_messages_py)
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(tauv_msgs_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET vision_msgs_generate_messages_py)
  add_dependencies(tauv_msgs_generate_messages_py vision_msgs_generate_messages_py)
endif()
if(TARGET jsk_recognition_msgs_generate_messages_py)
  add_dependencies(tauv_msgs_generate_messages_py jsk_recognition_msgs_generate_messages_py)
endif()
