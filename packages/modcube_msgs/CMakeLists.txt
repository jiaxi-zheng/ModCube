cmake_minimum_required(VERSION 3.0.2)
project(modcube_msgs)

find_package(catkin REQUIRED geometry_msgs sensor_msgs std_msgs vision_msgs jsk_recognition_msgs message_generation)


add_message_files(
  FILES
  AlarmReport.msg
  AlarmWithMessage.msg
  AprilTagDetection.msg
  AprilTagDetectionArray.msg
  ControllerCommand.msg
  ControllerDebug.msg
  FeatureDetection.msg
  FeatureDetections.msg
  FluidDepth.msg
  DynamicsTuning.msg
  DynamicsParametersEstimate.msg
  DynamicsParameterConfigUpdate.msg
  GateDetection.msg
  Message.msg
  SonarPulse.msg
  Servos.msg
  RegisterMeasurement.msg
  Pose.msg
  PoseGraphMeasurement.msg
  ReadableAlarmReport.msg
  Thrust.msg
  TrajPoint.msg
  Battery.msg
  XsensImuSync.msg
  XsensImuData.msg
  ImuData.msg
  TeledyneDvlData.msg
  NavigationState.msg
  PIDTuning.msg
  PIDDebug.msg
  PIDPlannerDebug.msg
  PingDetection.msg
  Ctrl_cmd.msg
  Can.msg
)


add_service_files(
  FILES
  FeatureDetectionsSync.srv
  SonarControl.srv
  MapFind.srv
  MapFindOne.srv
  MapFindClosest.srv
  GetCameraInfo.srv
  SyncAlarms.srv
  TuneDynamics.srv
  TuneController.srv
  # GetTrajectoryResponse.srv
  # GetTrajectoryRequest.srv
  TunePIDPlanner.srv
  GetTrajectory.srv
  SetTargetPose.srv
  SetPose.srv
  RunMission.srv
  UpdateDynamicsParameterConfigs.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
  std_msgs
  vision_msgs
  jsk_recognition_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES
  CATKIN_DEPENDS
  geometry_msgs
  sensor_msgs
  vision_msgs
  jsk_recognition_msgs
  std_msgs
  message_runtime
# DEPENDS
)
