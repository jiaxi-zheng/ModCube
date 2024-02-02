
"use strict";

let TuneDynamics = require('./TuneDynamics.js')
let SyncAlarms = require('./SyncAlarms.js')
let GetCameraInfo = require('./GetCameraInfo.js')
let TunePIDPlanner = require('./TunePIDPlanner.js')
let GetTrajectoryResponse_ = require('./GetTrajectoryResponse_.js')
let GetTrajectoryRequest_ = require('./GetTrajectoryRequest_.js')
let SetTargetPose = require('./SetTargetPose.js')
let SonarControl = require('./SonarControl.js')
let RunMission = require('./RunMission.js')
let SetPose = require('./SetPose.js')
let FeatureDetectionsSync = require('./FeatureDetectionsSync.js')
let MapFindOne = require('./MapFindOne.js')
let GetTrajectory = require('./GetTrajectory.js')
let UpdateDynamicsParameterConfigs = require('./UpdateDynamicsParameterConfigs.js')
let MapFindClosest = require('./MapFindClosest.js')
let TuneController = require('./TuneController.js')
let MapFind = require('./MapFind.js')

module.exports = {
  TuneDynamics: TuneDynamics,
  SyncAlarms: SyncAlarms,
  GetCameraInfo: GetCameraInfo,
  TunePIDPlanner: TunePIDPlanner,
  GetTrajectoryResponse_: GetTrajectoryResponse_,
  GetTrajectoryRequest_: GetTrajectoryRequest_,
  SetTargetPose: SetTargetPose,
  SonarControl: SonarControl,
  RunMission: RunMission,
  SetPose: SetPose,
  FeatureDetectionsSync: FeatureDetectionsSync,
  MapFindOne: MapFindOne,
  GetTrajectory: GetTrajectory,
  UpdateDynamicsParameterConfigs: UpdateDynamicsParameterConfigs,
  MapFindClosest: MapFindClosest,
  TuneController: TuneController,
  MapFind: MapFind,
};
