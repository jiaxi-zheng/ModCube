
"use strict";

let GetCurrentModel = require('./GetCurrentModel.js')
let SetCurrentVelocity = require('./SetCurrentVelocity.js')
let SetOriginSphericalCoord = require('./SetOriginSphericalCoord.js')
let SetCurrentModel = require('./SetCurrentModel.js')
let SetCurrentDirection = require('./SetCurrentDirection.js')
let GetOriginSphericalCoord = require('./GetOriginSphericalCoord.js')
let TransformToSphericalCoord = require('./TransformToSphericalCoord.js')
let TransformFromSphericalCoord = require('./TransformFromSphericalCoord.js')

module.exports = {
  GetCurrentModel: GetCurrentModel,
  SetCurrentVelocity: SetCurrentVelocity,
  SetOriginSphericalCoord: SetOriginSphericalCoord,
  SetCurrentModel: SetCurrentModel,
  SetCurrentDirection: SetCurrentDirection,
  GetOriginSphericalCoord: GetOriginSphericalCoord,
  TransformToSphericalCoord: TransformToSphericalCoord,
  TransformFromSphericalCoord: TransformFromSphericalCoord,
};
