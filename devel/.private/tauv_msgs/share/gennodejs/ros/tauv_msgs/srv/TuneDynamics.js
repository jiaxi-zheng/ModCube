// Auto-generated. Do not edit!

// (in-package tauv_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let DynamicsTuning = require('../msg/DynamicsTuning.js');

//-----------------------------------------------------------


//-----------------------------------------------------------

class TuneDynamicsRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.tuning = null;
    }
    else {
      if (initObj.hasOwnProperty('tuning')) {
        this.tuning = initObj.tuning
      }
      else {
        this.tuning = new DynamicsTuning();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type TuneDynamicsRequest
    // Serialize message field [tuning]
    bufferOffset = DynamicsTuning.serialize(obj.tuning, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TuneDynamicsRequest
    let len;
    let data = new TuneDynamicsRequest(null);
    // Deserialize message field [tuning]
    data.tuning = DynamicsTuning.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 273;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/TuneDynamicsRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '75519b80dff65935bfae9b6c04b2e1d8';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    tauv_msgs/DynamicsTuning tuning
    
    ================================================================================
    MSG: tauv_msgs/DynamicsTuning
    bool update_mass
    float64 mass
    bool update_volume
    float64 volume
    bool update_water_density
    float64 water_density
    bool update_center_of_gravity
    float64[3] center_of_gravity
    bool update_center_of_buoyancy
    float64[3] center_of_buoyancy
    bool update_moments
    float64[6] moments
    bool update_linear_damping
    float64[6] linear_damping
    bool update_quadratic_damping
    float64[6] quadratic_damping
    bool update_added_mass
    float64[6] added_mass
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new TuneDynamicsRequest(null);
    if (msg.tuning !== undefined) {
      resolved.tuning = DynamicsTuning.Resolve(msg.tuning)
    }
    else {
      resolved.tuning = new DynamicsTuning()
    }

    return resolved;
    }
};

class TuneDynamicsResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.success = null;
    }
    else {
      if (initObj.hasOwnProperty('success')) {
        this.success = initObj.success
      }
      else {
        this.success = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type TuneDynamicsResponse
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TuneDynamicsResponse
    let len;
    let data = new TuneDynamicsResponse(null);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/TuneDynamicsResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '358e233cde0c8a8bcfea4ce193f8fc15';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool success
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new TuneDynamicsResponse(null);
    if (msg.success !== undefined) {
      resolved.success = msg.success;
    }
    else {
      resolved.success = false
    }

    return resolved;
    }
};

module.exports = {
  Request: TuneDynamicsRequest,
  Response: TuneDynamicsResponse,
  md5sum() { return '7d0bd5eafc7372a83c5502516a094dbc'; },
  datatype() { return 'tauv_msgs/TuneDynamics'; }
};
