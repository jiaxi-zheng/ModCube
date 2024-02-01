// Auto-generated. Do not edit!

// (in-package tauv_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let PIDTuning = require('./PIDTuning.js');

//-----------------------------------------------------------

class PIDDebug {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.tuning = null;
      this.value = null;
      this.setpoint = null;
      this.error = null;
      this.proportional = null;
      this.integral = null;
      this.derivative = null;
      this.effort = null;
    }
    else {
      if (initObj.hasOwnProperty('tuning')) {
        this.tuning = initObj.tuning
      }
      else {
        this.tuning = new PIDTuning();
      }
      if (initObj.hasOwnProperty('value')) {
        this.value = initObj.value
      }
      else {
        this.value = 0.0;
      }
      if (initObj.hasOwnProperty('setpoint')) {
        this.setpoint = initObj.setpoint
      }
      else {
        this.setpoint = 0.0;
      }
      if (initObj.hasOwnProperty('error')) {
        this.error = initObj.error
      }
      else {
        this.error = 0.0;
      }
      if (initObj.hasOwnProperty('proportional')) {
        this.proportional = initObj.proportional
      }
      else {
        this.proportional = 0.0;
      }
      if (initObj.hasOwnProperty('integral')) {
        this.integral = initObj.integral
      }
      else {
        this.integral = 0.0;
      }
      if (initObj.hasOwnProperty('derivative')) {
        this.derivative = initObj.derivative
      }
      else {
        this.derivative = 0.0;
      }
      if (initObj.hasOwnProperty('effort')) {
        this.effort = initObj.effort
      }
      else {
        this.effort = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type PIDDebug
    // Serialize message field [tuning]
    bufferOffset = PIDTuning.serialize(obj.tuning, buffer, bufferOffset);
    // Serialize message field [value]
    bufferOffset = _serializer.float64(obj.value, buffer, bufferOffset);
    // Serialize message field [setpoint]
    bufferOffset = _serializer.float64(obj.setpoint, buffer, bufferOffset);
    // Serialize message field [error]
    bufferOffset = _serializer.float64(obj.error, buffer, bufferOffset);
    // Serialize message field [proportional]
    bufferOffset = _serializer.float64(obj.proportional, buffer, bufferOffset);
    // Serialize message field [integral]
    bufferOffset = _serializer.float64(obj.integral, buffer, bufferOffset);
    // Serialize message field [derivative]
    bufferOffset = _serializer.float64(obj.derivative, buffer, bufferOffset);
    // Serialize message field [effort]
    bufferOffset = _serializer.float64(obj.effort, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type PIDDebug
    let len;
    let data = new PIDDebug(null);
    // Deserialize message field [tuning]
    data.tuning = PIDTuning.deserialize(buffer, bufferOffset);
    // Deserialize message field [value]
    data.value = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [setpoint]
    data.setpoint = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [error]
    data.error = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [proportional]
    data.proportional = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [integral]
    data.integral = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [derivative]
    data.derivative = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [effort]
    data.effort = _deserializer.float64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += PIDTuning.getMessageSize(object.tuning);
    return length + 56;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/PIDDebug';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '16ed0a79b3cb076d76fdb91c1c560d6a';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    PIDTuning tuning
    
    float64 value
    float64 setpoint
    float64 error
    float64 proportional
    float64 integral
    float64 derivative
    float64 effort
    ================================================================================
    MSG: tauv_msgs/PIDTuning
    string axis
    float64 kp
    float64 ki
    float64 kd
    float64 tau
    float64[2] limits
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new PIDDebug(null);
    if (msg.tuning !== undefined) {
      resolved.tuning = PIDTuning.Resolve(msg.tuning)
    }
    else {
      resolved.tuning = new PIDTuning()
    }

    if (msg.value !== undefined) {
      resolved.value = msg.value;
    }
    else {
      resolved.value = 0.0
    }

    if (msg.setpoint !== undefined) {
      resolved.setpoint = msg.setpoint;
    }
    else {
      resolved.setpoint = 0.0
    }

    if (msg.error !== undefined) {
      resolved.error = msg.error;
    }
    else {
      resolved.error = 0.0
    }

    if (msg.proportional !== undefined) {
      resolved.proportional = msg.proportional;
    }
    else {
      resolved.proportional = 0.0
    }

    if (msg.integral !== undefined) {
      resolved.integral = msg.integral;
    }
    else {
      resolved.integral = 0.0
    }

    if (msg.derivative !== undefined) {
      resolved.derivative = msg.derivative;
    }
    else {
      resolved.derivative = 0.0
    }

    if (msg.effort !== undefined) {
      resolved.effort = msg.effort;
    }
    else {
      resolved.effort = 0.0
    }

    return resolved;
    }
};

module.exports = PIDDebug;
