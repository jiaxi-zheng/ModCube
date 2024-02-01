// Auto-generated. Do not edit!

// (in-package tauv_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let PIDDebug = require('./PIDDebug.js');

//-----------------------------------------------------------

class ControllerDebug {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.stamp = null;
      this.z = null;
      this.roll = null;
      this.pitch = null;
    }
    else {
      if (initObj.hasOwnProperty('stamp')) {
        this.stamp = initObj.stamp
      }
      else {
        this.stamp = {secs: 0, nsecs: 0};
      }
      if (initObj.hasOwnProperty('z')) {
        this.z = initObj.z
      }
      else {
        this.z = new PIDDebug();
      }
      if (initObj.hasOwnProperty('roll')) {
        this.roll = initObj.roll
      }
      else {
        this.roll = new PIDDebug();
      }
      if (initObj.hasOwnProperty('pitch')) {
        this.pitch = initObj.pitch
      }
      else {
        this.pitch = new PIDDebug();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ControllerDebug
    // Serialize message field [stamp]
    bufferOffset = _serializer.time(obj.stamp, buffer, bufferOffset);
    // Serialize message field [z]
    bufferOffset = PIDDebug.serialize(obj.z, buffer, bufferOffset);
    // Serialize message field [roll]
    bufferOffset = PIDDebug.serialize(obj.roll, buffer, bufferOffset);
    // Serialize message field [pitch]
    bufferOffset = PIDDebug.serialize(obj.pitch, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ControllerDebug
    let len;
    let data = new ControllerDebug(null);
    // Deserialize message field [stamp]
    data.stamp = _deserializer.time(buffer, bufferOffset);
    // Deserialize message field [z]
    data.z = PIDDebug.deserialize(buffer, bufferOffset);
    // Deserialize message field [roll]
    data.roll = PIDDebug.deserialize(buffer, bufferOffset);
    // Deserialize message field [pitch]
    data.pitch = PIDDebug.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += PIDDebug.getMessageSize(object.z);
    length += PIDDebug.getMessageSize(object.roll);
    length += PIDDebug.getMessageSize(object.pitch);
    return length + 8;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/ControllerDebug';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'bb322eef3e5c11d51e1cb54450d584b1';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    time stamp
    
    PIDDebug z
    PIDDebug roll
    PIDDebug pitch
    ================================================================================
    MSG: tauv_msgs/PIDDebug
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
    const resolved = new ControllerDebug(null);
    if (msg.stamp !== undefined) {
      resolved.stamp = msg.stamp;
    }
    else {
      resolved.stamp = {secs: 0, nsecs: 0}
    }

    if (msg.z !== undefined) {
      resolved.z = PIDDebug.Resolve(msg.z)
    }
    else {
      resolved.z = new PIDDebug()
    }

    if (msg.roll !== undefined) {
      resolved.roll = PIDDebug.Resolve(msg.roll)
    }
    else {
      resolved.roll = new PIDDebug()
    }

    if (msg.pitch !== undefined) {
      resolved.pitch = PIDDebug.Resolve(msg.pitch)
    }
    else {
      resolved.pitch = new PIDDebug()
    }

    return resolved;
    }
};

module.exports = ControllerDebug;
