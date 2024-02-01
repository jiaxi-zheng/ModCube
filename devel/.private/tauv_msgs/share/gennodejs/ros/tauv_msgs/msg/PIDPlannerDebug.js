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

class PIDPlannerDebug {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.stamp = null;
      this.x = null;
      this.y = null;
      this.z = null;
      this.roll = null;
      this.pitch = null;
      this.yaw = null;
    }
    else {
      if (initObj.hasOwnProperty('stamp')) {
        this.stamp = initObj.stamp
      }
      else {
        this.stamp = {secs: 0, nsecs: 0};
      }
      if (initObj.hasOwnProperty('x')) {
        this.x = initObj.x
      }
      else {
        this.x = new PIDDebug();
      }
      if (initObj.hasOwnProperty('y')) {
        this.y = initObj.y
      }
      else {
        this.y = new PIDDebug();
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
      if (initObj.hasOwnProperty('yaw')) {
        this.yaw = initObj.yaw
      }
      else {
        this.yaw = new PIDDebug();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type PIDPlannerDebug
    // Serialize message field [stamp]
    bufferOffset = _serializer.time(obj.stamp, buffer, bufferOffset);
    // Serialize message field [x]
    bufferOffset = PIDDebug.serialize(obj.x, buffer, bufferOffset);
    // Serialize message field [y]
    bufferOffset = PIDDebug.serialize(obj.y, buffer, bufferOffset);
    // Serialize message field [z]
    bufferOffset = PIDDebug.serialize(obj.z, buffer, bufferOffset);
    // Serialize message field [roll]
    bufferOffset = PIDDebug.serialize(obj.roll, buffer, bufferOffset);
    // Serialize message field [pitch]
    bufferOffset = PIDDebug.serialize(obj.pitch, buffer, bufferOffset);
    // Serialize message field [yaw]
    bufferOffset = PIDDebug.serialize(obj.yaw, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type PIDPlannerDebug
    let len;
    let data = new PIDPlannerDebug(null);
    // Deserialize message field [stamp]
    data.stamp = _deserializer.time(buffer, bufferOffset);
    // Deserialize message field [x]
    data.x = PIDDebug.deserialize(buffer, bufferOffset);
    // Deserialize message field [y]
    data.y = PIDDebug.deserialize(buffer, bufferOffset);
    // Deserialize message field [z]
    data.z = PIDDebug.deserialize(buffer, bufferOffset);
    // Deserialize message field [roll]
    data.roll = PIDDebug.deserialize(buffer, bufferOffset);
    // Deserialize message field [pitch]
    data.pitch = PIDDebug.deserialize(buffer, bufferOffset);
    // Deserialize message field [yaw]
    data.yaw = PIDDebug.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += PIDDebug.getMessageSize(object.x);
    length += PIDDebug.getMessageSize(object.y);
    length += PIDDebug.getMessageSize(object.z);
    length += PIDDebug.getMessageSize(object.roll);
    length += PIDDebug.getMessageSize(object.pitch);
    length += PIDDebug.getMessageSize(object.yaw);
    return length + 8;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/PIDPlannerDebug';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '0f555c58b06b2c2d39d97456d9e34ef3';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    time stamp
    
    PIDDebug x
    PIDDebug y
    PIDDebug z
    PIDDebug roll
    PIDDebug pitch
    PIDDebug yaw
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
    const resolved = new PIDPlannerDebug(null);
    if (msg.stamp !== undefined) {
      resolved.stamp = msg.stamp;
    }
    else {
      resolved.stamp = {secs: 0, nsecs: 0}
    }

    if (msg.x !== undefined) {
      resolved.x = PIDDebug.Resolve(msg.x)
    }
    else {
      resolved.x = new PIDDebug()
    }

    if (msg.y !== undefined) {
      resolved.y = PIDDebug.Resolve(msg.y)
    }
    else {
      resolved.y = new PIDDebug()
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

    if (msg.yaw !== undefined) {
      resolved.yaw = PIDDebug.Resolve(msg.yaw)
    }
    else {
      resolved.yaw = new PIDDebug()
    }

    return resolved;
    }
};

module.exports = PIDPlannerDebug;
