// Auto-generated. Do not edit!

// (in-package tauv_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class PIDTuning {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.axis = null;
      this.kp = null;
      this.ki = null;
      this.kd = null;
      this.tau = null;
      this.limits = null;
    }
    else {
      if (initObj.hasOwnProperty('axis')) {
        this.axis = initObj.axis
      }
      else {
        this.axis = '';
      }
      if (initObj.hasOwnProperty('kp')) {
        this.kp = initObj.kp
      }
      else {
        this.kp = 0.0;
      }
      if (initObj.hasOwnProperty('ki')) {
        this.ki = initObj.ki
      }
      else {
        this.ki = 0.0;
      }
      if (initObj.hasOwnProperty('kd')) {
        this.kd = initObj.kd
      }
      else {
        this.kd = 0.0;
      }
      if (initObj.hasOwnProperty('tau')) {
        this.tau = initObj.tau
      }
      else {
        this.tau = 0.0;
      }
      if (initObj.hasOwnProperty('limits')) {
        this.limits = initObj.limits
      }
      else {
        this.limits = new Array(2).fill(0);
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type PIDTuning
    // Serialize message field [axis]
    bufferOffset = _serializer.string(obj.axis, buffer, bufferOffset);
    // Serialize message field [kp]
    bufferOffset = _serializer.float64(obj.kp, buffer, bufferOffset);
    // Serialize message field [ki]
    bufferOffset = _serializer.float64(obj.ki, buffer, bufferOffset);
    // Serialize message field [kd]
    bufferOffset = _serializer.float64(obj.kd, buffer, bufferOffset);
    // Serialize message field [tau]
    bufferOffset = _serializer.float64(obj.tau, buffer, bufferOffset);
    // Check that the constant length array field [limits] has the right length
    if (obj.limits.length !== 2) {
      throw new Error('Unable to serialize array field limits - length must be 2')
    }
    // Serialize message field [limits]
    bufferOffset = _arraySerializer.float64(obj.limits, buffer, bufferOffset, 2);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type PIDTuning
    let len;
    let data = new PIDTuning(null);
    // Deserialize message field [axis]
    data.axis = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [kp]
    data.kp = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [ki]
    data.ki = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [kd]
    data.kd = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [tau]
    data.tau = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [limits]
    data.limits = _arrayDeserializer.float64(buffer, bufferOffset, 2)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.axis);
    return length + 52;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/PIDTuning';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '0c319ae505ebfe0fb1604e4f13fce3d5';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
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
    const resolved = new PIDTuning(null);
    if (msg.axis !== undefined) {
      resolved.axis = msg.axis;
    }
    else {
      resolved.axis = ''
    }

    if (msg.kp !== undefined) {
      resolved.kp = msg.kp;
    }
    else {
      resolved.kp = 0.0
    }

    if (msg.ki !== undefined) {
      resolved.ki = msg.ki;
    }
    else {
      resolved.ki = 0.0
    }

    if (msg.kd !== undefined) {
      resolved.kd = msg.kd;
    }
    else {
      resolved.kd = 0.0
    }

    if (msg.tau !== undefined) {
      resolved.tau = msg.tau;
    }
    else {
      resolved.tau = 0.0
    }

    if (msg.limits !== undefined) {
      resolved.limits = msg.limits;
    }
    else {
      resolved.limits = new Array(2).fill(0)
    }

    return resolved;
    }
};

module.exports = PIDTuning;
