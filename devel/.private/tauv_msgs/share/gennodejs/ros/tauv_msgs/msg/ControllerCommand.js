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

class ControllerCommand {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.a_x = null;
      this.a_y = null;
      this.a_z = null;
      this.a_roll = null;
      this.a_pitch = null;
      this.a_yaw = null;
      this.f_x = null;
      this.f_y = null;
      this.f_z = null;
      this.f_roll = null;
      this.f_pitch = null;
      this.f_yaw = null;
      this.use_f_x = null;
      this.use_f_y = null;
      this.use_f_z = null;
      this.use_f_roll = null;
      this.use_f_pitch = null;
      this.use_f_yaw = null;
      this.setpoint_z = null;
      this.setpoint_roll = null;
      this.setpoint_pitch = null;
      this.use_setpoint_z = null;
      this.use_setpoint_roll = null;
      this.use_setpoint_pitch = null;
    }
    else {
      if (initObj.hasOwnProperty('a_x')) {
        this.a_x = initObj.a_x
      }
      else {
        this.a_x = 0.0;
      }
      if (initObj.hasOwnProperty('a_y')) {
        this.a_y = initObj.a_y
      }
      else {
        this.a_y = 0.0;
      }
      if (initObj.hasOwnProperty('a_z')) {
        this.a_z = initObj.a_z
      }
      else {
        this.a_z = 0.0;
      }
      if (initObj.hasOwnProperty('a_roll')) {
        this.a_roll = initObj.a_roll
      }
      else {
        this.a_roll = 0.0;
      }
      if (initObj.hasOwnProperty('a_pitch')) {
        this.a_pitch = initObj.a_pitch
      }
      else {
        this.a_pitch = 0.0;
      }
      if (initObj.hasOwnProperty('a_yaw')) {
        this.a_yaw = initObj.a_yaw
      }
      else {
        this.a_yaw = 0.0;
      }
      if (initObj.hasOwnProperty('f_x')) {
        this.f_x = initObj.f_x
      }
      else {
        this.f_x = 0.0;
      }
      if (initObj.hasOwnProperty('f_y')) {
        this.f_y = initObj.f_y
      }
      else {
        this.f_y = 0.0;
      }
      if (initObj.hasOwnProperty('f_z')) {
        this.f_z = initObj.f_z
      }
      else {
        this.f_z = 0.0;
      }
      if (initObj.hasOwnProperty('f_roll')) {
        this.f_roll = initObj.f_roll
      }
      else {
        this.f_roll = 0.0;
      }
      if (initObj.hasOwnProperty('f_pitch')) {
        this.f_pitch = initObj.f_pitch
      }
      else {
        this.f_pitch = 0.0;
      }
      if (initObj.hasOwnProperty('f_yaw')) {
        this.f_yaw = initObj.f_yaw
      }
      else {
        this.f_yaw = 0.0;
      }
      if (initObj.hasOwnProperty('use_f_x')) {
        this.use_f_x = initObj.use_f_x
      }
      else {
        this.use_f_x = false;
      }
      if (initObj.hasOwnProperty('use_f_y')) {
        this.use_f_y = initObj.use_f_y
      }
      else {
        this.use_f_y = false;
      }
      if (initObj.hasOwnProperty('use_f_z')) {
        this.use_f_z = initObj.use_f_z
      }
      else {
        this.use_f_z = false;
      }
      if (initObj.hasOwnProperty('use_f_roll')) {
        this.use_f_roll = initObj.use_f_roll
      }
      else {
        this.use_f_roll = false;
      }
      if (initObj.hasOwnProperty('use_f_pitch')) {
        this.use_f_pitch = initObj.use_f_pitch
      }
      else {
        this.use_f_pitch = false;
      }
      if (initObj.hasOwnProperty('use_f_yaw')) {
        this.use_f_yaw = initObj.use_f_yaw
      }
      else {
        this.use_f_yaw = false;
      }
      if (initObj.hasOwnProperty('setpoint_z')) {
        this.setpoint_z = initObj.setpoint_z
      }
      else {
        this.setpoint_z = 0.0;
      }
      if (initObj.hasOwnProperty('setpoint_roll')) {
        this.setpoint_roll = initObj.setpoint_roll
      }
      else {
        this.setpoint_roll = 0.0;
      }
      if (initObj.hasOwnProperty('setpoint_pitch')) {
        this.setpoint_pitch = initObj.setpoint_pitch
      }
      else {
        this.setpoint_pitch = 0.0;
      }
      if (initObj.hasOwnProperty('use_setpoint_z')) {
        this.use_setpoint_z = initObj.use_setpoint_z
      }
      else {
        this.use_setpoint_z = false;
      }
      if (initObj.hasOwnProperty('use_setpoint_roll')) {
        this.use_setpoint_roll = initObj.use_setpoint_roll
      }
      else {
        this.use_setpoint_roll = false;
      }
      if (initObj.hasOwnProperty('use_setpoint_pitch')) {
        this.use_setpoint_pitch = initObj.use_setpoint_pitch
      }
      else {
        this.use_setpoint_pitch = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ControllerCommand
    // Serialize message field [a_x]
    bufferOffset = _serializer.float32(obj.a_x, buffer, bufferOffset);
    // Serialize message field [a_y]
    bufferOffset = _serializer.float32(obj.a_y, buffer, bufferOffset);
    // Serialize message field [a_z]
    bufferOffset = _serializer.float32(obj.a_z, buffer, bufferOffset);
    // Serialize message field [a_roll]
    bufferOffset = _serializer.float32(obj.a_roll, buffer, bufferOffset);
    // Serialize message field [a_pitch]
    bufferOffset = _serializer.float32(obj.a_pitch, buffer, bufferOffset);
    // Serialize message field [a_yaw]
    bufferOffset = _serializer.float32(obj.a_yaw, buffer, bufferOffset);
    // Serialize message field [f_x]
    bufferOffset = _serializer.float32(obj.f_x, buffer, bufferOffset);
    // Serialize message field [f_y]
    bufferOffset = _serializer.float32(obj.f_y, buffer, bufferOffset);
    // Serialize message field [f_z]
    bufferOffset = _serializer.float32(obj.f_z, buffer, bufferOffset);
    // Serialize message field [f_roll]
    bufferOffset = _serializer.float32(obj.f_roll, buffer, bufferOffset);
    // Serialize message field [f_pitch]
    bufferOffset = _serializer.float32(obj.f_pitch, buffer, bufferOffset);
    // Serialize message field [f_yaw]
    bufferOffset = _serializer.float32(obj.f_yaw, buffer, bufferOffset);
    // Serialize message field [use_f_x]
    bufferOffset = _serializer.bool(obj.use_f_x, buffer, bufferOffset);
    // Serialize message field [use_f_y]
    bufferOffset = _serializer.bool(obj.use_f_y, buffer, bufferOffset);
    // Serialize message field [use_f_z]
    bufferOffset = _serializer.bool(obj.use_f_z, buffer, bufferOffset);
    // Serialize message field [use_f_roll]
    bufferOffset = _serializer.bool(obj.use_f_roll, buffer, bufferOffset);
    // Serialize message field [use_f_pitch]
    bufferOffset = _serializer.bool(obj.use_f_pitch, buffer, bufferOffset);
    // Serialize message field [use_f_yaw]
    bufferOffset = _serializer.bool(obj.use_f_yaw, buffer, bufferOffset);
    // Serialize message field [setpoint_z]
    bufferOffset = _serializer.float32(obj.setpoint_z, buffer, bufferOffset);
    // Serialize message field [setpoint_roll]
    bufferOffset = _serializer.float32(obj.setpoint_roll, buffer, bufferOffset);
    // Serialize message field [setpoint_pitch]
    bufferOffset = _serializer.float32(obj.setpoint_pitch, buffer, bufferOffset);
    // Serialize message field [use_setpoint_z]
    bufferOffset = _serializer.bool(obj.use_setpoint_z, buffer, bufferOffset);
    // Serialize message field [use_setpoint_roll]
    bufferOffset = _serializer.bool(obj.use_setpoint_roll, buffer, bufferOffset);
    // Serialize message field [use_setpoint_pitch]
    bufferOffset = _serializer.bool(obj.use_setpoint_pitch, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ControllerCommand
    let len;
    let data = new ControllerCommand(null);
    // Deserialize message field [a_x]
    data.a_x = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [a_y]
    data.a_y = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [a_z]
    data.a_z = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [a_roll]
    data.a_roll = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [a_pitch]
    data.a_pitch = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [a_yaw]
    data.a_yaw = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [f_x]
    data.f_x = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [f_y]
    data.f_y = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [f_z]
    data.f_z = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [f_roll]
    data.f_roll = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [f_pitch]
    data.f_pitch = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [f_yaw]
    data.f_yaw = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [use_f_x]
    data.use_f_x = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [use_f_y]
    data.use_f_y = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [use_f_z]
    data.use_f_z = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [use_f_roll]
    data.use_f_roll = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [use_f_pitch]
    data.use_f_pitch = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [use_f_yaw]
    data.use_f_yaw = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [setpoint_z]
    data.setpoint_z = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [setpoint_roll]
    data.setpoint_roll = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [setpoint_pitch]
    data.setpoint_pitch = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [use_setpoint_z]
    data.use_setpoint_z = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [use_setpoint_roll]
    data.use_setpoint_roll = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [use_setpoint_pitch]
    data.use_setpoint_pitch = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 69;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/ControllerCommand';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'ad1b57ce703dafd167a4f28711d03e4e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Accelerations
    float32 a_x
    float32 a_y
    float32 a_z
    float32 a_roll
    float32 a_pitch
    float32 a_yaw
    
    # Forces
    float32 f_x
    float32 f_y
    float32 f_z
    float32 f_roll
    float32 f_pitch
    float32 f_yaw
    
    # If set, override accelerations and use forces
    bool use_f_x
    bool use_f_y
    bool use_f_z
    bool use_f_roll
    bool use_f_pitch
    bool use_f_yaw
    
    # Setpoints
    float32 setpoint_z
    float32 setpoint_roll
    float32 setpoint_pitch
    
    # If set, override accelerations and forces and use setpoints
    bool use_setpoint_z
    bool use_setpoint_roll
    bool use_setpoint_pitch
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ControllerCommand(null);
    if (msg.a_x !== undefined) {
      resolved.a_x = msg.a_x;
    }
    else {
      resolved.a_x = 0.0
    }

    if (msg.a_y !== undefined) {
      resolved.a_y = msg.a_y;
    }
    else {
      resolved.a_y = 0.0
    }

    if (msg.a_z !== undefined) {
      resolved.a_z = msg.a_z;
    }
    else {
      resolved.a_z = 0.0
    }

    if (msg.a_roll !== undefined) {
      resolved.a_roll = msg.a_roll;
    }
    else {
      resolved.a_roll = 0.0
    }

    if (msg.a_pitch !== undefined) {
      resolved.a_pitch = msg.a_pitch;
    }
    else {
      resolved.a_pitch = 0.0
    }

    if (msg.a_yaw !== undefined) {
      resolved.a_yaw = msg.a_yaw;
    }
    else {
      resolved.a_yaw = 0.0
    }

    if (msg.f_x !== undefined) {
      resolved.f_x = msg.f_x;
    }
    else {
      resolved.f_x = 0.0
    }

    if (msg.f_y !== undefined) {
      resolved.f_y = msg.f_y;
    }
    else {
      resolved.f_y = 0.0
    }

    if (msg.f_z !== undefined) {
      resolved.f_z = msg.f_z;
    }
    else {
      resolved.f_z = 0.0
    }

    if (msg.f_roll !== undefined) {
      resolved.f_roll = msg.f_roll;
    }
    else {
      resolved.f_roll = 0.0
    }

    if (msg.f_pitch !== undefined) {
      resolved.f_pitch = msg.f_pitch;
    }
    else {
      resolved.f_pitch = 0.0
    }

    if (msg.f_yaw !== undefined) {
      resolved.f_yaw = msg.f_yaw;
    }
    else {
      resolved.f_yaw = 0.0
    }

    if (msg.use_f_x !== undefined) {
      resolved.use_f_x = msg.use_f_x;
    }
    else {
      resolved.use_f_x = false
    }

    if (msg.use_f_y !== undefined) {
      resolved.use_f_y = msg.use_f_y;
    }
    else {
      resolved.use_f_y = false
    }

    if (msg.use_f_z !== undefined) {
      resolved.use_f_z = msg.use_f_z;
    }
    else {
      resolved.use_f_z = false
    }

    if (msg.use_f_roll !== undefined) {
      resolved.use_f_roll = msg.use_f_roll;
    }
    else {
      resolved.use_f_roll = false
    }

    if (msg.use_f_pitch !== undefined) {
      resolved.use_f_pitch = msg.use_f_pitch;
    }
    else {
      resolved.use_f_pitch = false
    }

    if (msg.use_f_yaw !== undefined) {
      resolved.use_f_yaw = msg.use_f_yaw;
    }
    else {
      resolved.use_f_yaw = false
    }

    if (msg.setpoint_z !== undefined) {
      resolved.setpoint_z = msg.setpoint_z;
    }
    else {
      resolved.setpoint_z = 0.0
    }

    if (msg.setpoint_roll !== undefined) {
      resolved.setpoint_roll = msg.setpoint_roll;
    }
    else {
      resolved.setpoint_roll = 0.0
    }

    if (msg.setpoint_pitch !== undefined) {
      resolved.setpoint_pitch = msg.setpoint_pitch;
    }
    else {
      resolved.setpoint_pitch = 0.0
    }

    if (msg.use_setpoint_z !== undefined) {
      resolved.use_setpoint_z = msg.use_setpoint_z;
    }
    else {
      resolved.use_setpoint_z = false
    }

    if (msg.use_setpoint_roll !== undefined) {
      resolved.use_setpoint_roll = msg.use_setpoint_roll;
    }
    else {
      resolved.use_setpoint_roll = false
    }

    if (msg.use_setpoint_pitch !== undefined) {
      resolved.use_setpoint_pitch = msg.use_setpoint_pitch;
    }
    else {
      resolved.use_setpoint_pitch = false
    }

    return resolved;
    }
};

module.exports = ControllerCommand;
