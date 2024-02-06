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

class Can {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.voltage_current = null;
      this.temperature_cabin_current = null;
      this.rm_speed_rpm = null;
      this.rm_given_current = null;
      this.rm_total_angle = null;
      this.FB_auv_pit = null;
      this.FB_auv_rol = null;
      this.FB_auv_yaw = null;
      this.FB_auv_deep = null;
      this.FB_auv_deep_vel = null;
      this.FB_auv_ang_vel_pit = null;
      this.FB_auv_ang_vel_rol = null;
      this.FB_auv_ang_vel_yaw = null;
      this.PWM_set = null;
    }
    else {
      if (initObj.hasOwnProperty('voltage_current')) {
        this.voltage_current = initObj.voltage_current
      }
      else {
        this.voltage_current = 0;
      }
      if (initObj.hasOwnProperty('temperature_cabin_current')) {
        this.temperature_cabin_current = initObj.temperature_cabin_current
      }
      else {
        this.temperature_cabin_current = 0;
      }
      if (initObj.hasOwnProperty('rm_speed_rpm')) {
        this.rm_speed_rpm = initObj.rm_speed_rpm
      }
      else {
        this.rm_speed_rpm = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('rm_given_current')) {
        this.rm_given_current = initObj.rm_given_current
      }
      else {
        this.rm_given_current = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('rm_total_angle')) {
        this.rm_total_angle = initObj.rm_total_angle
      }
      else {
        this.rm_total_angle = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('FB_auv_pit')) {
        this.FB_auv_pit = initObj.FB_auv_pit
      }
      else {
        this.FB_auv_pit = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('FB_auv_rol')) {
        this.FB_auv_rol = initObj.FB_auv_rol
      }
      else {
        this.FB_auv_rol = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('FB_auv_yaw')) {
        this.FB_auv_yaw = initObj.FB_auv_yaw
      }
      else {
        this.FB_auv_yaw = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('FB_auv_deep')) {
        this.FB_auv_deep = initObj.FB_auv_deep
      }
      else {
        this.FB_auv_deep = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('FB_auv_deep_vel')) {
        this.FB_auv_deep_vel = initObj.FB_auv_deep_vel
      }
      else {
        this.FB_auv_deep_vel = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('FB_auv_ang_vel_pit')) {
        this.FB_auv_ang_vel_pit = initObj.FB_auv_ang_vel_pit
      }
      else {
        this.FB_auv_ang_vel_pit = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('FB_auv_ang_vel_rol')) {
        this.FB_auv_ang_vel_rol = initObj.FB_auv_ang_vel_rol
      }
      else {
        this.FB_auv_ang_vel_rol = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('FB_auv_ang_vel_yaw')) {
        this.FB_auv_ang_vel_yaw = initObj.FB_auv_ang_vel_yaw
      }
      else {
        this.FB_auv_ang_vel_yaw = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('PWM_set')) {
        this.PWM_set = initObj.PWM_set
      }
      else {
        this.PWM_set = new Array(8).fill(0);
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Can
    // Serialize message field [voltage_current]
    bufferOffset = _serializer.uint8(obj.voltage_current, buffer, bufferOffset);
    // Serialize message field [temperature_cabin_current]
    bufferOffset = _serializer.uint8(obj.temperature_cabin_current, buffer, bufferOffset);
    // Check that the constant length array field [rm_speed_rpm] has the right length
    if (obj.rm_speed_rpm.length !== 4) {
      throw new Error('Unable to serialize array field rm_speed_rpm - length must be 4')
    }
    // Serialize message field [rm_speed_rpm]
    bufferOffset = _arraySerializer.int16(obj.rm_speed_rpm, buffer, bufferOffset, 4);
    // Check that the constant length array field [rm_given_current] has the right length
    if (obj.rm_given_current.length !== 4) {
      throw new Error('Unable to serialize array field rm_given_current - length must be 4')
    }
    // Serialize message field [rm_given_current]
    bufferOffset = _arraySerializer.int16(obj.rm_given_current, buffer, bufferOffset, 4);
    // Check that the constant length array field [rm_total_angle] has the right length
    if (obj.rm_total_angle.length !== 4) {
      throw new Error('Unable to serialize array field rm_total_angle - length must be 4')
    }
    // Serialize message field [rm_total_angle]
    bufferOffset = _arraySerializer.int32(obj.rm_total_angle, buffer, bufferOffset, 4);
    // Check that the constant length array field [FB_auv_pit] has the right length
    if (obj.FB_auv_pit.length !== 4) {
      throw new Error('Unable to serialize array field FB_auv_pit - length must be 4')
    }
    // Serialize message field [FB_auv_pit]
    bufferOffset = _arraySerializer.int16(obj.FB_auv_pit, buffer, bufferOffset, 4);
    // Check that the constant length array field [FB_auv_rol] has the right length
    if (obj.FB_auv_rol.length !== 4) {
      throw new Error('Unable to serialize array field FB_auv_rol - length must be 4')
    }
    // Serialize message field [FB_auv_rol]
    bufferOffset = _arraySerializer.int16(obj.FB_auv_rol, buffer, bufferOffset, 4);
    // Check that the constant length array field [FB_auv_yaw] has the right length
    if (obj.FB_auv_yaw.length !== 4) {
      throw new Error('Unable to serialize array field FB_auv_yaw - length must be 4')
    }
    // Serialize message field [FB_auv_yaw]
    bufferOffset = _arraySerializer.int16(obj.FB_auv_yaw, buffer, bufferOffset, 4);
    // Check that the constant length array field [FB_auv_deep] has the right length
    if (obj.FB_auv_deep.length !== 4) {
      throw new Error('Unable to serialize array field FB_auv_deep - length must be 4')
    }
    // Serialize message field [FB_auv_deep]
    bufferOffset = _arraySerializer.int16(obj.FB_auv_deep, buffer, bufferOffset, 4);
    // Check that the constant length array field [FB_auv_deep_vel] has the right length
    if (obj.FB_auv_deep_vel.length !== 4) {
      throw new Error('Unable to serialize array field FB_auv_deep_vel - length must be 4')
    }
    // Serialize message field [FB_auv_deep_vel]
    bufferOffset = _arraySerializer.int16(obj.FB_auv_deep_vel, buffer, bufferOffset, 4);
    // Check that the constant length array field [FB_auv_ang_vel_pit] has the right length
    if (obj.FB_auv_ang_vel_pit.length !== 4) {
      throw new Error('Unable to serialize array field FB_auv_ang_vel_pit - length must be 4')
    }
    // Serialize message field [FB_auv_ang_vel_pit]
    bufferOffset = _arraySerializer.int16(obj.FB_auv_ang_vel_pit, buffer, bufferOffset, 4);
    // Check that the constant length array field [FB_auv_ang_vel_rol] has the right length
    if (obj.FB_auv_ang_vel_rol.length !== 4) {
      throw new Error('Unable to serialize array field FB_auv_ang_vel_rol - length must be 4')
    }
    // Serialize message field [FB_auv_ang_vel_rol]
    bufferOffset = _arraySerializer.int16(obj.FB_auv_ang_vel_rol, buffer, bufferOffset, 4);
    // Check that the constant length array field [FB_auv_ang_vel_yaw] has the right length
    if (obj.FB_auv_ang_vel_yaw.length !== 4) {
      throw new Error('Unable to serialize array field FB_auv_ang_vel_yaw - length must be 4')
    }
    // Serialize message field [FB_auv_ang_vel_yaw]
    bufferOffset = _arraySerializer.int16(obj.FB_auv_ang_vel_yaw, buffer, bufferOffset, 4);
    // Check that the constant length array field [PWM_set] has the right length
    if (obj.PWM_set.length !== 8) {
      throw new Error('Unable to serialize array field PWM_set - length must be 8')
    }
    // Serialize message field [PWM_set]
    bufferOffset = _arraySerializer.int16(obj.PWM_set, buffer, bufferOffset, 8);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Can
    let len;
    let data = new Can(null);
    // Deserialize message field [voltage_current]
    data.voltage_current = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [temperature_cabin_current]
    data.temperature_cabin_current = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [rm_speed_rpm]
    data.rm_speed_rpm = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    // Deserialize message field [rm_given_current]
    data.rm_given_current = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    // Deserialize message field [rm_total_angle]
    data.rm_total_angle = _arrayDeserializer.int32(buffer, bufferOffset, 4)
    // Deserialize message field [FB_auv_pit]
    data.FB_auv_pit = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    // Deserialize message field [FB_auv_rol]
    data.FB_auv_rol = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    // Deserialize message field [FB_auv_yaw]
    data.FB_auv_yaw = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    // Deserialize message field [FB_auv_deep]
    data.FB_auv_deep = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    // Deserialize message field [FB_auv_deep_vel]
    data.FB_auv_deep_vel = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    // Deserialize message field [FB_auv_ang_vel_pit]
    data.FB_auv_ang_vel_pit = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    // Deserialize message field [FB_auv_ang_vel_rol]
    data.FB_auv_ang_vel_rol = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    // Deserialize message field [FB_auv_ang_vel_yaw]
    data.FB_auv_ang_vel_yaw = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    // Deserialize message field [PWM_set]
    data.PWM_set = _arrayDeserializer.int16(buffer, bufferOffset, 8)
    return data;
  }

  static getMessageSize(object) {
    return 114;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/Can';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '38928799346a1e1fb02c8f1ad6011cc1';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    uint8 voltage_current
    uint8 temperature_cabin_current
    
    int16[4] rm_speed_rpm
    int16[4] rm_given_current
    int32[4] rm_total_angle
    
    int16[4] FB_auv_pit
    int16[4] FB_auv_rol
    int16[4] FB_auv_yaw
    int16[4] FB_auv_deep
    int16[4] FB_auv_deep_vel
    int16[4] FB_auv_ang_vel_pit
    int16[4] FB_auv_ang_vel_rol
    int16[4] FB_auv_ang_vel_yaw
    
    int16[8] PWM_set
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Can(null);
    if (msg.voltage_current !== undefined) {
      resolved.voltage_current = msg.voltage_current;
    }
    else {
      resolved.voltage_current = 0
    }

    if (msg.temperature_cabin_current !== undefined) {
      resolved.temperature_cabin_current = msg.temperature_cabin_current;
    }
    else {
      resolved.temperature_cabin_current = 0
    }

    if (msg.rm_speed_rpm !== undefined) {
      resolved.rm_speed_rpm = msg.rm_speed_rpm;
    }
    else {
      resolved.rm_speed_rpm = new Array(4).fill(0)
    }

    if (msg.rm_given_current !== undefined) {
      resolved.rm_given_current = msg.rm_given_current;
    }
    else {
      resolved.rm_given_current = new Array(4).fill(0)
    }

    if (msg.rm_total_angle !== undefined) {
      resolved.rm_total_angle = msg.rm_total_angle;
    }
    else {
      resolved.rm_total_angle = new Array(4).fill(0)
    }

    if (msg.FB_auv_pit !== undefined) {
      resolved.FB_auv_pit = msg.FB_auv_pit;
    }
    else {
      resolved.FB_auv_pit = new Array(4).fill(0)
    }

    if (msg.FB_auv_rol !== undefined) {
      resolved.FB_auv_rol = msg.FB_auv_rol;
    }
    else {
      resolved.FB_auv_rol = new Array(4).fill(0)
    }

    if (msg.FB_auv_yaw !== undefined) {
      resolved.FB_auv_yaw = msg.FB_auv_yaw;
    }
    else {
      resolved.FB_auv_yaw = new Array(4).fill(0)
    }

    if (msg.FB_auv_deep !== undefined) {
      resolved.FB_auv_deep = msg.FB_auv_deep;
    }
    else {
      resolved.FB_auv_deep = new Array(4).fill(0)
    }

    if (msg.FB_auv_deep_vel !== undefined) {
      resolved.FB_auv_deep_vel = msg.FB_auv_deep_vel;
    }
    else {
      resolved.FB_auv_deep_vel = new Array(4).fill(0)
    }

    if (msg.FB_auv_ang_vel_pit !== undefined) {
      resolved.FB_auv_ang_vel_pit = msg.FB_auv_ang_vel_pit;
    }
    else {
      resolved.FB_auv_ang_vel_pit = new Array(4).fill(0)
    }

    if (msg.FB_auv_ang_vel_rol !== undefined) {
      resolved.FB_auv_ang_vel_rol = msg.FB_auv_ang_vel_rol;
    }
    else {
      resolved.FB_auv_ang_vel_rol = new Array(4).fill(0)
    }

    if (msg.FB_auv_ang_vel_yaw !== undefined) {
      resolved.FB_auv_ang_vel_yaw = msg.FB_auv_ang_vel_yaw;
    }
    else {
      resolved.FB_auv_ang_vel_yaw = new Array(4).fill(0)
    }

    if (msg.PWM_set !== undefined) {
      resolved.PWM_set = msg.PWM_set;
    }
    else {
      resolved.PWM_set = new Array(8).fill(0)
    }

    return resolved;
    }
};

module.exports = Can;
