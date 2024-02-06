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

class Ctrl_cmd {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.Ctrl_vel_X = null;
      this.Ctrl_vel_Y = null;
      this.Ctrl_vel_Z = null;
      this.Ctrl_fixed_Z = null;
      this.Ctrl_vel_Rol = null;
      this.Ctrl_vel_Pit = null;
      this.Ctrl_vel_Yaw = null;
      this.Ctrl_fixed_Yaw = null;
      this.Ctrl_pivot_1 = null;
      this.Ctrl_pivot_2 = null;
      this.Ctrl_pivot_3 = null;
      this.Ctrl_pivot_4 = null;
      this.Ctrl_emagnet_1 = null;
      this.Ctrl_emagnet_2 = null;
      this.Ctrl_emagnet_3 = null;
      this.Ctrl_emagnet_4 = null;
      this.Ctrl_arm_joint_1 = null;
      this.Ctrl_arm_joint_2 = null;
      this.Joy_Button_Y = null;
      this.Joy_Button_X = null;
      this.Joy_Button_A = null;
      this.Joy_Button_B = null;
      this.Joy_Button_LB = null;
      this.Joy_Button_RB = null;
      this.Joy_Button_STICK_LEFT = null;
      this.Joy_Button_STICK_RIGHT = null;
      this.Reset_pwm = null;
    }
    else {
      if (initObj.hasOwnProperty('Ctrl_vel_X')) {
        this.Ctrl_vel_X = initObj.Ctrl_vel_X
      }
      else {
        this.Ctrl_vel_X = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Ctrl_vel_Y')) {
        this.Ctrl_vel_Y = initObj.Ctrl_vel_Y
      }
      else {
        this.Ctrl_vel_Y = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Ctrl_vel_Z')) {
        this.Ctrl_vel_Z = initObj.Ctrl_vel_Z
      }
      else {
        this.Ctrl_vel_Z = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Ctrl_fixed_Z')) {
        this.Ctrl_fixed_Z = initObj.Ctrl_fixed_Z
      }
      else {
        this.Ctrl_fixed_Z = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Ctrl_vel_Rol')) {
        this.Ctrl_vel_Rol = initObj.Ctrl_vel_Rol
      }
      else {
        this.Ctrl_vel_Rol = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Ctrl_vel_Pit')) {
        this.Ctrl_vel_Pit = initObj.Ctrl_vel_Pit
      }
      else {
        this.Ctrl_vel_Pit = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Ctrl_vel_Yaw')) {
        this.Ctrl_vel_Yaw = initObj.Ctrl_vel_Yaw
      }
      else {
        this.Ctrl_vel_Yaw = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Ctrl_fixed_Yaw')) {
        this.Ctrl_fixed_Yaw = initObj.Ctrl_fixed_Yaw
      }
      else {
        this.Ctrl_fixed_Yaw = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Ctrl_pivot_1')) {
        this.Ctrl_pivot_1 = initObj.Ctrl_pivot_1
      }
      else {
        this.Ctrl_pivot_1 = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Ctrl_pivot_2')) {
        this.Ctrl_pivot_2 = initObj.Ctrl_pivot_2
      }
      else {
        this.Ctrl_pivot_2 = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Ctrl_pivot_3')) {
        this.Ctrl_pivot_3 = initObj.Ctrl_pivot_3
      }
      else {
        this.Ctrl_pivot_3 = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Ctrl_pivot_4')) {
        this.Ctrl_pivot_4 = initObj.Ctrl_pivot_4
      }
      else {
        this.Ctrl_pivot_4 = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Ctrl_emagnet_1')) {
        this.Ctrl_emagnet_1 = initObj.Ctrl_emagnet_1
      }
      else {
        this.Ctrl_emagnet_1 = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Ctrl_emagnet_2')) {
        this.Ctrl_emagnet_2 = initObj.Ctrl_emagnet_2
      }
      else {
        this.Ctrl_emagnet_2 = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Ctrl_emagnet_3')) {
        this.Ctrl_emagnet_3 = initObj.Ctrl_emagnet_3
      }
      else {
        this.Ctrl_emagnet_3 = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Ctrl_emagnet_4')) {
        this.Ctrl_emagnet_4 = initObj.Ctrl_emagnet_4
      }
      else {
        this.Ctrl_emagnet_4 = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Ctrl_arm_joint_1')) {
        this.Ctrl_arm_joint_1 = initObj.Ctrl_arm_joint_1
      }
      else {
        this.Ctrl_arm_joint_1 = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Ctrl_arm_joint_2')) {
        this.Ctrl_arm_joint_2 = initObj.Ctrl_arm_joint_2
      }
      else {
        this.Ctrl_arm_joint_2 = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Joy_Button_Y')) {
        this.Joy_Button_Y = initObj.Joy_Button_Y
      }
      else {
        this.Joy_Button_Y = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Joy_Button_X')) {
        this.Joy_Button_X = initObj.Joy_Button_X
      }
      else {
        this.Joy_Button_X = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Joy_Button_A')) {
        this.Joy_Button_A = initObj.Joy_Button_A
      }
      else {
        this.Joy_Button_A = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Joy_Button_B')) {
        this.Joy_Button_B = initObj.Joy_Button_B
      }
      else {
        this.Joy_Button_B = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Joy_Button_LB')) {
        this.Joy_Button_LB = initObj.Joy_Button_LB
      }
      else {
        this.Joy_Button_LB = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Joy_Button_RB')) {
        this.Joy_Button_RB = initObj.Joy_Button_RB
      }
      else {
        this.Joy_Button_RB = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Joy_Button_STICK_LEFT')) {
        this.Joy_Button_STICK_LEFT = initObj.Joy_Button_STICK_LEFT
      }
      else {
        this.Joy_Button_STICK_LEFT = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Joy_Button_STICK_RIGHT')) {
        this.Joy_Button_STICK_RIGHT = initObj.Joy_Button_STICK_RIGHT
      }
      else {
        this.Joy_Button_STICK_RIGHT = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('Reset_pwm')) {
        this.Reset_pwm = initObj.Reset_pwm
      }
      else {
        this.Reset_pwm = new Array(4).fill(0);
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Ctrl_cmd
    // Check that the constant length array field [Ctrl_vel_X] has the right length
    if (obj.Ctrl_vel_X.length !== 4) {
      throw new Error('Unable to serialize array field Ctrl_vel_X - length must be 4')
    }
    // Serialize message field [Ctrl_vel_X]
    bufferOffset = _arraySerializer.uint8(obj.Ctrl_vel_X, buffer, bufferOffset, 4);
    // Check that the constant length array field [Ctrl_vel_Y] has the right length
    if (obj.Ctrl_vel_Y.length !== 4) {
      throw new Error('Unable to serialize array field Ctrl_vel_Y - length must be 4')
    }
    // Serialize message field [Ctrl_vel_Y]
    bufferOffset = _arraySerializer.uint8(obj.Ctrl_vel_Y, buffer, bufferOffset, 4);
    // Check that the constant length array field [Ctrl_vel_Z] has the right length
    if (obj.Ctrl_vel_Z.length !== 4) {
      throw new Error('Unable to serialize array field Ctrl_vel_Z - length must be 4')
    }
    // Serialize message field [Ctrl_vel_Z]
    bufferOffset = _arraySerializer.uint8(obj.Ctrl_vel_Z, buffer, bufferOffset, 4);
    // Check that the constant length array field [Ctrl_fixed_Z] has the right length
    if (obj.Ctrl_fixed_Z.length !== 4) {
      throw new Error('Unable to serialize array field Ctrl_fixed_Z - length must be 4')
    }
    // Serialize message field [Ctrl_fixed_Z]
    bufferOffset = _arraySerializer.uint8(obj.Ctrl_fixed_Z, buffer, bufferOffset, 4);
    // Check that the constant length array field [Ctrl_vel_Rol] has the right length
    if (obj.Ctrl_vel_Rol.length !== 4) {
      throw new Error('Unable to serialize array field Ctrl_vel_Rol - length must be 4')
    }
    // Serialize message field [Ctrl_vel_Rol]
    bufferOffset = _arraySerializer.uint8(obj.Ctrl_vel_Rol, buffer, bufferOffset, 4);
    // Check that the constant length array field [Ctrl_vel_Pit] has the right length
    if (obj.Ctrl_vel_Pit.length !== 4) {
      throw new Error('Unable to serialize array field Ctrl_vel_Pit - length must be 4')
    }
    // Serialize message field [Ctrl_vel_Pit]
    bufferOffset = _arraySerializer.uint8(obj.Ctrl_vel_Pit, buffer, bufferOffset, 4);
    // Check that the constant length array field [Ctrl_vel_Yaw] has the right length
    if (obj.Ctrl_vel_Yaw.length !== 4) {
      throw new Error('Unable to serialize array field Ctrl_vel_Yaw - length must be 4')
    }
    // Serialize message field [Ctrl_vel_Yaw]
    bufferOffset = _arraySerializer.uint8(obj.Ctrl_vel_Yaw, buffer, bufferOffset, 4);
    // Check that the constant length array field [Ctrl_fixed_Yaw] has the right length
    if (obj.Ctrl_fixed_Yaw.length !== 4) {
      throw new Error('Unable to serialize array field Ctrl_fixed_Yaw - length must be 4')
    }
    // Serialize message field [Ctrl_fixed_Yaw]
    bufferOffset = _arraySerializer.uint8(obj.Ctrl_fixed_Yaw, buffer, bufferOffset, 4);
    // Check that the constant length array field [Ctrl_pivot_1] has the right length
    if (obj.Ctrl_pivot_1.length !== 4) {
      throw new Error('Unable to serialize array field Ctrl_pivot_1 - length must be 4')
    }
    // Serialize message field [Ctrl_pivot_1]
    bufferOffset = _arraySerializer.uint8(obj.Ctrl_pivot_1, buffer, bufferOffset, 4);
    // Check that the constant length array field [Ctrl_pivot_2] has the right length
    if (obj.Ctrl_pivot_2.length !== 4) {
      throw new Error('Unable to serialize array field Ctrl_pivot_2 - length must be 4')
    }
    // Serialize message field [Ctrl_pivot_2]
    bufferOffset = _arraySerializer.uint8(obj.Ctrl_pivot_2, buffer, bufferOffset, 4);
    // Check that the constant length array field [Ctrl_pivot_3] has the right length
    if (obj.Ctrl_pivot_3.length !== 4) {
      throw new Error('Unable to serialize array field Ctrl_pivot_3 - length must be 4')
    }
    // Serialize message field [Ctrl_pivot_3]
    bufferOffset = _arraySerializer.uint8(obj.Ctrl_pivot_3, buffer, bufferOffset, 4);
    // Check that the constant length array field [Ctrl_pivot_4] has the right length
    if (obj.Ctrl_pivot_4.length !== 4) {
      throw new Error('Unable to serialize array field Ctrl_pivot_4 - length must be 4')
    }
    // Serialize message field [Ctrl_pivot_4]
    bufferOffset = _arraySerializer.uint8(obj.Ctrl_pivot_4, buffer, bufferOffset, 4);
    // Check that the constant length array field [Ctrl_emagnet_1] has the right length
    if (obj.Ctrl_emagnet_1.length !== 4) {
      throw new Error('Unable to serialize array field Ctrl_emagnet_1 - length must be 4')
    }
    // Serialize message field [Ctrl_emagnet_1]
    bufferOffset = _arraySerializer.uint8(obj.Ctrl_emagnet_1, buffer, bufferOffset, 4);
    // Check that the constant length array field [Ctrl_emagnet_2] has the right length
    if (obj.Ctrl_emagnet_2.length !== 4) {
      throw new Error('Unable to serialize array field Ctrl_emagnet_2 - length must be 4')
    }
    // Serialize message field [Ctrl_emagnet_2]
    bufferOffset = _arraySerializer.uint8(obj.Ctrl_emagnet_2, buffer, bufferOffset, 4);
    // Check that the constant length array field [Ctrl_emagnet_3] has the right length
    if (obj.Ctrl_emagnet_3.length !== 4) {
      throw new Error('Unable to serialize array field Ctrl_emagnet_3 - length must be 4')
    }
    // Serialize message field [Ctrl_emagnet_3]
    bufferOffset = _arraySerializer.uint8(obj.Ctrl_emagnet_3, buffer, bufferOffset, 4);
    // Check that the constant length array field [Ctrl_emagnet_4] has the right length
    if (obj.Ctrl_emagnet_4.length !== 4) {
      throw new Error('Unable to serialize array field Ctrl_emagnet_4 - length must be 4')
    }
    // Serialize message field [Ctrl_emagnet_4]
    bufferOffset = _arraySerializer.uint8(obj.Ctrl_emagnet_4, buffer, bufferOffset, 4);
    // Check that the constant length array field [Ctrl_arm_joint_1] has the right length
    if (obj.Ctrl_arm_joint_1.length !== 4) {
      throw new Error('Unable to serialize array field Ctrl_arm_joint_1 - length must be 4')
    }
    // Serialize message field [Ctrl_arm_joint_1]
    bufferOffset = _arraySerializer.uint8(obj.Ctrl_arm_joint_1, buffer, bufferOffset, 4);
    // Check that the constant length array field [Ctrl_arm_joint_2] has the right length
    if (obj.Ctrl_arm_joint_2.length !== 4) {
      throw new Error('Unable to serialize array field Ctrl_arm_joint_2 - length must be 4')
    }
    // Serialize message field [Ctrl_arm_joint_2]
    bufferOffset = _arraySerializer.uint8(obj.Ctrl_arm_joint_2, buffer, bufferOffset, 4);
    // Check that the constant length array field [Joy_Button_Y] has the right length
    if (obj.Joy_Button_Y.length !== 4) {
      throw new Error('Unable to serialize array field Joy_Button_Y - length must be 4')
    }
    // Serialize message field [Joy_Button_Y]
    bufferOffset = _arraySerializer.int16(obj.Joy_Button_Y, buffer, bufferOffset, 4);
    // Check that the constant length array field [Joy_Button_X] has the right length
    if (obj.Joy_Button_X.length !== 4) {
      throw new Error('Unable to serialize array field Joy_Button_X - length must be 4')
    }
    // Serialize message field [Joy_Button_X]
    bufferOffset = _arraySerializer.int16(obj.Joy_Button_X, buffer, bufferOffset, 4);
    // Check that the constant length array field [Joy_Button_A] has the right length
    if (obj.Joy_Button_A.length !== 4) {
      throw new Error('Unable to serialize array field Joy_Button_A - length must be 4')
    }
    // Serialize message field [Joy_Button_A]
    bufferOffset = _arraySerializer.int16(obj.Joy_Button_A, buffer, bufferOffset, 4);
    // Check that the constant length array field [Joy_Button_B] has the right length
    if (obj.Joy_Button_B.length !== 4) {
      throw new Error('Unable to serialize array field Joy_Button_B - length must be 4')
    }
    // Serialize message field [Joy_Button_B]
    bufferOffset = _arraySerializer.int16(obj.Joy_Button_B, buffer, bufferOffset, 4);
    // Check that the constant length array field [Joy_Button_LB] has the right length
    if (obj.Joy_Button_LB.length !== 4) {
      throw new Error('Unable to serialize array field Joy_Button_LB - length must be 4')
    }
    // Serialize message field [Joy_Button_LB]
    bufferOffset = _arraySerializer.int16(obj.Joy_Button_LB, buffer, bufferOffset, 4);
    // Check that the constant length array field [Joy_Button_RB] has the right length
    if (obj.Joy_Button_RB.length !== 4) {
      throw new Error('Unable to serialize array field Joy_Button_RB - length must be 4')
    }
    // Serialize message field [Joy_Button_RB]
    bufferOffset = _arraySerializer.int16(obj.Joy_Button_RB, buffer, bufferOffset, 4);
    // Check that the constant length array field [Joy_Button_STICK_LEFT] has the right length
    if (obj.Joy_Button_STICK_LEFT.length !== 4) {
      throw new Error('Unable to serialize array field Joy_Button_STICK_LEFT - length must be 4')
    }
    // Serialize message field [Joy_Button_STICK_LEFT]
    bufferOffset = _arraySerializer.int16(obj.Joy_Button_STICK_LEFT, buffer, bufferOffset, 4);
    // Check that the constant length array field [Joy_Button_STICK_RIGHT] has the right length
    if (obj.Joy_Button_STICK_RIGHT.length !== 4) {
      throw new Error('Unable to serialize array field Joy_Button_STICK_RIGHT - length must be 4')
    }
    // Serialize message field [Joy_Button_STICK_RIGHT]
    bufferOffset = _arraySerializer.int16(obj.Joy_Button_STICK_RIGHT, buffer, bufferOffset, 4);
    // Check that the constant length array field [Reset_pwm] has the right length
    if (obj.Reset_pwm.length !== 4) {
      throw new Error('Unable to serialize array field Reset_pwm - length must be 4')
    }
    // Serialize message field [Reset_pwm]
    bufferOffset = _arraySerializer.int16(obj.Reset_pwm, buffer, bufferOffset, 4);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Ctrl_cmd
    let len;
    let data = new Ctrl_cmd(null);
    // Deserialize message field [Ctrl_vel_X]
    data.Ctrl_vel_X = _arrayDeserializer.uint8(buffer, bufferOffset, 4)
    // Deserialize message field [Ctrl_vel_Y]
    data.Ctrl_vel_Y = _arrayDeserializer.uint8(buffer, bufferOffset, 4)
    // Deserialize message field [Ctrl_vel_Z]
    data.Ctrl_vel_Z = _arrayDeserializer.uint8(buffer, bufferOffset, 4)
    // Deserialize message field [Ctrl_fixed_Z]
    data.Ctrl_fixed_Z = _arrayDeserializer.uint8(buffer, bufferOffset, 4)
    // Deserialize message field [Ctrl_vel_Rol]
    data.Ctrl_vel_Rol = _arrayDeserializer.uint8(buffer, bufferOffset, 4)
    // Deserialize message field [Ctrl_vel_Pit]
    data.Ctrl_vel_Pit = _arrayDeserializer.uint8(buffer, bufferOffset, 4)
    // Deserialize message field [Ctrl_vel_Yaw]
    data.Ctrl_vel_Yaw = _arrayDeserializer.uint8(buffer, bufferOffset, 4)
    // Deserialize message field [Ctrl_fixed_Yaw]
    data.Ctrl_fixed_Yaw = _arrayDeserializer.uint8(buffer, bufferOffset, 4)
    // Deserialize message field [Ctrl_pivot_1]
    data.Ctrl_pivot_1 = _arrayDeserializer.uint8(buffer, bufferOffset, 4)
    // Deserialize message field [Ctrl_pivot_2]
    data.Ctrl_pivot_2 = _arrayDeserializer.uint8(buffer, bufferOffset, 4)
    // Deserialize message field [Ctrl_pivot_3]
    data.Ctrl_pivot_3 = _arrayDeserializer.uint8(buffer, bufferOffset, 4)
    // Deserialize message field [Ctrl_pivot_4]
    data.Ctrl_pivot_4 = _arrayDeserializer.uint8(buffer, bufferOffset, 4)
    // Deserialize message field [Ctrl_emagnet_1]
    data.Ctrl_emagnet_1 = _arrayDeserializer.uint8(buffer, bufferOffset, 4)
    // Deserialize message field [Ctrl_emagnet_2]
    data.Ctrl_emagnet_2 = _arrayDeserializer.uint8(buffer, bufferOffset, 4)
    // Deserialize message field [Ctrl_emagnet_3]
    data.Ctrl_emagnet_3 = _arrayDeserializer.uint8(buffer, bufferOffset, 4)
    // Deserialize message field [Ctrl_emagnet_4]
    data.Ctrl_emagnet_4 = _arrayDeserializer.uint8(buffer, bufferOffset, 4)
    // Deserialize message field [Ctrl_arm_joint_1]
    data.Ctrl_arm_joint_1 = _arrayDeserializer.uint8(buffer, bufferOffset, 4)
    // Deserialize message field [Ctrl_arm_joint_2]
    data.Ctrl_arm_joint_2 = _arrayDeserializer.uint8(buffer, bufferOffset, 4)
    // Deserialize message field [Joy_Button_Y]
    data.Joy_Button_Y = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    // Deserialize message field [Joy_Button_X]
    data.Joy_Button_X = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    // Deserialize message field [Joy_Button_A]
    data.Joy_Button_A = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    // Deserialize message field [Joy_Button_B]
    data.Joy_Button_B = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    // Deserialize message field [Joy_Button_LB]
    data.Joy_Button_LB = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    // Deserialize message field [Joy_Button_RB]
    data.Joy_Button_RB = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    // Deserialize message field [Joy_Button_STICK_LEFT]
    data.Joy_Button_STICK_LEFT = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    // Deserialize message field [Joy_Button_STICK_RIGHT]
    data.Joy_Button_STICK_RIGHT = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    // Deserialize message field [Reset_pwm]
    data.Reset_pwm = _arrayDeserializer.int16(buffer, bufferOffset, 4)
    return data;
  }

  static getMessageSize(object) {
    return 144;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/Ctrl_cmd';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '7a746ffc64e1d5a26fb5d6cba04e63c2';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    uint8[4] Ctrl_vel_X
    uint8[4] Ctrl_vel_Y
    uint8[4] Ctrl_vel_Z
    uint8[4] Ctrl_fixed_Z
    uint8[4] Ctrl_vel_Rol
    uint8[4] Ctrl_vel_Pit
    uint8[4] Ctrl_vel_Yaw
    uint8[4] Ctrl_fixed_Yaw
    
    uint8[4] Ctrl_pivot_1
    uint8[4] Ctrl_pivot_2
    uint8[4] Ctrl_pivot_3
    uint8[4] Ctrl_pivot_4
    
    uint8[4] Ctrl_emagnet_1
    uint8[4] Ctrl_emagnet_2
    uint8[4] Ctrl_emagnet_3
    uint8[4] Ctrl_emagnet_4
    
    uint8[4] Ctrl_arm_joint_1
    uint8[4] Ctrl_arm_joint_2
    
    int16[4] Joy_Button_Y
    int16[4] Joy_Button_X
    int16[4] Joy_Button_A
    int16[4] Joy_Button_B
    int16[4] Joy_Button_LB
    int16[4] Joy_Button_RB
    int16[4] Joy_Button_STICK_LEFT
    int16[4] Joy_Button_STICK_RIGHT
    int16[4] Reset_pwm
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Ctrl_cmd(null);
    if (msg.Ctrl_vel_X !== undefined) {
      resolved.Ctrl_vel_X = msg.Ctrl_vel_X;
    }
    else {
      resolved.Ctrl_vel_X = new Array(4).fill(0)
    }

    if (msg.Ctrl_vel_Y !== undefined) {
      resolved.Ctrl_vel_Y = msg.Ctrl_vel_Y;
    }
    else {
      resolved.Ctrl_vel_Y = new Array(4).fill(0)
    }

    if (msg.Ctrl_vel_Z !== undefined) {
      resolved.Ctrl_vel_Z = msg.Ctrl_vel_Z;
    }
    else {
      resolved.Ctrl_vel_Z = new Array(4).fill(0)
    }

    if (msg.Ctrl_fixed_Z !== undefined) {
      resolved.Ctrl_fixed_Z = msg.Ctrl_fixed_Z;
    }
    else {
      resolved.Ctrl_fixed_Z = new Array(4).fill(0)
    }

    if (msg.Ctrl_vel_Rol !== undefined) {
      resolved.Ctrl_vel_Rol = msg.Ctrl_vel_Rol;
    }
    else {
      resolved.Ctrl_vel_Rol = new Array(4).fill(0)
    }

    if (msg.Ctrl_vel_Pit !== undefined) {
      resolved.Ctrl_vel_Pit = msg.Ctrl_vel_Pit;
    }
    else {
      resolved.Ctrl_vel_Pit = new Array(4).fill(0)
    }

    if (msg.Ctrl_vel_Yaw !== undefined) {
      resolved.Ctrl_vel_Yaw = msg.Ctrl_vel_Yaw;
    }
    else {
      resolved.Ctrl_vel_Yaw = new Array(4).fill(0)
    }

    if (msg.Ctrl_fixed_Yaw !== undefined) {
      resolved.Ctrl_fixed_Yaw = msg.Ctrl_fixed_Yaw;
    }
    else {
      resolved.Ctrl_fixed_Yaw = new Array(4).fill(0)
    }

    if (msg.Ctrl_pivot_1 !== undefined) {
      resolved.Ctrl_pivot_1 = msg.Ctrl_pivot_1;
    }
    else {
      resolved.Ctrl_pivot_1 = new Array(4).fill(0)
    }

    if (msg.Ctrl_pivot_2 !== undefined) {
      resolved.Ctrl_pivot_2 = msg.Ctrl_pivot_2;
    }
    else {
      resolved.Ctrl_pivot_2 = new Array(4).fill(0)
    }

    if (msg.Ctrl_pivot_3 !== undefined) {
      resolved.Ctrl_pivot_3 = msg.Ctrl_pivot_3;
    }
    else {
      resolved.Ctrl_pivot_3 = new Array(4).fill(0)
    }

    if (msg.Ctrl_pivot_4 !== undefined) {
      resolved.Ctrl_pivot_4 = msg.Ctrl_pivot_4;
    }
    else {
      resolved.Ctrl_pivot_4 = new Array(4).fill(0)
    }

    if (msg.Ctrl_emagnet_1 !== undefined) {
      resolved.Ctrl_emagnet_1 = msg.Ctrl_emagnet_1;
    }
    else {
      resolved.Ctrl_emagnet_1 = new Array(4).fill(0)
    }

    if (msg.Ctrl_emagnet_2 !== undefined) {
      resolved.Ctrl_emagnet_2 = msg.Ctrl_emagnet_2;
    }
    else {
      resolved.Ctrl_emagnet_2 = new Array(4).fill(0)
    }

    if (msg.Ctrl_emagnet_3 !== undefined) {
      resolved.Ctrl_emagnet_3 = msg.Ctrl_emagnet_3;
    }
    else {
      resolved.Ctrl_emagnet_3 = new Array(4).fill(0)
    }

    if (msg.Ctrl_emagnet_4 !== undefined) {
      resolved.Ctrl_emagnet_4 = msg.Ctrl_emagnet_4;
    }
    else {
      resolved.Ctrl_emagnet_4 = new Array(4).fill(0)
    }

    if (msg.Ctrl_arm_joint_1 !== undefined) {
      resolved.Ctrl_arm_joint_1 = msg.Ctrl_arm_joint_1;
    }
    else {
      resolved.Ctrl_arm_joint_1 = new Array(4).fill(0)
    }

    if (msg.Ctrl_arm_joint_2 !== undefined) {
      resolved.Ctrl_arm_joint_2 = msg.Ctrl_arm_joint_2;
    }
    else {
      resolved.Ctrl_arm_joint_2 = new Array(4).fill(0)
    }

    if (msg.Joy_Button_Y !== undefined) {
      resolved.Joy_Button_Y = msg.Joy_Button_Y;
    }
    else {
      resolved.Joy_Button_Y = new Array(4).fill(0)
    }

    if (msg.Joy_Button_X !== undefined) {
      resolved.Joy_Button_X = msg.Joy_Button_X;
    }
    else {
      resolved.Joy_Button_X = new Array(4).fill(0)
    }

    if (msg.Joy_Button_A !== undefined) {
      resolved.Joy_Button_A = msg.Joy_Button_A;
    }
    else {
      resolved.Joy_Button_A = new Array(4).fill(0)
    }

    if (msg.Joy_Button_B !== undefined) {
      resolved.Joy_Button_B = msg.Joy_Button_B;
    }
    else {
      resolved.Joy_Button_B = new Array(4).fill(0)
    }

    if (msg.Joy_Button_LB !== undefined) {
      resolved.Joy_Button_LB = msg.Joy_Button_LB;
    }
    else {
      resolved.Joy_Button_LB = new Array(4).fill(0)
    }

    if (msg.Joy_Button_RB !== undefined) {
      resolved.Joy_Button_RB = msg.Joy_Button_RB;
    }
    else {
      resolved.Joy_Button_RB = new Array(4).fill(0)
    }

    if (msg.Joy_Button_STICK_LEFT !== undefined) {
      resolved.Joy_Button_STICK_LEFT = msg.Joy_Button_STICK_LEFT;
    }
    else {
      resolved.Joy_Button_STICK_LEFT = new Array(4).fill(0)
    }

    if (msg.Joy_Button_STICK_RIGHT !== undefined) {
      resolved.Joy_Button_STICK_RIGHT = msg.Joy_Button_STICK_RIGHT;
    }
    else {
      resolved.Joy_Button_STICK_RIGHT = new Array(4).fill(0)
    }

    if (msg.Reset_pwm !== undefined) {
      resolved.Reset_pwm = msg.Reset_pwm;
    }
    else {
      resolved.Reset_pwm = new Array(4).fill(0)
    }

    return resolved;
    }
};

module.exports = Ctrl_cmd;
