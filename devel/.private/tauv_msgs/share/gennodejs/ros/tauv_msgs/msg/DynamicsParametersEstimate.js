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

class DynamicsParametersEstimate {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.stamp = null;
      this.m = null;
      this.cov_m = null;
      this.v = null;
      this.cov_v = null;
      this.g = null;
      this.cov_g = null;
      this.b = null;
      this.cov_b = null;
      this.I = null;
      this.cov_I = null;
      this.dl = null;
      this.cov_dl = null;
      this.dq = null;
      this.cov_dq = null;
      this.am = null;
      this.cov_am = null;
    }
    else {
      if (initObj.hasOwnProperty('stamp')) {
        this.stamp = initObj.stamp
      }
      else {
        this.stamp = {secs: 0, nsecs: 0};
      }
      if (initObj.hasOwnProperty('m')) {
        this.m = initObj.m
      }
      else {
        this.m = 0.0;
      }
      if (initObj.hasOwnProperty('cov_m')) {
        this.cov_m = initObj.cov_m
      }
      else {
        this.cov_m = 0.0;
      }
      if (initObj.hasOwnProperty('v')) {
        this.v = initObj.v
      }
      else {
        this.v = 0.0;
      }
      if (initObj.hasOwnProperty('cov_v')) {
        this.cov_v = initObj.cov_v
      }
      else {
        this.cov_v = 0.0;
      }
      if (initObj.hasOwnProperty('g')) {
        this.g = initObj.g
      }
      else {
        this.g = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('cov_g')) {
        this.cov_g = initObj.cov_g
      }
      else {
        this.cov_g = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('b')) {
        this.b = initObj.b
      }
      else {
        this.b = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('cov_b')) {
        this.cov_b = initObj.cov_b
      }
      else {
        this.cov_b = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('I')) {
        this.I = initObj.I
      }
      else {
        this.I = new Array(6).fill(0);
      }
      if (initObj.hasOwnProperty('cov_I')) {
        this.cov_I = initObj.cov_I
      }
      else {
        this.cov_I = new Array(6).fill(0);
      }
      if (initObj.hasOwnProperty('dl')) {
        this.dl = initObj.dl
      }
      else {
        this.dl = new Array(6).fill(0);
      }
      if (initObj.hasOwnProperty('cov_dl')) {
        this.cov_dl = initObj.cov_dl
      }
      else {
        this.cov_dl = new Array(6).fill(0);
      }
      if (initObj.hasOwnProperty('dq')) {
        this.dq = initObj.dq
      }
      else {
        this.dq = new Array(6).fill(0);
      }
      if (initObj.hasOwnProperty('cov_dq')) {
        this.cov_dq = initObj.cov_dq
      }
      else {
        this.cov_dq = new Array(6).fill(0);
      }
      if (initObj.hasOwnProperty('am')) {
        this.am = initObj.am
      }
      else {
        this.am = new Array(6).fill(0);
      }
      if (initObj.hasOwnProperty('cov_am')) {
        this.cov_am = initObj.cov_am
      }
      else {
        this.cov_am = new Array(6).fill(0);
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type DynamicsParametersEstimate
    // Serialize message field [stamp]
    bufferOffset = _serializer.time(obj.stamp, buffer, bufferOffset);
    // Serialize message field [m]
    bufferOffset = _serializer.float64(obj.m, buffer, bufferOffset);
    // Serialize message field [cov_m]
    bufferOffset = _serializer.float64(obj.cov_m, buffer, bufferOffset);
    // Serialize message field [v]
    bufferOffset = _serializer.float64(obj.v, buffer, bufferOffset);
    // Serialize message field [cov_v]
    bufferOffset = _serializer.float64(obj.cov_v, buffer, bufferOffset);
    // Check that the constant length array field [g] has the right length
    if (obj.g.length !== 3) {
      throw new Error('Unable to serialize array field g - length must be 3')
    }
    // Serialize message field [g]
    bufferOffset = _arraySerializer.float64(obj.g, buffer, bufferOffset, 3);
    // Check that the constant length array field [cov_g] has the right length
    if (obj.cov_g.length !== 3) {
      throw new Error('Unable to serialize array field cov_g - length must be 3')
    }
    // Serialize message field [cov_g]
    bufferOffset = _arraySerializer.float64(obj.cov_g, buffer, bufferOffset, 3);
    // Check that the constant length array field [b] has the right length
    if (obj.b.length !== 3) {
      throw new Error('Unable to serialize array field b - length must be 3')
    }
    // Serialize message field [b]
    bufferOffset = _arraySerializer.float64(obj.b, buffer, bufferOffset, 3);
    // Check that the constant length array field [cov_b] has the right length
    if (obj.cov_b.length !== 3) {
      throw new Error('Unable to serialize array field cov_b - length must be 3')
    }
    // Serialize message field [cov_b]
    bufferOffset = _arraySerializer.float64(obj.cov_b, buffer, bufferOffset, 3);
    // Check that the constant length array field [I] has the right length
    if (obj.I.length !== 6) {
      throw new Error('Unable to serialize array field I - length must be 6')
    }
    // Serialize message field [I]
    bufferOffset = _arraySerializer.float64(obj.I, buffer, bufferOffset, 6);
    // Check that the constant length array field [cov_I] has the right length
    if (obj.cov_I.length !== 6) {
      throw new Error('Unable to serialize array field cov_I - length must be 6')
    }
    // Serialize message field [cov_I]
    bufferOffset = _arraySerializer.float64(obj.cov_I, buffer, bufferOffset, 6);
    // Check that the constant length array field [dl] has the right length
    if (obj.dl.length !== 6) {
      throw new Error('Unable to serialize array field dl - length must be 6')
    }
    // Serialize message field [dl]
    bufferOffset = _arraySerializer.float64(obj.dl, buffer, bufferOffset, 6);
    // Check that the constant length array field [cov_dl] has the right length
    if (obj.cov_dl.length !== 6) {
      throw new Error('Unable to serialize array field cov_dl - length must be 6')
    }
    // Serialize message field [cov_dl]
    bufferOffset = _arraySerializer.float64(obj.cov_dl, buffer, bufferOffset, 6);
    // Check that the constant length array field [dq] has the right length
    if (obj.dq.length !== 6) {
      throw new Error('Unable to serialize array field dq - length must be 6')
    }
    // Serialize message field [dq]
    bufferOffset = _arraySerializer.float64(obj.dq, buffer, bufferOffset, 6);
    // Check that the constant length array field [cov_dq] has the right length
    if (obj.cov_dq.length !== 6) {
      throw new Error('Unable to serialize array field cov_dq - length must be 6')
    }
    // Serialize message field [cov_dq]
    bufferOffset = _arraySerializer.float64(obj.cov_dq, buffer, bufferOffset, 6);
    // Check that the constant length array field [am] has the right length
    if (obj.am.length !== 6) {
      throw new Error('Unable to serialize array field am - length must be 6')
    }
    // Serialize message field [am]
    bufferOffset = _arraySerializer.float64(obj.am, buffer, bufferOffset, 6);
    // Check that the constant length array field [cov_am] has the right length
    if (obj.cov_am.length !== 6) {
      throw new Error('Unable to serialize array field cov_am - length must be 6')
    }
    // Serialize message field [cov_am]
    bufferOffset = _arraySerializer.float64(obj.cov_am, buffer, bufferOffset, 6);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type DynamicsParametersEstimate
    let len;
    let data = new DynamicsParametersEstimate(null);
    // Deserialize message field [stamp]
    data.stamp = _deserializer.time(buffer, bufferOffset);
    // Deserialize message field [m]
    data.m = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [cov_m]
    data.cov_m = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [v]
    data.v = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [cov_v]
    data.cov_v = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [g]
    data.g = _arrayDeserializer.float64(buffer, bufferOffset, 3)
    // Deserialize message field [cov_g]
    data.cov_g = _arrayDeserializer.float64(buffer, bufferOffset, 3)
    // Deserialize message field [b]
    data.b = _arrayDeserializer.float64(buffer, bufferOffset, 3)
    // Deserialize message field [cov_b]
    data.cov_b = _arrayDeserializer.float64(buffer, bufferOffset, 3)
    // Deserialize message field [I]
    data.I = _arrayDeserializer.float64(buffer, bufferOffset, 6)
    // Deserialize message field [cov_I]
    data.cov_I = _arrayDeserializer.float64(buffer, bufferOffset, 6)
    // Deserialize message field [dl]
    data.dl = _arrayDeserializer.float64(buffer, bufferOffset, 6)
    // Deserialize message field [cov_dl]
    data.cov_dl = _arrayDeserializer.float64(buffer, bufferOffset, 6)
    // Deserialize message field [dq]
    data.dq = _arrayDeserializer.float64(buffer, bufferOffset, 6)
    // Deserialize message field [cov_dq]
    data.cov_dq = _arrayDeserializer.float64(buffer, bufferOffset, 6)
    // Deserialize message field [am]
    data.am = _arrayDeserializer.float64(buffer, bufferOffset, 6)
    // Deserialize message field [cov_am]
    data.cov_am = _arrayDeserializer.float64(buffer, bufferOffset, 6)
    return data;
  }

  static getMessageSize(object) {
    return 520;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/DynamicsParametersEstimate';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'bd252a6b82405355d619dce17f3eaa07';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    time stamp
    
    float64 m
    float64 cov_m
    float64 v
    float64 cov_v
    float64[3] g
    float64[3] cov_g
    float64[3] b
    float64[3] cov_b
    float64[6] I
    float64[6] cov_I
    float64[6] dl
    float64[6] cov_dl
    float64[6] dq
    float64[6] cov_dq
    float64[6] am
    float64[6] cov_am
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new DynamicsParametersEstimate(null);
    if (msg.stamp !== undefined) {
      resolved.stamp = msg.stamp;
    }
    else {
      resolved.stamp = {secs: 0, nsecs: 0}
    }

    if (msg.m !== undefined) {
      resolved.m = msg.m;
    }
    else {
      resolved.m = 0.0
    }

    if (msg.cov_m !== undefined) {
      resolved.cov_m = msg.cov_m;
    }
    else {
      resolved.cov_m = 0.0
    }

    if (msg.v !== undefined) {
      resolved.v = msg.v;
    }
    else {
      resolved.v = 0.0
    }

    if (msg.cov_v !== undefined) {
      resolved.cov_v = msg.cov_v;
    }
    else {
      resolved.cov_v = 0.0
    }

    if (msg.g !== undefined) {
      resolved.g = msg.g;
    }
    else {
      resolved.g = new Array(3).fill(0)
    }

    if (msg.cov_g !== undefined) {
      resolved.cov_g = msg.cov_g;
    }
    else {
      resolved.cov_g = new Array(3).fill(0)
    }

    if (msg.b !== undefined) {
      resolved.b = msg.b;
    }
    else {
      resolved.b = new Array(3).fill(0)
    }

    if (msg.cov_b !== undefined) {
      resolved.cov_b = msg.cov_b;
    }
    else {
      resolved.cov_b = new Array(3).fill(0)
    }

    if (msg.I !== undefined) {
      resolved.I = msg.I;
    }
    else {
      resolved.I = new Array(6).fill(0)
    }

    if (msg.cov_I !== undefined) {
      resolved.cov_I = msg.cov_I;
    }
    else {
      resolved.cov_I = new Array(6).fill(0)
    }

    if (msg.dl !== undefined) {
      resolved.dl = msg.dl;
    }
    else {
      resolved.dl = new Array(6).fill(0)
    }

    if (msg.cov_dl !== undefined) {
      resolved.cov_dl = msg.cov_dl;
    }
    else {
      resolved.cov_dl = new Array(6).fill(0)
    }

    if (msg.dq !== undefined) {
      resolved.dq = msg.dq;
    }
    else {
      resolved.dq = new Array(6).fill(0)
    }

    if (msg.cov_dq !== undefined) {
      resolved.cov_dq = msg.cov_dq;
    }
    else {
      resolved.cov_dq = new Array(6).fill(0)
    }

    if (msg.am !== undefined) {
      resolved.am = msg.am;
    }
    else {
      resolved.am = new Array(6).fill(0)
    }

    if (msg.cov_am !== undefined) {
      resolved.cov_am = msg.cov_am;
    }
    else {
      resolved.cov_am = new Array(6).fill(0)
    }

    return resolved;
    }
};

module.exports = DynamicsParametersEstimate;
