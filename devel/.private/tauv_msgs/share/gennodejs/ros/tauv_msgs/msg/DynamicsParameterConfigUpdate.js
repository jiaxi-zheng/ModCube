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

class DynamicsParameterConfigUpdate {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.name = null;
      this.update_initial_value = null;
      this.initial_value = null;
      this.update_fixed = null;
      this.fixed = null;
      this.update_initial_covariance = null;
      this.initial_covariance = null;
      this.update_process_covariance = null;
      this.process_covariance = null;
      this.update_limits = null;
      this.limits = null;
      this.reset = null;
    }
    else {
      if (initObj.hasOwnProperty('name')) {
        this.name = initObj.name
      }
      else {
        this.name = '';
      }
      if (initObj.hasOwnProperty('update_initial_value')) {
        this.update_initial_value = initObj.update_initial_value
      }
      else {
        this.update_initial_value = false;
      }
      if (initObj.hasOwnProperty('initial_value')) {
        this.initial_value = initObj.initial_value
      }
      else {
        this.initial_value = 0.0;
      }
      if (initObj.hasOwnProperty('update_fixed')) {
        this.update_fixed = initObj.update_fixed
      }
      else {
        this.update_fixed = false;
      }
      if (initObj.hasOwnProperty('fixed')) {
        this.fixed = initObj.fixed
      }
      else {
        this.fixed = false;
      }
      if (initObj.hasOwnProperty('update_initial_covariance')) {
        this.update_initial_covariance = initObj.update_initial_covariance
      }
      else {
        this.update_initial_covariance = false;
      }
      if (initObj.hasOwnProperty('initial_covariance')) {
        this.initial_covariance = initObj.initial_covariance
      }
      else {
        this.initial_covariance = 0.0;
      }
      if (initObj.hasOwnProperty('update_process_covariance')) {
        this.update_process_covariance = initObj.update_process_covariance
      }
      else {
        this.update_process_covariance = false;
      }
      if (initObj.hasOwnProperty('process_covariance')) {
        this.process_covariance = initObj.process_covariance
      }
      else {
        this.process_covariance = 0.0;
      }
      if (initObj.hasOwnProperty('update_limits')) {
        this.update_limits = initObj.update_limits
      }
      else {
        this.update_limits = false;
      }
      if (initObj.hasOwnProperty('limits')) {
        this.limits = initObj.limits
      }
      else {
        this.limits = new Array(2).fill(0);
      }
      if (initObj.hasOwnProperty('reset')) {
        this.reset = initObj.reset
      }
      else {
        this.reset = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type DynamicsParameterConfigUpdate
    // Serialize message field [name]
    bufferOffset = _serializer.string(obj.name, buffer, bufferOffset);
    // Serialize message field [update_initial_value]
    bufferOffset = _serializer.bool(obj.update_initial_value, buffer, bufferOffset);
    // Serialize message field [initial_value]
    bufferOffset = _serializer.float64(obj.initial_value, buffer, bufferOffset);
    // Serialize message field [update_fixed]
    bufferOffset = _serializer.bool(obj.update_fixed, buffer, bufferOffset);
    // Serialize message field [fixed]
    bufferOffset = _serializer.bool(obj.fixed, buffer, bufferOffset);
    // Serialize message field [update_initial_covariance]
    bufferOffset = _serializer.bool(obj.update_initial_covariance, buffer, bufferOffset);
    // Serialize message field [initial_covariance]
    bufferOffset = _serializer.float64(obj.initial_covariance, buffer, bufferOffset);
    // Serialize message field [update_process_covariance]
    bufferOffset = _serializer.bool(obj.update_process_covariance, buffer, bufferOffset);
    // Serialize message field [process_covariance]
    bufferOffset = _serializer.float64(obj.process_covariance, buffer, bufferOffset);
    // Serialize message field [update_limits]
    bufferOffset = _serializer.bool(obj.update_limits, buffer, bufferOffset);
    // Check that the constant length array field [limits] has the right length
    if (obj.limits.length !== 2) {
      throw new Error('Unable to serialize array field limits - length must be 2')
    }
    // Serialize message field [limits]
    bufferOffset = _arraySerializer.float64(obj.limits, buffer, bufferOffset, 2);
    // Serialize message field [reset]
    bufferOffset = _serializer.bool(obj.reset, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type DynamicsParameterConfigUpdate
    let len;
    let data = new DynamicsParameterConfigUpdate(null);
    // Deserialize message field [name]
    data.name = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [update_initial_value]
    data.update_initial_value = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [initial_value]
    data.initial_value = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [update_fixed]
    data.update_fixed = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [fixed]
    data.fixed = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [update_initial_covariance]
    data.update_initial_covariance = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [initial_covariance]
    data.initial_covariance = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [update_process_covariance]
    data.update_process_covariance = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [process_covariance]
    data.process_covariance = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [update_limits]
    data.update_limits = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [limits]
    data.limits = _arrayDeserializer.float64(buffer, bufferOffset, 2)
    // Deserialize message field [reset]
    data.reset = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.name);
    return length + 51;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/DynamicsParameterConfigUpdate';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '31294fd8c67bf91ba516e5502711b385';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string name
    
    bool update_initial_value
    float64 initial_value
    
    bool update_fixed
    bool fixed
    
    bool update_initial_covariance
    float64 initial_covariance
    
    bool update_process_covariance
    float64 process_covariance
    
    bool update_limits
    float64[2] limits
    
    bool reset
    
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new DynamicsParameterConfigUpdate(null);
    if (msg.name !== undefined) {
      resolved.name = msg.name;
    }
    else {
      resolved.name = ''
    }

    if (msg.update_initial_value !== undefined) {
      resolved.update_initial_value = msg.update_initial_value;
    }
    else {
      resolved.update_initial_value = false
    }

    if (msg.initial_value !== undefined) {
      resolved.initial_value = msg.initial_value;
    }
    else {
      resolved.initial_value = 0.0
    }

    if (msg.update_fixed !== undefined) {
      resolved.update_fixed = msg.update_fixed;
    }
    else {
      resolved.update_fixed = false
    }

    if (msg.fixed !== undefined) {
      resolved.fixed = msg.fixed;
    }
    else {
      resolved.fixed = false
    }

    if (msg.update_initial_covariance !== undefined) {
      resolved.update_initial_covariance = msg.update_initial_covariance;
    }
    else {
      resolved.update_initial_covariance = false
    }

    if (msg.initial_covariance !== undefined) {
      resolved.initial_covariance = msg.initial_covariance;
    }
    else {
      resolved.initial_covariance = 0.0
    }

    if (msg.update_process_covariance !== undefined) {
      resolved.update_process_covariance = msg.update_process_covariance;
    }
    else {
      resolved.update_process_covariance = false
    }

    if (msg.process_covariance !== undefined) {
      resolved.process_covariance = msg.process_covariance;
    }
    else {
      resolved.process_covariance = 0.0
    }

    if (msg.update_limits !== undefined) {
      resolved.update_limits = msg.update_limits;
    }
    else {
      resolved.update_limits = false
    }

    if (msg.limits !== undefined) {
      resolved.limits = msg.limits;
    }
    else {
      resolved.limits = new Array(2).fill(0)
    }

    if (msg.reset !== undefined) {
      resolved.reset = msg.reset;
    }
    else {
      resolved.reset = false
    }

    return resolved;
    }
};

module.exports = DynamicsParameterConfigUpdate;
