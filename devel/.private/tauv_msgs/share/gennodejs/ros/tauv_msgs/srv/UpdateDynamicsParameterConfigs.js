// Auto-generated. Do not edit!

// (in-package tauv_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let DynamicsParameterConfigUpdate = require('../msg/DynamicsParameterConfigUpdate.js');

//-----------------------------------------------------------


//-----------------------------------------------------------

class UpdateDynamicsParameterConfigsRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.updates = null;
    }
    else {
      if (initObj.hasOwnProperty('updates')) {
        this.updates = initObj.updates
      }
      else {
        this.updates = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type UpdateDynamicsParameterConfigsRequest
    // Serialize message field [updates]
    // Serialize the length for message field [updates]
    bufferOffset = _serializer.uint32(obj.updates.length, buffer, bufferOffset);
    obj.updates.forEach((val) => {
      bufferOffset = DynamicsParameterConfigUpdate.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type UpdateDynamicsParameterConfigsRequest
    let len;
    let data = new UpdateDynamicsParameterConfigsRequest(null);
    // Deserialize message field [updates]
    // Deserialize array length for message field [updates]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.updates = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.updates[i] = DynamicsParameterConfigUpdate.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.updates.forEach((val) => {
      length += DynamicsParameterConfigUpdate.getMessageSize(val);
    });
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/UpdateDynamicsParameterConfigsRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '6b0568555c382068683b9ab2dfb31b6b';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    tauv_msgs/DynamicsParameterConfigUpdate[] updates
    
    ================================================================================
    MSG: tauv_msgs/DynamicsParameterConfigUpdate
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
    const resolved = new UpdateDynamicsParameterConfigsRequest(null);
    if (msg.updates !== undefined) {
      resolved.updates = new Array(msg.updates.length);
      for (let i = 0; i < resolved.updates.length; ++i) {
        resolved.updates[i] = DynamicsParameterConfigUpdate.Resolve(msg.updates[i]);
      }
    }
    else {
      resolved.updates = []
    }

    return resolved;
    }
};

class UpdateDynamicsParameterConfigsResponse {
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
    // Serializes a message object of type UpdateDynamicsParameterConfigsResponse
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type UpdateDynamicsParameterConfigsResponse
    let len;
    let data = new UpdateDynamicsParameterConfigsResponse(null);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/UpdateDynamicsParameterConfigsResponse';
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
    const resolved = new UpdateDynamicsParameterConfigsResponse(null);
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
  Request: UpdateDynamicsParameterConfigsRequest,
  Response: UpdateDynamicsParameterConfigsResponse,
  md5sum() { return '613592a768bbac2700fe9ff9c5b0cffe'; },
  datatype() { return 'tauv_msgs/UpdateDynamicsParameterConfigs'; }
};
