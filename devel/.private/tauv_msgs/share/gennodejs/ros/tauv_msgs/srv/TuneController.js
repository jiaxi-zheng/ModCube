// Auto-generated. Do not edit!

// (in-package tauv_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let PIDTuning = require('../msg/PIDTuning.js');

//-----------------------------------------------------------


//-----------------------------------------------------------

class TuneControllerRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.tunings = null;
    }
    else {
      if (initObj.hasOwnProperty('tunings')) {
        this.tunings = initObj.tunings
      }
      else {
        this.tunings = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type TuneControllerRequest
    // Serialize message field [tunings]
    // Serialize the length for message field [tunings]
    bufferOffset = _serializer.uint32(obj.tunings.length, buffer, bufferOffset);
    obj.tunings.forEach((val) => {
      bufferOffset = PIDTuning.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TuneControllerRequest
    let len;
    let data = new TuneControllerRequest(null);
    // Deserialize message field [tunings]
    // Deserialize array length for message field [tunings]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.tunings = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.tunings[i] = PIDTuning.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.tunings.forEach((val) => {
      length += PIDTuning.getMessageSize(val);
    });
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/TuneControllerRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '6eb5d206e69b070e4d9bf07120aa9b75';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    tauv_msgs/PIDTuning[] tunings
    
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
    const resolved = new TuneControllerRequest(null);
    if (msg.tunings !== undefined) {
      resolved.tunings = new Array(msg.tunings.length);
      for (let i = 0; i < resolved.tunings.length; ++i) {
        resolved.tunings[i] = PIDTuning.Resolve(msg.tunings[i]);
      }
    }
    else {
      resolved.tunings = []
    }

    return resolved;
    }
};

class TuneControllerResponse {
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
    // Serializes a message object of type TuneControllerResponse
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TuneControllerResponse
    let len;
    let data = new TuneControllerResponse(null);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/TuneControllerResponse';
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
    const resolved = new TuneControllerResponse(null);
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
  Request: TuneControllerRequest,
  Response: TuneControllerResponse,
  md5sum() { return 'c6a95158ee66091a3c801f9968586b2d'; },
  datatype() { return 'tauv_msgs/TuneController'; }
};
