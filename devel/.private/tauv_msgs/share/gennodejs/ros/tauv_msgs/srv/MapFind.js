// Auto-generated. Do not edit!

// (in-package tauv_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

let FeatureDetection = require('../msg/FeatureDetection.js');

//-----------------------------------------------------------

class MapFindRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.tag = null;
    }
    else {
      if (initObj.hasOwnProperty('tag')) {
        this.tag = initObj.tag
      }
      else {
        this.tag = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type MapFindRequest
    // Serialize message field [tag]
    bufferOffset = _serializer.string(obj.tag, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type MapFindRequest
    let len;
    let data = new MapFindRequest(null);
    // Deserialize message field [tag]
    data.tag = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.tag);
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/MapFindRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '5f71b33bec3d95b07048be7f138c8710';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string tag
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new MapFindRequest(null);
    if (msg.tag !== undefined) {
      resolved.tag = msg.tag;
    }
    else {
      resolved.tag = ''
    }

    return resolved;
    }
};

class MapFindResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.detections = null;
      this.success = null;
    }
    else {
      if (initObj.hasOwnProperty('detections')) {
        this.detections = initObj.detections
      }
      else {
        this.detections = [];
      }
      if (initObj.hasOwnProperty('success')) {
        this.success = initObj.success
      }
      else {
        this.success = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type MapFindResponse
    // Serialize message field [detections]
    // Serialize the length for message field [detections]
    bufferOffset = _serializer.uint32(obj.detections.length, buffer, bufferOffset);
    obj.detections.forEach((val) => {
      bufferOffset = FeatureDetection.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type MapFindResponse
    let len;
    let data = new MapFindResponse(null);
    // Deserialize message field [detections]
    // Deserialize array length for message field [detections]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.detections = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.detections[i] = FeatureDetection.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.detections.forEach((val) => {
      length += FeatureDetection.getMessageSize(val);
    });
    return length + 5;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/MapFindResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '3550f71bd1354de3af109636d90de6c0';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    tauv_msgs/FeatureDetection[] detections
    bool success
    
    ================================================================================
    MSG: tauv_msgs/FeatureDetection
    Header header
    geometry_msgs/Point position #SE2 msgs will only use x y
    geometry_msgs/Point orientation #SE2 msgs will only use x (theta)
    string tag
    float64 confidence
    bool SE2
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    string frame_id
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new MapFindResponse(null);
    if (msg.detections !== undefined) {
      resolved.detections = new Array(msg.detections.length);
      for (let i = 0; i < resolved.detections.length; ++i) {
        resolved.detections[i] = FeatureDetection.Resolve(msg.detections[i]);
      }
    }
    else {
      resolved.detections = []
    }

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
  Request: MapFindRequest,
  Response: MapFindResponse,
  md5sum() { return 'c4b23e80f3f361fabb0b381682792779'; },
  datatype() { return 'tauv_msgs/MapFind'; }
};
