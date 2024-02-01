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

class MapFindOneRequest {
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
    // Serializes a message object of type MapFindOneRequest
    // Serialize message field [tag]
    bufferOffset = _serializer.string(obj.tag, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type MapFindOneRequest
    let len;
    let data = new MapFindOneRequest(null);
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
    return 'tauv_msgs/MapFindOneRequest';
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
    const resolved = new MapFindOneRequest(null);
    if (msg.tag !== undefined) {
      resolved.tag = msg.tag;
    }
    else {
      resolved.tag = ''
    }

    return resolved;
    }
};

class MapFindOneResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.detection = null;
      this.success = null;
    }
    else {
      if (initObj.hasOwnProperty('detection')) {
        this.detection = initObj.detection
      }
      else {
        this.detection = new FeatureDetection();
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
    // Serializes a message object of type MapFindOneResponse
    // Serialize message field [detection]
    bufferOffset = FeatureDetection.serialize(obj.detection, buffer, bufferOffset);
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type MapFindOneResponse
    let len;
    let data = new MapFindOneResponse(null);
    // Deserialize message field [detection]
    data.detection = FeatureDetection.deserialize(buffer, bufferOffset);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += FeatureDetection.getMessageSize(object.detection);
    return length + 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/MapFindOneResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'a3723c79370a7a20ad0722931618c5db';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    tauv_msgs/FeatureDetection detection
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
    const resolved = new MapFindOneResponse(null);
    if (msg.detection !== undefined) {
      resolved.detection = FeatureDetection.Resolve(msg.detection)
    }
    else {
      resolved.detection = new FeatureDetection()
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
  Request: MapFindOneRequest,
  Response: MapFindOneResponse,
  md5sum() { return '0b9e59f5c45dc6444e155df33a798059'; },
  datatype() { return 'tauv_msgs/MapFindOne'; }
};
