// Auto-generated. Do not edit!

// (in-package tauv_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let FeatureDetection = require('./FeatureDetection.js');

//-----------------------------------------------------------

class FeatureDetections {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.detections = null;
      this.detector_tag = null;
    }
    else {
      if (initObj.hasOwnProperty('detections')) {
        this.detections = initObj.detections
      }
      else {
        this.detections = [];
      }
      if (initObj.hasOwnProperty('detector_tag')) {
        this.detector_tag = initObj.detector_tag
      }
      else {
        this.detector_tag = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type FeatureDetections
    // Serialize message field [detections]
    // Serialize the length for message field [detections]
    bufferOffset = _serializer.uint32(obj.detections.length, buffer, bufferOffset);
    obj.detections.forEach((val) => {
      bufferOffset = FeatureDetection.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [detector_tag]
    bufferOffset = _serializer.string(obj.detector_tag, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type FeatureDetections
    let len;
    let data = new FeatureDetections(null);
    // Deserialize message field [detections]
    // Deserialize array length for message field [detections]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.detections = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.detections[i] = FeatureDetection.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [detector_tag]
    data.detector_tag = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.detections.forEach((val) => {
      length += FeatureDetection.getMessageSize(val);
    });
    length += _getByteLength(object.detector_tag);
    return length + 8;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/FeatureDetections';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'b198f96e11b160e3e0b3e1f890a3f57d';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    FeatureDetection[] detections
    string detector_tag
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
    const resolved = new FeatureDetections(null);
    if (msg.detections !== undefined) {
      resolved.detections = new Array(msg.detections.length);
      for (let i = 0; i < resolved.detections.length; ++i) {
        resolved.detections[i] = FeatureDetection.Resolve(msg.detections[i]);
      }
    }
    else {
      resolved.detections = []
    }

    if (msg.detector_tag !== undefined) {
      resolved.detector_tag = msg.detector_tag;
    }
    else {
      resolved.detector_tag = ''
    }

    return resolved;
    }
};

module.exports = FeatureDetections;
