// Auto-generated. Do not edit!

// (in-package tauv_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let FeatureDetections = require('../msg/FeatureDetections.js');

//-----------------------------------------------------------


//-----------------------------------------------------------

class FeatureDetectionsSyncRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.detections = null;
    }
    else {
      if (initObj.hasOwnProperty('detections')) {
        this.detections = initObj.detections
      }
      else {
        this.detections = new FeatureDetections();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type FeatureDetectionsSyncRequest
    // Serialize message field [detections]
    bufferOffset = FeatureDetections.serialize(obj.detections, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type FeatureDetectionsSyncRequest
    let len;
    let data = new FeatureDetectionsSyncRequest(null);
    // Deserialize message field [detections]
    data.detections = FeatureDetections.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += FeatureDetections.getMessageSize(object.detections);
    return length;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/FeatureDetectionsSyncRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'b4cef97e8164c531a1f9db3da9148fd5';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    tauv_msgs/FeatureDetections detections
    
    ================================================================================
    MSG: tauv_msgs/FeatureDetections
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
    const resolved = new FeatureDetectionsSyncRequest(null);
    if (msg.detections !== undefined) {
      resolved.detections = FeatureDetections.Resolve(msg.detections)
    }
    else {
      resolved.detections = new FeatureDetections()
    }

    return resolved;
    }
};

class FeatureDetectionsSyncResponse {
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
    // Serializes a message object of type FeatureDetectionsSyncResponse
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type FeatureDetectionsSyncResponse
    let len;
    let data = new FeatureDetectionsSyncResponse(null);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/FeatureDetectionsSyncResponse';
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
    const resolved = new FeatureDetectionsSyncResponse(null);
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
  Request: FeatureDetectionsSyncRequest,
  Response: FeatureDetectionsSyncResponse,
  md5sum() { return '2da451629f8ca3f1c8b6e832ba66c6c5'; },
  datatype() { return 'tauv_msgs/FeatureDetectionsSync'; }
};
