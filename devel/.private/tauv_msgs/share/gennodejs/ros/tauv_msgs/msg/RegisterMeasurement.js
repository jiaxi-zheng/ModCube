// Auto-generated. Do not edit!

// (in-package tauv_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let PoseGraphMeasurement = require('./PoseGraphMeasurement.js');

//-----------------------------------------------------------

class RegisterMeasurement {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.pg_meas = null;
    }
    else {
      if (initObj.hasOwnProperty('pg_meas')) {
        this.pg_meas = initObj.pg_meas
      }
      else {
        this.pg_meas = new PoseGraphMeasurement();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type RegisterMeasurement
    // Serialize message field [pg_meas]
    bufferOffset = PoseGraphMeasurement.serialize(obj.pg_meas, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type RegisterMeasurement
    let len;
    let data = new RegisterMeasurement(null);
    // Deserialize message field [pg_meas]
    data.pg_meas = PoseGraphMeasurement.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += PoseGraphMeasurement.getMessageSize(object.pg_meas);
    return length;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/RegisterMeasurement';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'b355dd17bfdad2a0499de8384660e7ff';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    PoseGraphMeasurement pg_meas
    ================================================================================
    MSG: tauv_msgs/PoseGraphMeasurement
    Header header
    uint32 landmark_id
    geometry_msgs/Point position
    
    
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
    const resolved = new RegisterMeasurement(null);
    if (msg.pg_meas !== undefined) {
      resolved.pg_meas = PoseGraphMeasurement.Resolve(msg.pg_meas)
    }
    else {
      resolved.pg_meas = new PoseGraphMeasurement()
    }

    return resolved;
    }
};

module.exports = RegisterMeasurement;
