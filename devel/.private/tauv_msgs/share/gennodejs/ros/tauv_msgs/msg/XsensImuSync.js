// Auto-generated. Do not edit!

// (in-package tauv_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class XsensImuSync {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.ros_time = null;
      this.imu_time = null;
      this.triggered_dvl = null;
      this.d_corrected = null;
      this.d_ros = null;
      this.d_imu = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('ros_time')) {
        this.ros_time = initObj.ros_time
      }
      else {
        this.ros_time = {secs: 0, nsecs: 0};
      }
      if (initObj.hasOwnProperty('imu_time')) {
        this.imu_time = initObj.imu_time
      }
      else {
        this.imu_time = {secs: 0, nsecs: 0};
      }
      if (initObj.hasOwnProperty('triggered_dvl')) {
        this.triggered_dvl = initObj.triggered_dvl
      }
      else {
        this.triggered_dvl = false;
      }
      if (initObj.hasOwnProperty('d_corrected')) {
        this.d_corrected = initObj.d_corrected
      }
      else {
        this.d_corrected = 0.0;
      }
      if (initObj.hasOwnProperty('d_ros')) {
        this.d_ros = initObj.d_ros
      }
      else {
        this.d_ros = 0.0;
      }
      if (initObj.hasOwnProperty('d_imu')) {
        this.d_imu = initObj.d_imu
      }
      else {
        this.d_imu = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type XsensImuSync
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [ros_time]
    bufferOffset = _serializer.time(obj.ros_time, buffer, bufferOffset);
    // Serialize message field [imu_time]
    bufferOffset = _serializer.time(obj.imu_time, buffer, bufferOffset);
    // Serialize message field [triggered_dvl]
    bufferOffset = _serializer.bool(obj.triggered_dvl, buffer, bufferOffset);
    // Serialize message field [d_corrected]
    bufferOffset = _serializer.float64(obj.d_corrected, buffer, bufferOffset);
    // Serialize message field [d_ros]
    bufferOffset = _serializer.float64(obj.d_ros, buffer, bufferOffset);
    // Serialize message field [d_imu]
    bufferOffset = _serializer.float64(obj.d_imu, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type XsensImuSync
    let len;
    let data = new XsensImuSync(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [ros_time]
    data.ros_time = _deserializer.time(buffer, bufferOffset);
    // Deserialize message field [imu_time]
    data.imu_time = _deserializer.time(buffer, bufferOffset);
    // Deserialize message field [triggered_dvl]
    data.triggered_dvl = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [d_corrected]
    data.d_corrected = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [d_ros]
    data.d_ros = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [d_imu]
    data.d_imu = _deserializer.float64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    return length + 41;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/XsensImuSync';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '8d95951a55cfd457da142f2cc6b05ae6';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    
    time ros_time
    
    time imu_time
    
    bool triggered_dvl
    
    float64 d_corrected # optional
    
    float64 d_ros # optional
    
    float64 d_imu # optional
    
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
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new XsensImuSync(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.ros_time !== undefined) {
      resolved.ros_time = msg.ros_time;
    }
    else {
      resolved.ros_time = {secs: 0, nsecs: 0}
    }

    if (msg.imu_time !== undefined) {
      resolved.imu_time = msg.imu_time;
    }
    else {
      resolved.imu_time = {secs: 0, nsecs: 0}
    }

    if (msg.triggered_dvl !== undefined) {
      resolved.triggered_dvl = msg.triggered_dvl;
    }
    else {
      resolved.triggered_dvl = false
    }

    if (msg.d_corrected !== undefined) {
      resolved.d_corrected = msg.d_corrected;
    }
    else {
      resolved.d_corrected = 0.0
    }

    if (msg.d_ros !== undefined) {
      resolved.d_ros = msg.d_ros;
    }
    else {
      resolved.d_ros = 0.0
    }

    if (msg.d_imu !== undefined) {
      resolved.d_imu = msg.d_imu;
    }
    else {
      resolved.d_imu = 0.0
    }

    return resolved;
    }
};

module.exports = XsensImuSync;
