// Auto-generated. Do not edit!

// (in-package tauv_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let geometry_msgs = _finder('geometry_msgs');
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class XsensImuData {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.ros_time = null;
      this.imu_time = null;
      this.triggered_dvl = null;
      this.orientation = null;
      this.rate_of_turn = null;
      this.linear_acceleration = null;
      this.free_acceleration = null;
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
      if (initObj.hasOwnProperty('orientation')) {
        this.orientation = initObj.orientation
      }
      else {
        this.orientation = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('rate_of_turn')) {
        this.rate_of_turn = initObj.rate_of_turn
      }
      else {
        this.rate_of_turn = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('linear_acceleration')) {
        this.linear_acceleration = initObj.linear_acceleration
      }
      else {
        this.linear_acceleration = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('free_acceleration')) {
        this.free_acceleration = initObj.free_acceleration
      }
      else {
        this.free_acceleration = new geometry_msgs.msg.Vector3();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type XsensImuData
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [ros_time]
    bufferOffset = _serializer.time(obj.ros_time, buffer, bufferOffset);
    // Serialize message field [imu_time]
    bufferOffset = _serializer.time(obj.imu_time, buffer, bufferOffset);
    // Serialize message field [triggered_dvl]
    bufferOffset = _serializer.bool(obj.triggered_dvl, buffer, bufferOffset);
    // Serialize message field [orientation]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.orientation, buffer, bufferOffset);
    // Serialize message field [rate_of_turn]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.rate_of_turn, buffer, bufferOffset);
    // Serialize message field [linear_acceleration]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.linear_acceleration, buffer, bufferOffset);
    // Serialize message field [free_acceleration]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.free_acceleration, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type XsensImuData
    let len;
    let data = new XsensImuData(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [ros_time]
    data.ros_time = _deserializer.time(buffer, bufferOffset);
    // Deserialize message field [imu_time]
    data.imu_time = _deserializer.time(buffer, bufferOffset);
    // Deserialize message field [triggered_dvl]
    data.triggered_dvl = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [orientation]
    data.orientation = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [rate_of_turn]
    data.rate_of_turn = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [linear_acceleration]
    data.linear_acceleration = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [free_acceleration]
    data.free_acceleration = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    return length + 113;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/XsensImuData';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'a5e7b14c591863b869ed2281b0f6b1ed';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    
    time ros_time
    
    time imu_time
    
    bool triggered_dvl
    
    geometry_msgs/Vector3 orientation
    
    geometry_msgs/Vector3 rate_of_turn
    
    geometry_msgs/Vector3 linear_acceleration
    
    geometry_msgs/Vector3 free_acceleration
    
    
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
    MSG: geometry_msgs/Vector3
    # This represents a vector in free space. 
    # It is only meant to represent a direction. Therefore, it does not
    # make sense to apply a translation to it (e.g., when applying a 
    # generic rigid transformation to a Vector3, tf2 will only apply the
    # rotation). If you want your data to be translatable too, use the
    # geometry_msgs/Point message instead.
    
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
    const resolved = new XsensImuData(null);
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

    if (msg.orientation !== undefined) {
      resolved.orientation = geometry_msgs.msg.Vector3.Resolve(msg.orientation)
    }
    else {
      resolved.orientation = new geometry_msgs.msg.Vector3()
    }

    if (msg.rate_of_turn !== undefined) {
      resolved.rate_of_turn = geometry_msgs.msg.Vector3.Resolve(msg.rate_of_turn)
    }
    else {
      resolved.rate_of_turn = new geometry_msgs.msg.Vector3()
    }

    if (msg.linear_acceleration !== undefined) {
      resolved.linear_acceleration = geometry_msgs.msg.Vector3.Resolve(msg.linear_acceleration)
    }
    else {
      resolved.linear_acceleration = new geometry_msgs.msg.Vector3()
    }

    if (msg.free_acceleration !== undefined) {
      resolved.free_acceleration = geometry_msgs.msg.Vector3.Resolve(msg.free_acceleration)
    }
    else {
      resolved.free_acceleration = new geometry_msgs.msg.Vector3()
    }

    return resolved;
    }
};

module.exports = XsensImuData;
