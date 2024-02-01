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
let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------

class NavigationState {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.position = null;
      this.linear_velocity = null;
      this.linear_acceleration = null;
      this.orientation = null;
      this.euler_velocity = null;
      this.euler_acceleration = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('position')) {
        this.position = initObj.position
      }
      else {
        this.position = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('linear_velocity')) {
        this.linear_velocity = initObj.linear_velocity
      }
      else {
        this.linear_velocity = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('linear_acceleration')) {
        this.linear_acceleration = initObj.linear_acceleration
      }
      else {
        this.linear_acceleration = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('orientation')) {
        this.orientation = initObj.orientation
      }
      else {
        this.orientation = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('euler_velocity')) {
        this.euler_velocity = initObj.euler_velocity
      }
      else {
        this.euler_velocity = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('euler_acceleration')) {
        this.euler_acceleration = initObj.euler_acceleration
      }
      else {
        this.euler_acceleration = new geometry_msgs.msg.Vector3();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type NavigationState
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [position]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.position, buffer, bufferOffset);
    // Serialize message field [linear_velocity]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.linear_velocity, buffer, bufferOffset);
    // Serialize message field [linear_acceleration]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.linear_acceleration, buffer, bufferOffset);
    // Serialize message field [orientation]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.orientation, buffer, bufferOffset);
    // Serialize message field [euler_velocity]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.euler_velocity, buffer, bufferOffset);
    // Serialize message field [euler_acceleration]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.euler_acceleration, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type NavigationState
    let len;
    let data = new NavigationState(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [position]
    data.position = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [linear_velocity]
    data.linear_velocity = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [linear_acceleration]
    data.linear_acceleration = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [orientation]
    data.orientation = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [euler_velocity]
    data.euler_velocity = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [euler_acceleration]
    data.euler_acceleration = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    return length + 144;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/NavigationState';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '3507e091ce71f01fa9dea0061304746a';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    
    geometry_msgs/Vector3 position
    geometry_msgs/Vector3 linear_velocity      # body
    geometry_msgs/Vector3 linear_acceleration  # body
    geometry_msgs/Vector3 orientation          # (r, p, y)
    geometry_msgs/Vector3 euler_velocity       # (dr/dt, dp/dt, dy/dt)
    geometry_msgs/Vector3 euler_acceleration   # (d2r/dt2, d2p/dt2, d2y/dt2)
    
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
    const resolved = new NavigationState(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.position !== undefined) {
      resolved.position = geometry_msgs.msg.Vector3.Resolve(msg.position)
    }
    else {
      resolved.position = new geometry_msgs.msg.Vector3()
    }

    if (msg.linear_velocity !== undefined) {
      resolved.linear_velocity = geometry_msgs.msg.Vector3.Resolve(msg.linear_velocity)
    }
    else {
      resolved.linear_velocity = new geometry_msgs.msg.Vector3()
    }

    if (msg.linear_acceleration !== undefined) {
      resolved.linear_acceleration = geometry_msgs.msg.Vector3.Resolve(msg.linear_acceleration)
    }
    else {
      resolved.linear_acceleration = new geometry_msgs.msg.Vector3()
    }

    if (msg.orientation !== undefined) {
      resolved.orientation = geometry_msgs.msg.Vector3.Resolve(msg.orientation)
    }
    else {
      resolved.orientation = new geometry_msgs.msg.Vector3()
    }

    if (msg.euler_velocity !== undefined) {
      resolved.euler_velocity = geometry_msgs.msg.Vector3.Resolve(msg.euler_velocity)
    }
    else {
      resolved.euler_velocity = new geometry_msgs.msg.Vector3()
    }

    if (msg.euler_acceleration !== undefined) {
      resolved.euler_acceleration = geometry_msgs.msg.Vector3.Resolve(msg.euler_acceleration)
    }
    else {
      resolved.euler_acceleration = new geometry_msgs.msg.Vector3()
    }

    return resolved;
    }
};

module.exports = NavigationState;
