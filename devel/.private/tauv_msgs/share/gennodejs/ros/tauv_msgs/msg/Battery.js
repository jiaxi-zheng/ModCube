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

class Battery {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.state_of_charge = null;
      this.voltage = null;
      this.average_current = null;
      this.remaining_capacity = null;
      this.full_capacity = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('state_of_charge')) {
        this.state_of_charge = initObj.state_of_charge
      }
      else {
        this.state_of_charge = 0.0;
      }
      if (initObj.hasOwnProperty('voltage')) {
        this.voltage = initObj.voltage
      }
      else {
        this.voltage = 0.0;
      }
      if (initObj.hasOwnProperty('average_current')) {
        this.average_current = initObj.average_current
      }
      else {
        this.average_current = 0.0;
      }
      if (initObj.hasOwnProperty('remaining_capacity')) {
        this.remaining_capacity = initObj.remaining_capacity
      }
      else {
        this.remaining_capacity = 0.0;
      }
      if (initObj.hasOwnProperty('full_capacity')) {
        this.full_capacity = initObj.full_capacity
      }
      else {
        this.full_capacity = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Battery
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [state_of_charge]
    bufferOffset = _serializer.float64(obj.state_of_charge, buffer, bufferOffset);
    // Serialize message field [voltage]
    bufferOffset = _serializer.float64(obj.voltage, buffer, bufferOffset);
    // Serialize message field [average_current]
    bufferOffset = _serializer.float64(obj.average_current, buffer, bufferOffset);
    // Serialize message field [remaining_capacity]
    bufferOffset = _serializer.float64(obj.remaining_capacity, buffer, bufferOffset);
    // Serialize message field [full_capacity]
    bufferOffset = _serializer.float64(obj.full_capacity, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Battery
    let len;
    let data = new Battery(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [state_of_charge]
    data.state_of_charge = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [voltage]
    data.voltage = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [average_current]
    data.average_current = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [remaining_capacity]
    data.remaining_capacity = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [full_capacity]
    data.full_capacity = _deserializer.float64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    return length + 40;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/Battery';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '6a49cc2bae2938fa186ed67011f717c3';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    
    float64 state_of_charge
    float64 voltage
    float64 average_current
    float64 remaining_capacity
    float64 full_capacity
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
    const resolved = new Battery(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.state_of_charge !== undefined) {
      resolved.state_of_charge = msg.state_of_charge;
    }
    else {
      resolved.state_of_charge = 0.0
    }

    if (msg.voltage !== undefined) {
      resolved.voltage = msg.voltage;
    }
    else {
      resolved.voltage = 0.0
    }

    if (msg.average_current !== undefined) {
      resolved.average_current = msg.average_current;
    }
    else {
      resolved.average_current = 0.0
    }

    if (msg.remaining_capacity !== undefined) {
      resolved.remaining_capacity = msg.remaining_capacity;
    }
    else {
      resolved.remaining_capacity = 0.0
    }

    if (msg.full_capacity !== undefined) {
      resolved.full_capacity = msg.full_capacity;
    }
    else {
      resolved.full_capacity = 0.0
    }

    return resolved;
    }
};

module.exports = Battery;
