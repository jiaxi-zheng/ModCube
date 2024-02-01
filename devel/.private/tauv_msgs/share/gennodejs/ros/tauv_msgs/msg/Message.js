// Auto-generated. Do not edit!

// (in-package tauv_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class Message {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.stamp = null;
      this.color_code_256 = null;
      this.severity = null;
      this.message = null;
    }
    else {
      if (initObj.hasOwnProperty('stamp')) {
        this.stamp = initObj.stamp
      }
      else {
        this.stamp = {secs: 0, nsecs: 0};
      }
      if (initObj.hasOwnProperty('color_code_256')) {
        this.color_code_256 = initObj.color_code_256
      }
      else {
        this.color_code_256 = 0;
      }
      if (initObj.hasOwnProperty('severity')) {
        this.severity = initObj.severity
      }
      else {
        this.severity = 0;
      }
      if (initObj.hasOwnProperty('message')) {
        this.message = initObj.message
      }
      else {
        this.message = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Message
    // Serialize message field [stamp]
    bufferOffset = _serializer.time(obj.stamp, buffer, bufferOffset);
    // Serialize message field [color_code_256]
    bufferOffset = _serializer.uint8(obj.color_code_256, buffer, bufferOffset);
    // Serialize message field [severity]
    bufferOffset = _serializer.int8(obj.severity, buffer, bufferOffset);
    // Serialize message field [message]
    bufferOffset = _serializer.string(obj.message, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Message
    let len;
    let data = new Message(null);
    // Deserialize message field [stamp]
    data.stamp = _deserializer.time(buffer, bufferOffset);
    // Deserialize message field [color_code_256]
    data.color_code_256 = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [severity]
    data.severity = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [message]
    data.message = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.message);
    return length + 14;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/Message';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '9553ea1e0a9cfe897226edaceb34218e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int8 ERROR=0
    int8 WARNING=1
    int8 INFO=2
    int8 DEBUG=3
    
    time stamp         # time stamp of the message
    uint8 color_code_256   # color code of the message
    int8 severity       # severity level
    string message      # message
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Message(null);
    if (msg.stamp !== undefined) {
      resolved.stamp = msg.stamp;
    }
    else {
      resolved.stamp = {secs: 0, nsecs: 0}
    }

    if (msg.color_code_256 !== undefined) {
      resolved.color_code_256 = msg.color_code_256;
    }
    else {
      resolved.color_code_256 = 0
    }

    if (msg.severity !== undefined) {
      resolved.severity = msg.severity;
    }
    else {
      resolved.severity = 0
    }

    if (msg.message !== undefined) {
      resolved.message = msg.message;
    }
    else {
      resolved.message = ''
    }

    return resolved;
    }
};

// Constants for message
Message.Constants = {
  ERROR: 0,
  WARNING: 1,
  INFO: 2,
  DEBUG: 3,
}

module.exports = Message;
