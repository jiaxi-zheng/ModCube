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

class Servos {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.targets = null;
    }
    else {
      if (initObj.hasOwnProperty('targets')) {
        this.targets = initObj.targets
      }
      else {
        this.targets = new Array(4).fill(0);
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Servos
    // Check that the constant length array field [targets] has the right length
    if (obj.targets.length !== 4) {
      throw new Error('Unable to serialize array field targets - length must be 4')
    }
    // Serialize message field [targets]
    bufferOffset = _arraySerializer.int64(obj.targets, buffer, bufferOffset, 4);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Servos
    let len;
    let data = new Servos(null);
    // Deserialize message field [targets]
    data.targets = _arrayDeserializer.int64(buffer, bufferOffset, 4)
    return data;
  }

  static getMessageSize(object) {
    return 32;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/Servos';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'e8f52576b380ad554e7b8eee4133713d';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int64[4] targets
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Servos(null);
    if (msg.targets !== undefined) {
      resolved.targets = msg.targets;
    }
    else {
      resolved.targets = new Array(4).fill(0)
    }

    return resolved;
    }
};

module.exports = Servos;
