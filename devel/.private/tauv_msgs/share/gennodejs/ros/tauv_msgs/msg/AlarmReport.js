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

class AlarmReport {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.stamp = null;
      this.active_alarms = null;
    }
    else {
      if (initObj.hasOwnProperty('stamp')) {
        this.stamp = initObj.stamp
      }
      else {
        this.stamp = {secs: 0, nsecs: 0};
      }
      if (initObj.hasOwnProperty('active_alarms')) {
        this.active_alarms = initObj.active_alarms
      }
      else {
        this.active_alarms = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type AlarmReport
    // Serialize message field [stamp]
    bufferOffset = _serializer.time(obj.stamp, buffer, bufferOffset);
    // Serialize message field [active_alarms]
    bufferOffset = _arraySerializer.int32(obj.active_alarms, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type AlarmReport
    let len;
    let data = new AlarmReport(null);
    // Deserialize message field [stamp]
    data.stamp = _deserializer.time(buffer, bufferOffset);
    // Deserialize message field [active_alarms]
    data.active_alarms = _arrayDeserializer.int32(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 4 * object.active_alarms.length;
    return length + 12;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/AlarmReport';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '6041271f37a12a54ca5b8c77ba39eab9';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    time stamp
    int32[] active_alarms
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new AlarmReport(null);
    if (msg.stamp !== undefined) {
      resolved.stamp = msg.stamp;
    }
    else {
      resolved.stamp = {secs: 0, nsecs: 0}
    }

    if (msg.active_alarms !== undefined) {
      resolved.active_alarms = msg.active_alarms;
    }
    else {
      resolved.active_alarms = []
    }

    return resolved;
    }
};

module.exports = AlarmReport;
