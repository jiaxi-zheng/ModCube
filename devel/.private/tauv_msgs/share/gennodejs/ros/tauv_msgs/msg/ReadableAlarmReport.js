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

class ReadableAlarmReport {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.failure_level = null;
      this.alarms_no_failure = null;
      this.alarms_predive_failure = null;
      this.alarms_mission_failure = null;
      this.alarms_critical_failure = null;
    }
    else {
      if (initObj.hasOwnProperty('failure_level')) {
        this.failure_level = initObj.failure_level
      }
      else {
        this.failure_level = '';
      }
      if (initObj.hasOwnProperty('alarms_no_failure')) {
        this.alarms_no_failure = initObj.alarms_no_failure
      }
      else {
        this.alarms_no_failure = [];
      }
      if (initObj.hasOwnProperty('alarms_predive_failure')) {
        this.alarms_predive_failure = initObj.alarms_predive_failure
      }
      else {
        this.alarms_predive_failure = [];
      }
      if (initObj.hasOwnProperty('alarms_mission_failure')) {
        this.alarms_mission_failure = initObj.alarms_mission_failure
      }
      else {
        this.alarms_mission_failure = [];
      }
      if (initObj.hasOwnProperty('alarms_critical_failure')) {
        this.alarms_critical_failure = initObj.alarms_critical_failure
      }
      else {
        this.alarms_critical_failure = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ReadableAlarmReport
    // Serialize message field [failure_level]
    bufferOffset = _serializer.string(obj.failure_level, buffer, bufferOffset);
    // Serialize message field [alarms_no_failure]
    bufferOffset = _arraySerializer.string(obj.alarms_no_failure, buffer, bufferOffset, null);
    // Serialize message field [alarms_predive_failure]
    bufferOffset = _arraySerializer.string(obj.alarms_predive_failure, buffer, bufferOffset, null);
    // Serialize message field [alarms_mission_failure]
    bufferOffset = _arraySerializer.string(obj.alarms_mission_failure, buffer, bufferOffset, null);
    // Serialize message field [alarms_critical_failure]
    bufferOffset = _arraySerializer.string(obj.alarms_critical_failure, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ReadableAlarmReport
    let len;
    let data = new ReadableAlarmReport(null);
    // Deserialize message field [failure_level]
    data.failure_level = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [alarms_no_failure]
    data.alarms_no_failure = _arrayDeserializer.string(buffer, bufferOffset, null)
    // Deserialize message field [alarms_predive_failure]
    data.alarms_predive_failure = _arrayDeserializer.string(buffer, bufferOffset, null)
    // Deserialize message field [alarms_mission_failure]
    data.alarms_mission_failure = _arrayDeserializer.string(buffer, bufferOffset, null)
    // Deserialize message field [alarms_critical_failure]
    data.alarms_critical_failure = _arrayDeserializer.string(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.failure_level);
    object.alarms_no_failure.forEach((val) => {
      length += 4 + _getByteLength(val);
    });
    object.alarms_predive_failure.forEach((val) => {
      length += 4 + _getByteLength(val);
    });
    object.alarms_mission_failure.forEach((val) => {
      length += 4 + _getByteLength(val);
    });
    object.alarms_critical_failure.forEach((val) => {
      length += 4 + _getByteLength(val);
    });
    return length + 20;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/ReadableAlarmReport';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '5420ac12365c08ba46b84a0a6b7db17d';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string failure_level
    string[] alarms_no_failure
    string[] alarms_predive_failure
    string[] alarms_mission_failure
    string[] alarms_critical_failure
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ReadableAlarmReport(null);
    if (msg.failure_level !== undefined) {
      resolved.failure_level = msg.failure_level;
    }
    else {
      resolved.failure_level = ''
    }

    if (msg.alarms_no_failure !== undefined) {
      resolved.alarms_no_failure = msg.alarms_no_failure;
    }
    else {
      resolved.alarms_no_failure = []
    }

    if (msg.alarms_predive_failure !== undefined) {
      resolved.alarms_predive_failure = msg.alarms_predive_failure;
    }
    else {
      resolved.alarms_predive_failure = []
    }

    if (msg.alarms_mission_failure !== undefined) {
      resolved.alarms_mission_failure = msg.alarms_mission_failure;
    }
    else {
      resolved.alarms_mission_failure = []
    }

    if (msg.alarms_critical_failure !== undefined) {
      resolved.alarms_critical_failure = msg.alarms_critical_failure;
    }
    else {
      resolved.alarms_critical_failure = []
    }

    return resolved;
    }
};

module.exports = ReadableAlarmReport;
