// Auto-generated. Do not edit!

// (in-package tauv_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let AlarmWithMessage = require('../msg/AlarmWithMessage.js');

//-----------------------------------------------------------


//-----------------------------------------------------------

class SyncAlarmsRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.diff = null;
    }
    else {
      if (initObj.hasOwnProperty('diff')) {
        this.diff = initObj.diff
      }
      else {
        this.diff = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SyncAlarmsRequest
    // Serialize message field [diff]
    // Serialize the length for message field [diff]
    bufferOffset = _serializer.uint32(obj.diff.length, buffer, bufferOffset);
    obj.diff.forEach((val) => {
      bufferOffset = AlarmWithMessage.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SyncAlarmsRequest
    let len;
    let data = new SyncAlarmsRequest(null);
    // Deserialize message field [diff]
    // Deserialize array length for message field [diff]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.diff = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.diff[i] = AlarmWithMessage.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.diff.forEach((val) => {
      length += AlarmWithMessage.getMessageSize(val);
    });
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/SyncAlarmsRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '968621d8f52d5d5fb6c3529a8a70d975';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Note: Angular velocities outside of yaw (z axis) are currently unused.
    
    tauv_msgs/AlarmWithMessage[] diff
    
    
    ================================================================================
    MSG: tauv_msgs/AlarmWithMessage
    int32 id            # ID of the alarm
    bool set            # True = set, False = Cleared
    string message      # Readable message
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SyncAlarmsRequest(null);
    if (msg.diff !== undefined) {
      resolved.diff = new Array(msg.diff.length);
      for (let i = 0; i < resolved.diff.length; ++i) {
        resolved.diff[i] = AlarmWithMessage.Resolve(msg.diff[i]);
      }
    }
    else {
      resolved.diff = []
    }

    return resolved;
    }
};

class SyncAlarmsResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.stamp = null;
      this.active_alarms = null;
      this.success = null;
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
      if (initObj.hasOwnProperty('success')) {
        this.success = initObj.success
      }
      else {
        this.success = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SyncAlarmsResponse
    // Serialize message field [stamp]
    bufferOffset = _serializer.time(obj.stamp, buffer, bufferOffset);
    // Serialize message field [active_alarms]
    bufferOffset = _arraySerializer.int32(obj.active_alarms, buffer, bufferOffset, null);
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SyncAlarmsResponse
    let len;
    let data = new SyncAlarmsResponse(null);
    // Deserialize message field [stamp]
    data.stamp = _deserializer.time(buffer, bufferOffset);
    // Deserialize message field [active_alarms]
    data.active_alarms = _arrayDeserializer.int32(buffer, bufferOffset, null)
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 4 * object.active_alarms.length;
    return length + 13;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/SyncAlarmsResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'a432d60ab588421c6ba91ab7242b55f2';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    time stamp
    int32[] active_alarms
    bool success  # false indicates some sort of failure
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SyncAlarmsResponse(null);
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
  Request: SyncAlarmsRequest,
  Response: SyncAlarmsResponse,
  md5sum() { return '54b1739021e723bf57d59b6622adc3ef'; },
  datatype() { return 'tauv_msgs/SyncAlarms'; }
};
