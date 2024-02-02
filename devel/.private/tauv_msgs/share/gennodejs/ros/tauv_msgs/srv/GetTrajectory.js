// Auto-generated. Do not edit!

// (in-package tauv_msgs.srv)


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


//-----------------------------------------------------------

class GetTrajectoryRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.curr_pose = null;
      this.curr_twist = null;
      this.curr_time = null;
      this.len = null;
      this.dt = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('curr_pose')) {
        this.curr_pose = initObj.curr_pose
      }
      else {
        this.curr_pose = new geometry_msgs.msg.Pose();
      }
      if (initObj.hasOwnProperty('curr_twist')) {
        this.curr_twist = initObj.curr_twist
      }
      else {
        this.curr_twist = new geometry_msgs.msg.Twist();
      }
      if (initObj.hasOwnProperty('curr_time')) {
        this.curr_time = initObj.curr_time
      }
      else {
        this.curr_time = {secs: 0, nsecs: 0};
      }
      if (initObj.hasOwnProperty('len')) {
        this.len = initObj.len
      }
      else {
        this.len = 0;
      }
      if (initObj.hasOwnProperty('dt')) {
        this.dt = initObj.dt
      }
      else {
        this.dt = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetTrajectoryRequest
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [curr_pose]
    bufferOffset = geometry_msgs.msg.Pose.serialize(obj.curr_pose, buffer, bufferOffset);
    // Serialize message field [curr_twist]
    bufferOffset = geometry_msgs.msg.Twist.serialize(obj.curr_twist, buffer, bufferOffset);
    // Serialize message field [curr_time]
    bufferOffset = _serializer.time(obj.curr_time, buffer, bufferOffset);
    // Serialize message field [len]
    bufferOffset = _serializer.int32(obj.len, buffer, bufferOffset);
    // Serialize message field [dt]
    bufferOffset = _serializer.float32(obj.dt, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetTrajectoryRequest
    let len;
    let data = new GetTrajectoryRequest(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [curr_pose]
    data.curr_pose = geometry_msgs.msg.Pose.deserialize(buffer, bufferOffset);
    // Deserialize message field [curr_twist]
    data.curr_twist = geometry_msgs.msg.Twist.deserialize(buffer, bufferOffset);
    // Deserialize message field [curr_time]
    data.curr_time = _deserializer.time(buffer, bufferOffset);
    // Deserialize message field [len]
    data.len = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [dt]
    data.dt = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    return length + 120;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/GetTrajectoryRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd20d43afe10933a453d78bb46d60acf3';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Note: Angular velocities outside of yaw (z axis) are currently unused.
    
    std_msgs/Header header
    geometry_msgs/Pose curr_pose  # Current positions
    geometry_msgs/Twist curr_twist  # Current velocities (in world frame! Not body velocities!)
    time curr_time
    int32 len  # Number of samples to look ahead on the trajectory. (First sample corresponds to current time, second is time + dt, etc)
    float32 dt  # time difference between samples
    
    
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
    MSG: geometry_msgs/Pose
    # A representation of pose in free space, composed of position and orientation. 
    Point position
    Quaternion orientation
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    ================================================================================
    MSG: geometry_msgs/Quaternion
    # This represents an orientation in free space in quaternion form.
    
    float64 x
    float64 y
    float64 z
    float64 w
    
    ================================================================================
    MSG: geometry_msgs/Twist
    # This expresses velocity in free space broken into its linear and angular parts.
    Vector3  linear
    Vector3  angular
    
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
    const resolved = new GetTrajectoryRequest(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.curr_pose !== undefined) {
      resolved.curr_pose = geometry_msgs.msg.Pose.Resolve(msg.curr_pose)
    }
    else {
      resolved.curr_pose = new geometry_msgs.msg.Pose()
    }

    if (msg.curr_twist !== undefined) {
      resolved.curr_twist = geometry_msgs.msg.Twist.Resolve(msg.curr_twist)
    }
    else {
      resolved.curr_twist = new geometry_msgs.msg.Twist()
    }

    if (msg.curr_time !== undefined) {
      resolved.curr_time = msg.curr_time;
    }
    else {
      resolved.curr_time = {secs: 0, nsecs: 0}
    }

    if (msg.len !== undefined) {
      resolved.len = msg.len;
    }
    else {
      resolved.len = 0
    }

    if (msg.dt !== undefined) {
      resolved.dt = msg.dt;
    }
    else {
      resolved.dt = 0.0
    }

    return resolved;
    }
};

class GetTrajectoryResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.poses = null;
      this.twists = null;
      this.success = null;
      this.auto_twists = null;
      this.message = null;
    }
    else {
      if (initObj.hasOwnProperty('poses')) {
        this.poses = initObj.poses
      }
      else {
        this.poses = [];
      }
      if (initObj.hasOwnProperty('twists')) {
        this.twists = initObj.twists
      }
      else {
        this.twists = [];
      }
      if (initObj.hasOwnProperty('success')) {
        this.success = initObj.success
      }
      else {
        this.success = false;
      }
      if (initObj.hasOwnProperty('auto_twists')) {
        this.auto_twists = initObj.auto_twists
      }
      else {
        this.auto_twists = false;
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
    // Serializes a message object of type GetTrajectoryResponse
    // Serialize message field [poses]
    // Serialize the length for message field [poses]
    bufferOffset = _serializer.uint32(obj.poses.length, buffer, bufferOffset);
    obj.poses.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Pose.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [twists]
    // Serialize the length for message field [twists]
    bufferOffset = _serializer.uint32(obj.twists.length, buffer, bufferOffset);
    obj.twists.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Twist.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    // Serialize message field [auto_twists]
    bufferOffset = _serializer.bool(obj.auto_twists, buffer, bufferOffset);
    // Serialize message field [message]
    bufferOffset = _serializer.string(obj.message, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetTrajectoryResponse
    let len;
    let data = new GetTrajectoryResponse(null);
    // Deserialize message field [poses]
    // Deserialize array length for message field [poses]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.poses = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.poses[i] = geometry_msgs.msg.Pose.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [twists]
    // Deserialize array length for message field [twists]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.twists = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.twists[i] = geometry_msgs.msg.Twist.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [auto_twists]
    data.auto_twists = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [message]
    data.message = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 56 * object.poses.length;
    length += 48 * object.twists.length;
    length += _getByteLength(object.message);
    return length + 14;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tauv_msgs/GetTrajectoryResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '441dacfa79b4a746f7f7e5eb72b39605';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    geometry_msgs/Pose[] poses  # list of poses on trajectory
    geometry_msgs/Twist[] twists  # list of twists on trajectory (in world frame! Not body velocities!)
    bool success  # false indicates some sort of failure
    bool auto_twists
    string message
    
    ================================================================================
    MSG: geometry_msgs/Pose
    # A representation of pose in free space, composed of position and orientation. 
    Point position
    Quaternion orientation
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    ================================================================================
    MSG: geometry_msgs/Quaternion
    # This represents an orientation in free space in quaternion form.
    
    float64 x
    float64 y
    float64 z
    float64 w
    
    ================================================================================
    MSG: geometry_msgs/Twist
    # This expresses velocity in free space broken into its linear and angular parts.
    Vector3  linear
    Vector3  angular
    
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
    const resolved = new GetTrajectoryResponse(null);
    if (msg.poses !== undefined) {
      resolved.poses = new Array(msg.poses.length);
      for (let i = 0; i < resolved.poses.length; ++i) {
        resolved.poses[i] = geometry_msgs.msg.Pose.Resolve(msg.poses[i]);
      }
    }
    else {
      resolved.poses = []
    }

    if (msg.twists !== undefined) {
      resolved.twists = new Array(msg.twists.length);
      for (let i = 0; i < resolved.twists.length; ++i) {
        resolved.twists[i] = geometry_msgs.msg.Twist.Resolve(msg.twists[i]);
      }
    }
    else {
      resolved.twists = []
    }

    if (msg.success !== undefined) {
      resolved.success = msg.success;
    }
    else {
      resolved.success = false
    }

    if (msg.auto_twists !== undefined) {
      resolved.auto_twists = msg.auto_twists;
    }
    else {
      resolved.auto_twists = false
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

module.exports = {
  Request: GetTrajectoryRequest,
  Response: GetTrajectoryResponse,
  md5sum() { return '44146c6b1e5214a56fdd6c14fe6382be'; },
  datatype() { return 'tauv_msgs/GetTrajectory'; }
};
