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

class TeledyneDvlData {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.ensemble_number = null;
      this.test_status = null;
      this.health_status = null;
      this.shallow_mode = null;
      this.depth = null;
      this.pressure = null;
      this.pressure_variance = null;
      this.heading = null;
      this.pitch = null;
      this.roll = null;
      this.speed_of_sound = null;
      this.salinity = null;
      this.temperature = null;
      this.transmit_voltage = null;
      this.transmit_current = null;
      this.transmit_impedance = null;
      this.velocity = null;
      this.velocity_error = null;
      this.hr_velocity = null;
      this.hr_velocity_error = null;
      this.is_hr_velocity_valid = null;
      this.beam_ranges = null;
      this.beam_correlations = null;
      this.beam_amplitudes = null;
      this.beam_percent_goods = null;
      this.beam_time_to_bottoms = null;
      this.beam_standard_deviations = null;
      this.beam_time_of_validities = null;
      this.beam_rssis = null;
      this.slant_range = null;
      this.axis_delta_range = null;
      this.vertical_range = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('ensemble_number')) {
        this.ensemble_number = initObj.ensemble_number
      }
      else {
        this.ensemble_number = 0;
      }
      if (initObj.hasOwnProperty('test_status')) {
        this.test_status = initObj.test_status
      }
      else {
        this.test_status = new std_msgs.msg.String();
      }
      if (initObj.hasOwnProperty('health_status')) {
        this.health_status = initObj.health_status
      }
      else {
        this.health_status = new std_msgs.msg.String();
      }
      if (initObj.hasOwnProperty('shallow_mode')) {
        this.shallow_mode = initObj.shallow_mode
      }
      else {
        this.shallow_mode = false;
      }
      if (initObj.hasOwnProperty('depth')) {
        this.depth = initObj.depth
      }
      else {
        this.depth = 0.0;
      }
      if (initObj.hasOwnProperty('pressure')) {
        this.pressure = initObj.pressure
      }
      else {
        this.pressure = 0.0;
      }
      if (initObj.hasOwnProperty('pressure_variance')) {
        this.pressure_variance = initObj.pressure_variance
      }
      else {
        this.pressure_variance = 0.0;
      }
      if (initObj.hasOwnProperty('heading')) {
        this.heading = initObj.heading
      }
      else {
        this.heading = 0.0;
      }
      if (initObj.hasOwnProperty('pitch')) {
        this.pitch = initObj.pitch
      }
      else {
        this.pitch = 0.0;
      }
      if (initObj.hasOwnProperty('roll')) {
        this.roll = initObj.roll
      }
      else {
        this.roll = 0.0;
      }
      if (initObj.hasOwnProperty('speed_of_sound')) {
        this.speed_of_sound = initObj.speed_of_sound
      }
      else {
        this.speed_of_sound = 0.0;
      }
      if (initObj.hasOwnProperty('salinity')) {
        this.salinity = initObj.salinity
      }
      else {
        this.salinity = 0.0;
      }
      if (initObj.hasOwnProperty('temperature')) {
        this.temperature = initObj.temperature
      }
      else {
        this.temperature = 0.0;
      }
      if (initObj.hasOwnProperty('transmit_voltage')) {
        this.transmit_voltage = initObj.transmit_voltage
      }
      else {
        this.transmit_voltage = 0.0;
      }
      if (initObj.hasOwnProperty('transmit_current')) {
        this.transmit_current = initObj.transmit_current
      }
      else {
        this.transmit_current = 0.0;
      }
      if (initObj.hasOwnProperty('transmit_impedance')) {
        this.transmit_impedance = initObj.transmit_impedance
      }
      else {
        this.transmit_impedance = 0.0;
      }
      if (initObj.hasOwnProperty('velocity')) {
        this.velocity = initObj.velocity
      }
      else {
        this.velocity = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('velocity_error')) {
        this.velocity_error = initObj.velocity_error
      }
      else {
        this.velocity_error = 0.0;
      }
      if (initObj.hasOwnProperty('hr_velocity')) {
        this.hr_velocity = initObj.hr_velocity
      }
      else {
        this.hr_velocity = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('hr_velocity_error')) {
        this.hr_velocity_error = initObj.hr_velocity_error
      }
      else {
        this.hr_velocity_error = 0.0;
      }
      if (initObj.hasOwnProperty('is_hr_velocity_valid')) {
        this.is_hr_velocity_valid = initObj.is_hr_velocity_valid
      }
      else {
        this.is_hr_velocity_valid = false;
      }
      if (initObj.hasOwnProperty('beam_ranges')) {
        this.beam_ranges = initObj.beam_ranges
      }
      else {
        this.beam_ranges = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('beam_correlations')) {
        this.beam_correlations = initObj.beam_correlations
      }
      else {
        this.beam_correlations = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('beam_amplitudes')) {
        this.beam_amplitudes = initObj.beam_amplitudes
      }
      else {
        this.beam_amplitudes = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('beam_percent_goods')) {
        this.beam_percent_goods = initObj.beam_percent_goods
      }
      else {
        this.beam_percent_goods = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('beam_time_to_bottoms')) {
        this.beam_time_to_bottoms = initObj.beam_time_to_bottoms
      }
      else {
        this.beam_time_to_bottoms = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('beam_standard_deviations')) {
        this.beam_standard_deviations = initObj.beam_standard_deviations
      }
      else {
        this.beam_standard_deviations = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('beam_time_of_validities')) {
        this.beam_time_of_validities = initObj.beam_time_of_validities
      }
      else {
        this.beam_time_of_validities = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('beam_rssis')) {
        this.beam_rssis = initObj.beam_rssis
      }
      else {
        this.beam_rssis = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('slant_range')) {
        this.slant_range = initObj.slant_range
      }
      else {
        this.slant_range = 0.0;
      }
      if (initObj.hasOwnProperty('axis_delta_range')) {
        this.axis_delta_range = initObj.axis_delta_range
      }
      else {
        this.axis_delta_range = 0.0;
      }
      if (initObj.hasOwnProperty('vertical_range')) {
        this.vertical_range = initObj.vertical_range
      }
      else {
        this.vertical_range = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type TeledyneDvlData
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [ensemble_number]
    bufferOffset = _serializer.int64(obj.ensemble_number, buffer, bufferOffset);
    // Serialize message field [test_status]
    bufferOffset = std_msgs.msg.String.serialize(obj.test_status, buffer, bufferOffset);
    // Serialize message field [health_status]
    bufferOffset = std_msgs.msg.String.serialize(obj.health_status, buffer, bufferOffset);
    // Serialize message field [shallow_mode]
    bufferOffset = _serializer.bool(obj.shallow_mode, buffer, bufferOffset);
    // Serialize message field [depth]
    bufferOffset = _serializer.float64(obj.depth, buffer, bufferOffset);
    // Serialize message field [pressure]
    bufferOffset = _serializer.float64(obj.pressure, buffer, bufferOffset);
    // Serialize message field [pressure_variance]
    bufferOffset = _serializer.float64(obj.pressure_variance, buffer, bufferOffset);
    // Serialize message field [heading]
    bufferOffset = _serializer.float64(obj.heading, buffer, bufferOffset);
    // Serialize message field [pitch]
    bufferOffset = _serializer.float64(obj.pitch, buffer, bufferOffset);
    // Serialize message field [roll]
    bufferOffset = _serializer.float64(obj.roll, buffer, bufferOffset);
    // Serialize message field [speed_of_sound]
    bufferOffset = _serializer.float64(obj.speed_of_sound, buffer, bufferOffset);
    // Serialize message field [salinity]
    bufferOffset = _serializer.float64(obj.salinity, buffer, bufferOffset);
    // Serialize message field [temperature]
    bufferOffset = _serializer.float64(obj.temperature, buffer, bufferOffset);
    // Serialize message field [transmit_voltage]
    bufferOffset = _serializer.float64(obj.transmit_voltage, buffer, bufferOffset);
    // Serialize message field [transmit_current]
    bufferOffset = _serializer.float64(obj.transmit_current, buffer, bufferOffset);
    // Serialize message field [transmit_impedance]
    bufferOffset = _serializer.float64(obj.transmit_impedance, buffer, bufferOffset);
    // Serialize message field [velocity]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.velocity, buffer, bufferOffset);
    // Serialize message field [velocity_error]
    bufferOffset = _serializer.float64(obj.velocity_error, buffer, bufferOffset);
    // Serialize message field [hr_velocity]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.hr_velocity, buffer, bufferOffset);
    // Serialize message field [hr_velocity_error]
    bufferOffset = _serializer.float64(obj.hr_velocity_error, buffer, bufferOffset);
    // Serialize message field [is_hr_velocity_valid]
    bufferOffset = _serializer.bool(obj.is_hr_velocity_valid, buffer, bufferOffset);
    // Check that the constant length array field [beam_ranges] has the right length
    if (obj.beam_ranges.length !== 4) {
      throw new Error('Unable to serialize array field beam_ranges - length must be 4')
    }
    // Serialize message field [beam_ranges]
    bufferOffset = _arraySerializer.float64(obj.beam_ranges, buffer, bufferOffset, 4);
    // Check that the constant length array field [beam_correlations] has the right length
    if (obj.beam_correlations.length !== 4) {
      throw new Error('Unable to serialize array field beam_correlations - length must be 4')
    }
    // Serialize message field [beam_correlations]
    bufferOffset = _arraySerializer.float64(obj.beam_correlations, buffer, bufferOffset, 4);
    // Check that the constant length array field [beam_amplitudes] has the right length
    if (obj.beam_amplitudes.length !== 4) {
      throw new Error('Unable to serialize array field beam_amplitudes - length must be 4')
    }
    // Serialize message field [beam_amplitudes]
    bufferOffset = _arraySerializer.float64(obj.beam_amplitudes, buffer, bufferOffset, 4);
    // Check that the constant length array field [beam_percent_goods] has the right length
    if (obj.beam_percent_goods.length !== 4) {
      throw new Error('Unable to serialize array field beam_percent_goods - length must be 4')
    }
    // Serialize message field [beam_percent_goods]
    bufferOffset = _arraySerializer.float64(obj.beam_percent_goods, buffer, bufferOffset, 4);
    // Check that the constant length array field [beam_time_to_bottoms] has the right length
    if (obj.beam_time_to_bottoms.length !== 4) {
      throw new Error('Unable to serialize array field beam_time_to_bottoms - length must be 4')
    }
    // Serialize message field [beam_time_to_bottoms]
    bufferOffset = _arraySerializer.float64(obj.beam_time_to_bottoms, buffer, bufferOffset, 4);
    // Check that the constant length array field [beam_standard_deviations] has the right length
    if (obj.beam_standard_deviations.length !== 4) {
      throw new Error('Unable to serialize array field beam_standard_deviations - length must be 4')
    }
    // Serialize message field [beam_standard_deviations]
    bufferOffset = _arraySerializer.float64(obj.beam_standard_deviations, buffer, bufferOffset, 4);
    // Check that the constant length array field [beam_time_of_validities] has the right length
    if (obj.beam_time_of_validities.length !== 4) {
      throw new Error('Unable to serialize array field beam_time_of_validities - length must be 4')
    }
    // Serialize message field [beam_time_of_validities]
    bufferOffset = _arraySerializer.float64(obj.beam_time_of_validities, buffer, bufferOffset, 4);
    // Check that the constant length array field [beam_rssis] has the right length
    if (obj.beam_rssis.length !== 4) {
      throw new Error('Unable to serialize array field beam_rssis - length must be 4')
    }
    // Serialize message field [beam_rssis]
    bufferOffset = _arraySerializer.float64(obj.beam_rssis, buffer, bufferOffset, 4);
    // Serialize message field [slant_range]
    bufferOffset = _serializer.float64(obj.slant_range, buffer, bufferOffset);
    // Serialize message field [axis_delta_range]
    bufferOffset = _serializer.float64(obj.axis_delta_range, buffer, bufferOffset);
    // Serialize message field [vertical_range]
    bufferOffset = _serializer.float64(obj.vertical_range, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type TeledyneDvlData
    let len;
    let data = new TeledyneDvlData(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [ensemble_number]
    data.ensemble_number = _deserializer.int64(buffer, bufferOffset);
    // Deserialize message field [test_status]
    data.test_status = std_msgs.msg.String.deserialize(buffer, bufferOffset);
    // Deserialize message field [health_status]
    data.health_status = std_msgs.msg.String.deserialize(buffer, bufferOffset);
    // Deserialize message field [shallow_mode]
    data.shallow_mode = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [depth]
    data.depth = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [pressure]
    data.pressure = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [pressure_variance]
    data.pressure_variance = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [heading]
    data.heading = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [pitch]
    data.pitch = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [roll]
    data.roll = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [speed_of_sound]
    data.speed_of_sound = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [salinity]
    data.salinity = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [temperature]
    data.temperature = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [transmit_voltage]
    data.transmit_voltage = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [transmit_current]
    data.transmit_current = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [transmit_impedance]
    data.transmit_impedance = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [velocity]
    data.velocity = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [velocity_error]
    data.velocity_error = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [hr_velocity]
    data.hr_velocity = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [hr_velocity_error]
    data.hr_velocity_error = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [is_hr_velocity_valid]
    data.is_hr_velocity_valid = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [beam_ranges]
    data.beam_ranges = _arrayDeserializer.float64(buffer, bufferOffset, 4)
    // Deserialize message field [beam_correlations]
    data.beam_correlations = _arrayDeserializer.float64(buffer, bufferOffset, 4)
    // Deserialize message field [beam_amplitudes]
    data.beam_amplitudes = _arrayDeserializer.float64(buffer, bufferOffset, 4)
    // Deserialize message field [beam_percent_goods]
    data.beam_percent_goods = _arrayDeserializer.float64(buffer, bufferOffset, 4)
    // Deserialize message field [beam_time_to_bottoms]
    data.beam_time_to_bottoms = _arrayDeserializer.float64(buffer, bufferOffset, 4)
    // Deserialize message field [beam_standard_deviations]
    data.beam_standard_deviations = _arrayDeserializer.float64(buffer, bufferOffset, 4)
    // Deserialize message field [beam_time_of_validities]
    data.beam_time_of_validities = _arrayDeserializer.float64(buffer, bufferOffset, 4)
    // Deserialize message field [beam_rssis]
    data.beam_rssis = _arrayDeserializer.float64(buffer, bufferOffset, 4)
    // Deserialize message field [slant_range]
    data.slant_range = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [axis_delta_range]
    data.axis_delta_range = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [vertical_range]
    data.vertical_range = _deserializer.float64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += std_msgs.msg.String.getMessageSize(object.test_status);
    length += std_msgs.msg.String.getMessageSize(object.health_status);
    return length + 450;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/TeledyneDvlData';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '65bd41b11739b2abbd5b158cfef08147';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    
    int64 ensemble_number
    
    std_msgs/String test_status
    std_msgs/String health_status
    
    bool shallow_mode
    float64 depth
    float64 pressure
    float64 pressure_variance
    
    float64 heading
    float64 pitch
    float64 roll
    float64 speed_of_sound
    float64 salinity
    float64 temperature
    
    float64 transmit_voltage
    float64 transmit_current
    float64 transmit_impedance
    
    geometry_msgs/Vector3 velocity
    float64 velocity_error
    
    geometry_msgs/Vector3 hr_velocity
    float64 hr_velocity_error
    bool is_hr_velocity_valid
    
    float64[4] beam_ranges
    float64[4] beam_correlations
    float64[4] beam_amplitudes
    float64[4] beam_percent_goods
    float64[4] beam_time_to_bottoms
    float64[4] beam_standard_deviations
    float64[4] beam_time_of_validities
    float64[4] beam_rssis
    
    float64 slant_range
    float64 axis_delta_range
    float64 vertical_range
    
    
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
    MSG: std_msgs/String
    string data
    
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
    const resolved = new TeledyneDvlData(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.ensemble_number !== undefined) {
      resolved.ensemble_number = msg.ensemble_number;
    }
    else {
      resolved.ensemble_number = 0
    }

    if (msg.test_status !== undefined) {
      resolved.test_status = std_msgs.msg.String.Resolve(msg.test_status)
    }
    else {
      resolved.test_status = new std_msgs.msg.String()
    }

    if (msg.health_status !== undefined) {
      resolved.health_status = std_msgs.msg.String.Resolve(msg.health_status)
    }
    else {
      resolved.health_status = new std_msgs.msg.String()
    }

    if (msg.shallow_mode !== undefined) {
      resolved.shallow_mode = msg.shallow_mode;
    }
    else {
      resolved.shallow_mode = false
    }

    if (msg.depth !== undefined) {
      resolved.depth = msg.depth;
    }
    else {
      resolved.depth = 0.0
    }

    if (msg.pressure !== undefined) {
      resolved.pressure = msg.pressure;
    }
    else {
      resolved.pressure = 0.0
    }

    if (msg.pressure_variance !== undefined) {
      resolved.pressure_variance = msg.pressure_variance;
    }
    else {
      resolved.pressure_variance = 0.0
    }

    if (msg.heading !== undefined) {
      resolved.heading = msg.heading;
    }
    else {
      resolved.heading = 0.0
    }

    if (msg.pitch !== undefined) {
      resolved.pitch = msg.pitch;
    }
    else {
      resolved.pitch = 0.0
    }

    if (msg.roll !== undefined) {
      resolved.roll = msg.roll;
    }
    else {
      resolved.roll = 0.0
    }

    if (msg.speed_of_sound !== undefined) {
      resolved.speed_of_sound = msg.speed_of_sound;
    }
    else {
      resolved.speed_of_sound = 0.0
    }

    if (msg.salinity !== undefined) {
      resolved.salinity = msg.salinity;
    }
    else {
      resolved.salinity = 0.0
    }

    if (msg.temperature !== undefined) {
      resolved.temperature = msg.temperature;
    }
    else {
      resolved.temperature = 0.0
    }

    if (msg.transmit_voltage !== undefined) {
      resolved.transmit_voltage = msg.transmit_voltage;
    }
    else {
      resolved.transmit_voltage = 0.0
    }

    if (msg.transmit_current !== undefined) {
      resolved.transmit_current = msg.transmit_current;
    }
    else {
      resolved.transmit_current = 0.0
    }

    if (msg.transmit_impedance !== undefined) {
      resolved.transmit_impedance = msg.transmit_impedance;
    }
    else {
      resolved.transmit_impedance = 0.0
    }

    if (msg.velocity !== undefined) {
      resolved.velocity = geometry_msgs.msg.Vector3.Resolve(msg.velocity)
    }
    else {
      resolved.velocity = new geometry_msgs.msg.Vector3()
    }

    if (msg.velocity_error !== undefined) {
      resolved.velocity_error = msg.velocity_error;
    }
    else {
      resolved.velocity_error = 0.0
    }

    if (msg.hr_velocity !== undefined) {
      resolved.hr_velocity = geometry_msgs.msg.Vector3.Resolve(msg.hr_velocity)
    }
    else {
      resolved.hr_velocity = new geometry_msgs.msg.Vector3()
    }

    if (msg.hr_velocity_error !== undefined) {
      resolved.hr_velocity_error = msg.hr_velocity_error;
    }
    else {
      resolved.hr_velocity_error = 0.0
    }

    if (msg.is_hr_velocity_valid !== undefined) {
      resolved.is_hr_velocity_valid = msg.is_hr_velocity_valid;
    }
    else {
      resolved.is_hr_velocity_valid = false
    }

    if (msg.beam_ranges !== undefined) {
      resolved.beam_ranges = msg.beam_ranges;
    }
    else {
      resolved.beam_ranges = new Array(4).fill(0)
    }

    if (msg.beam_correlations !== undefined) {
      resolved.beam_correlations = msg.beam_correlations;
    }
    else {
      resolved.beam_correlations = new Array(4).fill(0)
    }

    if (msg.beam_amplitudes !== undefined) {
      resolved.beam_amplitudes = msg.beam_amplitudes;
    }
    else {
      resolved.beam_amplitudes = new Array(4).fill(0)
    }

    if (msg.beam_percent_goods !== undefined) {
      resolved.beam_percent_goods = msg.beam_percent_goods;
    }
    else {
      resolved.beam_percent_goods = new Array(4).fill(0)
    }

    if (msg.beam_time_to_bottoms !== undefined) {
      resolved.beam_time_to_bottoms = msg.beam_time_to_bottoms;
    }
    else {
      resolved.beam_time_to_bottoms = new Array(4).fill(0)
    }

    if (msg.beam_standard_deviations !== undefined) {
      resolved.beam_standard_deviations = msg.beam_standard_deviations;
    }
    else {
      resolved.beam_standard_deviations = new Array(4).fill(0)
    }

    if (msg.beam_time_of_validities !== undefined) {
      resolved.beam_time_of_validities = msg.beam_time_of_validities;
    }
    else {
      resolved.beam_time_of_validities = new Array(4).fill(0)
    }

    if (msg.beam_rssis !== undefined) {
      resolved.beam_rssis = msg.beam_rssis;
    }
    else {
      resolved.beam_rssis = new Array(4).fill(0)
    }

    if (msg.slant_range !== undefined) {
      resolved.slant_range = msg.slant_range;
    }
    else {
      resolved.slant_range = 0.0
    }

    if (msg.axis_delta_range !== undefined) {
      resolved.axis_delta_range = msg.axis_delta_range;
    }
    else {
      resolved.axis_delta_range = 0.0
    }

    if (msg.vertical_range !== undefined) {
      resolved.vertical_range = msg.vertical_range;
    }
    else {
      resolved.vertical_range = 0.0
    }

    return resolved;
    }
};

module.exports = TeledyneDvlData;
