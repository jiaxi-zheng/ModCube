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

class DynamicsTuning {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.update_mass = null;
      this.mass = null;
      this.update_volume = null;
      this.volume = null;
      this.update_water_density = null;
      this.water_density = null;
      this.update_center_of_gravity = null;
      this.center_of_gravity = null;
      this.update_center_of_buoyancy = null;
      this.center_of_buoyancy = null;
      this.update_moments = null;
      this.moments = null;
      this.update_linear_damping = null;
      this.linear_damping = null;
      this.update_quadratic_damping = null;
      this.quadratic_damping = null;
      this.update_added_mass = null;
      this.added_mass = null;
    }
    else {
      if (initObj.hasOwnProperty('update_mass')) {
        this.update_mass = initObj.update_mass
      }
      else {
        this.update_mass = false;
      }
      if (initObj.hasOwnProperty('mass')) {
        this.mass = initObj.mass
      }
      else {
        this.mass = 0.0;
      }
      if (initObj.hasOwnProperty('update_volume')) {
        this.update_volume = initObj.update_volume
      }
      else {
        this.update_volume = false;
      }
      if (initObj.hasOwnProperty('volume')) {
        this.volume = initObj.volume
      }
      else {
        this.volume = 0.0;
      }
      if (initObj.hasOwnProperty('update_water_density')) {
        this.update_water_density = initObj.update_water_density
      }
      else {
        this.update_water_density = false;
      }
      if (initObj.hasOwnProperty('water_density')) {
        this.water_density = initObj.water_density
      }
      else {
        this.water_density = 0.0;
      }
      if (initObj.hasOwnProperty('update_center_of_gravity')) {
        this.update_center_of_gravity = initObj.update_center_of_gravity
      }
      else {
        this.update_center_of_gravity = false;
      }
      if (initObj.hasOwnProperty('center_of_gravity')) {
        this.center_of_gravity = initObj.center_of_gravity
      }
      else {
        this.center_of_gravity = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('update_center_of_buoyancy')) {
        this.update_center_of_buoyancy = initObj.update_center_of_buoyancy
      }
      else {
        this.update_center_of_buoyancy = false;
      }
      if (initObj.hasOwnProperty('center_of_buoyancy')) {
        this.center_of_buoyancy = initObj.center_of_buoyancy
      }
      else {
        this.center_of_buoyancy = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('update_moments')) {
        this.update_moments = initObj.update_moments
      }
      else {
        this.update_moments = false;
      }
      if (initObj.hasOwnProperty('moments')) {
        this.moments = initObj.moments
      }
      else {
        this.moments = new Array(6).fill(0);
      }
      if (initObj.hasOwnProperty('update_linear_damping')) {
        this.update_linear_damping = initObj.update_linear_damping
      }
      else {
        this.update_linear_damping = false;
      }
      if (initObj.hasOwnProperty('linear_damping')) {
        this.linear_damping = initObj.linear_damping
      }
      else {
        this.linear_damping = new Array(6).fill(0);
      }
      if (initObj.hasOwnProperty('update_quadratic_damping')) {
        this.update_quadratic_damping = initObj.update_quadratic_damping
      }
      else {
        this.update_quadratic_damping = false;
      }
      if (initObj.hasOwnProperty('quadratic_damping')) {
        this.quadratic_damping = initObj.quadratic_damping
      }
      else {
        this.quadratic_damping = new Array(6).fill(0);
      }
      if (initObj.hasOwnProperty('update_added_mass')) {
        this.update_added_mass = initObj.update_added_mass
      }
      else {
        this.update_added_mass = false;
      }
      if (initObj.hasOwnProperty('added_mass')) {
        this.added_mass = initObj.added_mass
      }
      else {
        this.added_mass = new Array(6).fill(0);
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type DynamicsTuning
    // Serialize message field [update_mass]
    bufferOffset = _serializer.bool(obj.update_mass, buffer, bufferOffset);
    // Serialize message field [mass]
    bufferOffset = _serializer.float64(obj.mass, buffer, bufferOffset);
    // Serialize message field [update_volume]
    bufferOffset = _serializer.bool(obj.update_volume, buffer, bufferOffset);
    // Serialize message field [volume]
    bufferOffset = _serializer.float64(obj.volume, buffer, bufferOffset);
    // Serialize message field [update_water_density]
    bufferOffset = _serializer.bool(obj.update_water_density, buffer, bufferOffset);
    // Serialize message field [water_density]
    bufferOffset = _serializer.float64(obj.water_density, buffer, bufferOffset);
    // Serialize message field [update_center_of_gravity]
    bufferOffset = _serializer.bool(obj.update_center_of_gravity, buffer, bufferOffset);
    // Check that the constant length array field [center_of_gravity] has the right length
    if (obj.center_of_gravity.length !== 3) {
      throw new Error('Unable to serialize array field center_of_gravity - length must be 3')
    }
    // Serialize message field [center_of_gravity]
    bufferOffset = _arraySerializer.float64(obj.center_of_gravity, buffer, bufferOffset, 3);
    // Serialize message field [update_center_of_buoyancy]
    bufferOffset = _serializer.bool(obj.update_center_of_buoyancy, buffer, bufferOffset);
    // Check that the constant length array field [center_of_buoyancy] has the right length
    if (obj.center_of_buoyancy.length !== 3) {
      throw new Error('Unable to serialize array field center_of_buoyancy - length must be 3')
    }
    // Serialize message field [center_of_buoyancy]
    bufferOffset = _arraySerializer.float64(obj.center_of_buoyancy, buffer, bufferOffset, 3);
    // Serialize message field [update_moments]
    bufferOffset = _serializer.bool(obj.update_moments, buffer, bufferOffset);
    // Check that the constant length array field [moments] has the right length
    if (obj.moments.length !== 6) {
      throw new Error('Unable to serialize array field moments - length must be 6')
    }
    // Serialize message field [moments]
    bufferOffset = _arraySerializer.float64(obj.moments, buffer, bufferOffset, 6);
    // Serialize message field [update_linear_damping]
    bufferOffset = _serializer.bool(obj.update_linear_damping, buffer, bufferOffset);
    // Check that the constant length array field [linear_damping] has the right length
    if (obj.linear_damping.length !== 6) {
      throw new Error('Unable to serialize array field linear_damping - length must be 6')
    }
    // Serialize message field [linear_damping]
    bufferOffset = _arraySerializer.float64(obj.linear_damping, buffer, bufferOffset, 6);
    // Serialize message field [update_quadratic_damping]
    bufferOffset = _serializer.bool(obj.update_quadratic_damping, buffer, bufferOffset);
    // Check that the constant length array field [quadratic_damping] has the right length
    if (obj.quadratic_damping.length !== 6) {
      throw new Error('Unable to serialize array field quadratic_damping - length must be 6')
    }
    // Serialize message field [quadratic_damping]
    bufferOffset = _arraySerializer.float64(obj.quadratic_damping, buffer, bufferOffset, 6);
    // Serialize message field [update_added_mass]
    bufferOffset = _serializer.bool(obj.update_added_mass, buffer, bufferOffset);
    // Check that the constant length array field [added_mass] has the right length
    if (obj.added_mass.length !== 6) {
      throw new Error('Unable to serialize array field added_mass - length must be 6')
    }
    // Serialize message field [added_mass]
    bufferOffset = _arraySerializer.float64(obj.added_mass, buffer, bufferOffset, 6);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type DynamicsTuning
    let len;
    let data = new DynamicsTuning(null);
    // Deserialize message field [update_mass]
    data.update_mass = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [mass]
    data.mass = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [update_volume]
    data.update_volume = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [volume]
    data.volume = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [update_water_density]
    data.update_water_density = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [water_density]
    data.water_density = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [update_center_of_gravity]
    data.update_center_of_gravity = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [center_of_gravity]
    data.center_of_gravity = _arrayDeserializer.float64(buffer, bufferOffset, 3)
    // Deserialize message field [update_center_of_buoyancy]
    data.update_center_of_buoyancy = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [center_of_buoyancy]
    data.center_of_buoyancy = _arrayDeserializer.float64(buffer, bufferOffset, 3)
    // Deserialize message field [update_moments]
    data.update_moments = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [moments]
    data.moments = _arrayDeserializer.float64(buffer, bufferOffset, 6)
    // Deserialize message field [update_linear_damping]
    data.update_linear_damping = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [linear_damping]
    data.linear_damping = _arrayDeserializer.float64(buffer, bufferOffset, 6)
    // Deserialize message field [update_quadratic_damping]
    data.update_quadratic_damping = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [quadratic_damping]
    data.quadratic_damping = _arrayDeserializer.float64(buffer, bufferOffset, 6)
    // Deserialize message field [update_added_mass]
    data.update_added_mass = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [added_mass]
    data.added_mass = _arrayDeserializer.float64(buffer, bufferOffset, 6)
    return data;
  }

  static getMessageSize(object) {
    return 273;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tauv_msgs/DynamicsTuning';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '1657e74b53352c5e93a01a5d1743eeaa';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool update_mass
    float64 mass
    bool update_volume
    float64 volume
    bool update_water_density
    float64 water_density
    bool update_center_of_gravity
    float64[3] center_of_gravity
    bool update_center_of_buoyancy
    float64[3] center_of_buoyancy
    bool update_moments
    float64[6] moments
    bool update_linear_damping
    float64[6] linear_damping
    bool update_quadratic_damping
    float64[6] quadratic_damping
    bool update_added_mass
    float64[6] added_mass
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new DynamicsTuning(null);
    if (msg.update_mass !== undefined) {
      resolved.update_mass = msg.update_mass;
    }
    else {
      resolved.update_mass = false
    }

    if (msg.mass !== undefined) {
      resolved.mass = msg.mass;
    }
    else {
      resolved.mass = 0.0
    }

    if (msg.update_volume !== undefined) {
      resolved.update_volume = msg.update_volume;
    }
    else {
      resolved.update_volume = false
    }

    if (msg.volume !== undefined) {
      resolved.volume = msg.volume;
    }
    else {
      resolved.volume = 0.0
    }

    if (msg.update_water_density !== undefined) {
      resolved.update_water_density = msg.update_water_density;
    }
    else {
      resolved.update_water_density = false
    }

    if (msg.water_density !== undefined) {
      resolved.water_density = msg.water_density;
    }
    else {
      resolved.water_density = 0.0
    }

    if (msg.update_center_of_gravity !== undefined) {
      resolved.update_center_of_gravity = msg.update_center_of_gravity;
    }
    else {
      resolved.update_center_of_gravity = false
    }

    if (msg.center_of_gravity !== undefined) {
      resolved.center_of_gravity = msg.center_of_gravity;
    }
    else {
      resolved.center_of_gravity = new Array(3).fill(0)
    }

    if (msg.update_center_of_buoyancy !== undefined) {
      resolved.update_center_of_buoyancy = msg.update_center_of_buoyancy;
    }
    else {
      resolved.update_center_of_buoyancy = false
    }

    if (msg.center_of_buoyancy !== undefined) {
      resolved.center_of_buoyancy = msg.center_of_buoyancy;
    }
    else {
      resolved.center_of_buoyancy = new Array(3).fill(0)
    }

    if (msg.update_moments !== undefined) {
      resolved.update_moments = msg.update_moments;
    }
    else {
      resolved.update_moments = false
    }

    if (msg.moments !== undefined) {
      resolved.moments = msg.moments;
    }
    else {
      resolved.moments = new Array(6).fill(0)
    }

    if (msg.update_linear_damping !== undefined) {
      resolved.update_linear_damping = msg.update_linear_damping;
    }
    else {
      resolved.update_linear_damping = false
    }

    if (msg.linear_damping !== undefined) {
      resolved.linear_damping = msg.linear_damping;
    }
    else {
      resolved.linear_damping = new Array(6).fill(0)
    }

    if (msg.update_quadratic_damping !== undefined) {
      resolved.update_quadratic_damping = msg.update_quadratic_damping;
    }
    else {
      resolved.update_quadratic_damping = false
    }

    if (msg.quadratic_damping !== undefined) {
      resolved.quadratic_damping = msg.quadratic_damping;
    }
    else {
      resolved.quadratic_damping = new Array(6).fill(0)
    }

    if (msg.update_added_mass !== undefined) {
      resolved.update_added_mass = msg.update_added_mass;
    }
    else {
      resolved.update_added_mass = false
    }

    if (msg.added_mass !== undefined) {
      resolved.added_mass = msg.added_mass;
    }
    else {
      resolved.added_mass = new Array(6).fill(0)
    }

    return resolved;
    }
};

module.exports = DynamicsTuning;
