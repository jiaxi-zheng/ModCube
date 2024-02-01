; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude PingDetection.msg.html

(cl:defclass <PingDetection> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (frequency
    :reader frequency
    :initarg :frequency
    :type cl:float
    :initform 0.0)
   (direction
    :reader direction
    :initarg :direction
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3)))
)

(cl:defclass PingDetection (<PingDetection>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PingDetection>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PingDetection)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<PingDetection> is deprecated: use tauv_msgs-msg:PingDetection instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <PingDetection>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:header-val is deprecated.  Use tauv_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'frequency-val :lambda-list '(m))
(cl:defmethod frequency-val ((m <PingDetection>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:frequency-val is deprecated.  Use tauv_msgs-msg:frequency instead.")
  (frequency m))

(cl:ensure-generic-function 'direction-val :lambda-list '(m))
(cl:defmethod direction-val ((m <PingDetection>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:direction-val is deprecated.  Use tauv_msgs-msg:direction instead.")
  (direction m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PingDetection>) ostream)
  "Serializes a message object of type '<PingDetection>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'frequency))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'direction) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PingDetection>) istream)
  "Deserializes a message object of type '<PingDetection>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'frequency) (roslisp-utils:decode-double-float-bits bits)))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'direction) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PingDetection>)))
  "Returns string type for a message object of type '<PingDetection>"
  "tauv_msgs/PingDetection")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PingDetection)))
  "Returns string type for a message object of type 'PingDetection"
  "tauv_msgs/PingDetection")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PingDetection>)))
  "Returns md5sum for a message object of type '<PingDetection>"
  "a9abd7bac39db1726e1408962ec95f05")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PingDetection)))
  "Returns md5sum for a message object of type 'PingDetection"
  "a9abd7bac39db1726e1408962ec95f05")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PingDetection>)))
  "Returns full string definition for message of type '<PingDetection>"
  (cl:format cl:nil "Header header~%float64 frequency~%geometry_msgs/Vector3 direction~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PingDetection)))
  "Returns full string definition for message of type 'PingDetection"
  (cl:format cl:nil "Header header~%float64 frequency~%geometry_msgs/Vector3 direction~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PingDetection>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     8
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'direction))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PingDetection>))
  "Converts a ROS message object to a list"
  (cl:list 'PingDetection
    (cl:cons ':header (header msg))
    (cl:cons ':frequency (frequency msg))
    (cl:cons ':direction (direction msg))
))
