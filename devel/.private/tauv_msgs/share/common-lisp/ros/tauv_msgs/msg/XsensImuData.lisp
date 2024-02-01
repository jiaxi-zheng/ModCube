; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude XsensImuData.msg.html

(cl:defclass <XsensImuData> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (ros_time
    :reader ros_time
    :initarg :ros_time
    :type cl:real
    :initform 0)
   (imu_time
    :reader imu_time
    :initarg :imu_time
    :type cl:real
    :initform 0)
   (triggered_dvl
    :reader triggered_dvl
    :initarg :triggered_dvl
    :type cl:boolean
    :initform cl:nil)
   (orientation
    :reader orientation
    :initarg :orientation
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (rate_of_turn
    :reader rate_of_turn
    :initarg :rate_of_turn
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (linear_acceleration
    :reader linear_acceleration
    :initarg :linear_acceleration
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (free_acceleration
    :reader free_acceleration
    :initarg :free_acceleration
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3)))
)

(cl:defclass XsensImuData (<XsensImuData>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <XsensImuData>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'XsensImuData)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<XsensImuData> is deprecated: use tauv_msgs-msg:XsensImuData instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <XsensImuData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:header-val is deprecated.  Use tauv_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'ros_time-val :lambda-list '(m))
(cl:defmethod ros_time-val ((m <XsensImuData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:ros_time-val is deprecated.  Use tauv_msgs-msg:ros_time instead.")
  (ros_time m))

(cl:ensure-generic-function 'imu_time-val :lambda-list '(m))
(cl:defmethod imu_time-val ((m <XsensImuData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:imu_time-val is deprecated.  Use tauv_msgs-msg:imu_time instead.")
  (imu_time m))

(cl:ensure-generic-function 'triggered_dvl-val :lambda-list '(m))
(cl:defmethod triggered_dvl-val ((m <XsensImuData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:triggered_dvl-val is deprecated.  Use tauv_msgs-msg:triggered_dvl instead.")
  (triggered_dvl m))

(cl:ensure-generic-function 'orientation-val :lambda-list '(m))
(cl:defmethod orientation-val ((m <XsensImuData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:orientation-val is deprecated.  Use tauv_msgs-msg:orientation instead.")
  (orientation m))

(cl:ensure-generic-function 'rate_of_turn-val :lambda-list '(m))
(cl:defmethod rate_of_turn-val ((m <XsensImuData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:rate_of_turn-val is deprecated.  Use tauv_msgs-msg:rate_of_turn instead.")
  (rate_of_turn m))

(cl:ensure-generic-function 'linear_acceleration-val :lambda-list '(m))
(cl:defmethod linear_acceleration-val ((m <XsensImuData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:linear_acceleration-val is deprecated.  Use tauv_msgs-msg:linear_acceleration instead.")
  (linear_acceleration m))

(cl:ensure-generic-function 'free_acceleration-val :lambda-list '(m))
(cl:defmethod free_acceleration-val ((m <XsensImuData>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:free_acceleration-val is deprecated.  Use tauv_msgs-msg:free_acceleration instead.")
  (free_acceleration m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <XsensImuData>) ostream)
  "Serializes a message object of type '<XsensImuData>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((__sec (cl:floor (cl:slot-value msg 'ros_time)))
        (__nsec (cl:round (cl:* 1e9 (cl:- (cl:slot-value msg 'ros_time) (cl:floor (cl:slot-value msg 'ros_time)))))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 0) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __nsec) ostream))
  (cl:let ((__sec (cl:floor (cl:slot-value msg 'imu_time)))
        (__nsec (cl:round (cl:* 1e9 (cl:- (cl:slot-value msg 'imu_time) (cl:floor (cl:slot-value msg 'imu_time)))))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 0) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __nsec) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'triggered_dvl) 1 0)) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'orientation) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'rate_of_turn) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'linear_acceleration) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'free_acceleration) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <XsensImuData>) istream)
  "Deserializes a message object of type '<XsensImuData>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((__sec 0) (__nsec 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 0) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __nsec) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'ros_time) (cl:+ (cl:coerce __sec 'cl:double-float) (cl:/ __nsec 1e9))))
    (cl:let ((__sec 0) (__nsec 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 0) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __nsec) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'imu_time) (cl:+ (cl:coerce __sec 'cl:double-float) (cl:/ __nsec 1e9))))
    (cl:setf (cl:slot-value msg 'triggered_dvl) (cl:not (cl:zerop (cl:read-byte istream))))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'orientation) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'rate_of_turn) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'linear_acceleration) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'free_acceleration) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<XsensImuData>)))
  "Returns string type for a message object of type '<XsensImuData>"
  "tauv_msgs/XsensImuData")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'XsensImuData)))
  "Returns string type for a message object of type 'XsensImuData"
  "tauv_msgs/XsensImuData")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<XsensImuData>)))
  "Returns md5sum for a message object of type '<XsensImuData>"
  "a5e7b14c591863b869ed2281b0f6b1ed")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'XsensImuData)))
  "Returns md5sum for a message object of type 'XsensImuData"
  "a5e7b14c591863b869ed2281b0f6b1ed")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<XsensImuData>)))
  "Returns full string definition for message of type '<XsensImuData>"
  (cl:format cl:nil "Header header~%~%time ros_time~%~%time imu_time~%~%bool triggered_dvl~%~%geometry_msgs/Vector3 orientation~%~%geometry_msgs/Vector3 rate_of_turn~%~%geometry_msgs/Vector3 linear_acceleration~%~%geometry_msgs/Vector3 free_acceleration~%~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'XsensImuData)))
  "Returns full string definition for message of type 'XsensImuData"
  (cl:format cl:nil "Header header~%~%time ros_time~%~%time imu_time~%~%bool triggered_dvl~%~%geometry_msgs/Vector3 orientation~%~%geometry_msgs/Vector3 rate_of_turn~%~%geometry_msgs/Vector3 linear_acceleration~%~%geometry_msgs/Vector3 free_acceleration~%~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <XsensImuData>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     8
     8
     1
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'orientation))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'rate_of_turn))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'linear_acceleration))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'free_acceleration))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <XsensImuData>))
  "Converts a ROS message object to a list"
  (cl:list 'XsensImuData
    (cl:cons ':header (header msg))
    (cl:cons ':ros_time (ros_time msg))
    (cl:cons ':imu_time (imu_time msg))
    (cl:cons ':triggered_dvl (triggered_dvl msg))
    (cl:cons ':orientation (orientation msg))
    (cl:cons ':rate_of_turn (rate_of_turn msg))
    (cl:cons ':linear_acceleration (linear_acceleration msg))
    (cl:cons ':free_acceleration (free_acceleration msg))
))
