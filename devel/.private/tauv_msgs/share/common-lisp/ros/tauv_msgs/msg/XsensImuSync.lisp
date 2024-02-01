; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude XsensImuSync.msg.html

(cl:defclass <XsensImuSync> (roslisp-msg-protocol:ros-message)
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
   (d_corrected
    :reader d_corrected
    :initarg :d_corrected
    :type cl:float
    :initform 0.0)
   (d_ros
    :reader d_ros
    :initarg :d_ros
    :type cl:float
    :initform 0.0)
   (d_imu
    :reader d_imu
    :initarg :d_imu
    :type cl:float
    :initform 0.0))
)

(cl:defclass XsensImuSync (<XsensImuSync>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <XsensImuSync>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'XsensImuSync)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<XsensImuSync> is deprecated: use tauv_msgs-msg:XsensImuSync instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <XsensImuSync>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:header-val is deprecated.  Use tauv_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'ros_time-val :lambda-list '(m))
(cl:defmethod ros_time-val ((m <XsensImuSync>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:ros_time-val is deprecated.  Use tauv_msgs-msg:ros_time instead.")
  (ros_time m))

(cl:ensure-generic-function 'imu_time-val :lambda-list '(m))
(cl:defmethod imu_time-val ((m <XsensImuSync>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:imu_time-val is deprecated.  Use tauv_msgs-msg:imu_time instead.")
  (imu_time m))

(cl:ensure-generic-function 'triggered_dvl-val :lambda-list '(m))
(cl:defmethod triggered_dvl-val ((m <XsensImuSync>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:triggered_dvl-val is deprecated.  Use tauv_msgs-msg:triggered_dvl instead.")
  (triggered_dvl m))

(cl:ensure-generic-function 'd_corrected-val :lambda-list '(m))
(cl:defmethod d_corrected-val ((m <XsensImuSync>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:d_corrected-val is deprecated.  Use tauv_msgs-msg:d_corrected instead.")
  (d_corrected m))

(cl:ensure-generic-function 'd_ros-val :lambda-list '(m))
(cl:defmethod d_ros-val ((m <XsensImuSync>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:d_ros-val is deprecated.  Use tauv_msgs-msg:d_ros instead.")
  (d_ros m))

(cl:ensure-generic-function 'd_imu-val :lambda-list '(m))
(cl:defmethod d_imu-val ((m <XsensImuSync>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:d_imu-val is deprecated.  Use tauv_msgs-msg:d_imu instead.")
  (d_imu m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <XsensImuSync>) ostream)
  "Serializes a message object of type '<XsensImuSync>"
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
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'd_corrected))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'd_ros))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'd_imu))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <XsensImuSync>) istream)
  "Deserializes a message object of type '<XsensImuSync>"
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
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'd_corrected) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'd_ros) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'd_imu) (roslisp-utils:decode-double-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<XsensImuSync>)))
  "Returns string type for a message object of type '<XsensImuSync>"
  "tauv_msgs/XsensImuSync")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'XsensImuSync)))
  "Returns string type for a message object of type 'XsensImuSync"
  "tauv_msgs/XsensImuSync")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<XsensImuSync>)))
  "Returns md5sum for a message object of type '<XsensImuSync>"
  "8d95951a55cfd457da142f2cc6b05ae6")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'XsensImuSync)))
  "Returns md5sum for a message object of type 'XsensImuSync"
  "8d95951a55cfd457da142f2cc6b05ae6")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<XsensImuSync>)))
  "Returns full string definition for message of type '<XsensImuSync>"
  (cl:format cl:nil "Header header~%~%time ros_time~%~%time imu_time~%~%bool triggered_dvl~%~%float64 d_corrected # optional~%~%float64 d_ros # optional~%~%float64 d_imu # optional~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'XsensImuSync)))
  "Returns full string definition for message of type 'XsensImuSync"
  (cl:format cl:nil "Header header~%~%time ros_time~%~%time imu_time~%~%bool triggered_dvl~%~%float64 d_corrected # optional~%~%float64 d_ros # optional~%~%float64 d_imu # optional~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <XsensImuSync>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     8
     8
     1
     8
     8
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <XsensImuSync>))
  "Converts a ROS message object to a list"
  (cl:list 'XsensImuSync
    (cl:cons ':header (header msg))
    (cl:cons ':ros_time (ros_time msg))
    (cl:cons ':imu_time (imu_time msg))
    (cl:cons ':triggered_dvl (triggered_dvl msg))
    (cl:cons ':d_corrected (d_corrected msg))
    (cl:cons ':d_ros (d_ros msg))
    (cl:cons ':d_imu (d_imu msg))
))
