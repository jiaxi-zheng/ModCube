; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude FeatureDetection.msg.html

(cl:defclass <FeatureDetection> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (position
    :reader position
    :initarg :position
    :type geometry_msgs-msg:Point
    :initform (cl:make-instance 'geometry_msgs-msg:Point))
   (orientation
    :reader orientation
    :initarg :orientation
    :type geometry_msgs-msg:Point
    :initform (cl:make-instance 'geometry_msgs-msg:Point))
   (tag
    :reader tag
    :initarg :tag
    :type cl:string
    :initform "")
   (confidence
    :reader confidence
    :initarg :confidence
    :type cl:float
    :initform 0.0)
   (SE2
    :reader SE2
    :initarg :SE2
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass FeatureDetection (<FeatureDetection>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <FeatureDetection>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'FeatureDetection)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<FeatureDetection> is deprecated: use tauv_msgs-msg:FeatureDetection instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <FeatureDetection>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:header-val is deprecated.  Use tauv_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'position-val :lambda-list '(m))
(cl:defmethod position-val ((m <FeatureDetection>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:position-val is deprecated.  Use tauv_msgs-msg:position instead.")
  (position m))

(cl:ensure-generic-function 'orientation-val :lambda-list '(m))
(cl:defmethod orientation-val ((m <FeatureDetection>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:orientation-val is deprecated.  Use tauv_msgs-msg:orientation instead.")
  (orientation m))

(cl:ensure-generic-function 'tag-val :lambda-list '(m))
(cl:defmethod tag-val ((m <FeatureDetection>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:tag-val is deprecated.  Use tauv_msgs-msg:tag instead.")
  (tag m))

(cl:ensure-generic-function 'confidence-val :lambda-list '(m))
(cl:defmethod confidence-val ((m <FeatureDetection>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:confidence-val is deprecated.  Use tauv_msgs-msg:confidence instead.")
  (confidence m))

(cl:ensure-generic-function 'SE2-val :lambda-list '(m))
(cl:defmethod SE2-val ((m <FeatureDetection>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:SE2-val is deprecated.  Use tauv_msgs-msg:SE2 instead.")
  (SE2 m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <FeatureDetection>) ostream)
  "Serializes a message object of type '<FeatureDetection>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'position) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'orientation) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'tag))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'tag))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'confidence))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'SE2) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <FeatureDetection>) istream)
  "Deserializes a message object of type '<FeatureDetection>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'position) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'orientation) istream)
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'tag) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'tag) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'confidence) (roslisp-utils:decode-double-float-bits bits)))
    (cl:setf (cl:slot-value msg 'SE2) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<FeatureDetection>)))
  "Returns string type for a message object of type '<FeatureDetection>"
  "tauv_msgs/FeatureDetection")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'FeatureDetection)))
  "Returns string type for a message object of type 'FeatureDetection"
  "tauv_msgs/FeatureDetection")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<FeatureDetection>)))
  "Returns md5sum for a message object of type '<FeatureDetection>"
  "a256c9fb7a965addac0015cd369172cd")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'FeatureDetection)))
  "Returns md5sum for a message object of type 'FeatureDetection"
  "a256c9fb7a965addac0015cd369172cd")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<FeatureDetection>)))
  "Returns full string definition for message of type '<FeatureDetection>"
  (cl:format cl:nil "Header header~%geometry_msgs/Point position #SE2 msgs will only use x y~%geometry_msgs/Point orientation #SE2 msgs will only use x (theta)~%string tag~%float64 confidence~%bool SE2~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'FeatureDetection)))
  "Returns full string definition for message of type 'FeatureDetection"
  (cl:format cl:nil "Header header~%geometry_msgs/Point position #SE2 msgs will only use x y~%geometry_msgs/Point orientation #SE2 msgs will only use x (theta)~%string tag~%float64 confidence~%bool SE2~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <FeatureDetection>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'position))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'orientation))
     4 (cl:length (cl:slot-value msg 'tag))
     8
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <FeatureDetection>))
  "Converts a ROS message object to a list"
  (cl:list 'FeatureDetection
    (cl:cons ':header (header msg))
    (cl:cons ':position (position msg))
    (cl:cons ':orientation (orientation msg))
    (cl:cons ':tag (tag msg))
    (cl:cons ':confidence (confidence msg))
    (cl:cons ':SE2 (SE2 msg))
))
