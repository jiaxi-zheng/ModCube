; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude ControllerDebug.msg.html

(cl:defclass <ControllerDebug> (roslisp-msg-protocol:ros-message)
  ((stamp
    :reader stamp
    :initarg :stamp
    :type cl:real
    :initform 0)
   (z
    :reader z
    :initarg :z
    :type tauv_msgs-msg:PIDDebug
    :initform (cl:make-instance 'tauv_msgs-msg:PIDDebug))
   (roll
    :reader roll
    :initarg :roll
    :type tauv_msgs-msg:PIDDebug
    :initform (cl:make-instance 'tauv_msgs-msg:PIDDebug))
   (pitch
    :reader pitch
    :initarg :pitch
    :type tauv_msgs-msg:PIDDebug
    :initform (cl:make-instance 'tauv_msgs-msg:PIDDebug)))
)

(cl:defclass ControllerDebug (<ControllerDebug>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ControllerDebug>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ControllerDebug)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<ControllerDebug> is deprecated: use tauv_msgs-msg:ControllerDebug instead.")))

(cl:ensure-generic-function 'stamp-val :lambda-list '(m))
(cl:defmethod stamp-val ((m <ControllerDebug>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:stamp-val is deprecated.  Use tauv_msgs-msg:stamp instead.")
  (stamp m))

(cl:ensure-generic-function 'z-val :lambda-list '(m))
(cl:defmethod z-val ((m <ControllerDebug>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:z-val is deprecated.  Use tauv_msgs-msg:z instead.")
  (z m))

(cl:ensure-generic-function 'roll-val :lambda-list '(m))
(cl:defmethod roll-val ((m <ControllerDebug>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:roll-val is deprecated.  Use tauv_msgs-msg:roll instead.")
  (roll m))

(cl:ensure-generic-function 'pitch-val :lambda-list '(m))
(cl:defmethod pitch-val ((m <ControllerDebug>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:pitch-val is deprecated.  Use tauv_msgs-msg:pitch instead.")
  (pitch m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ControllerDebug>) ostream)
  "Serializes a message object of type '<ControllerDebug>"
  (cl:let ((__sec (cl:floor (cl:slot-value msg 'stamp)))
        (__nsec (cl:round (cl:* 1e9 (cl:- (cl:slot-value msg 'stamp) (cl:floor (cl:slot-value msg 'stamp)))))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 0) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __nsec) ostream))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'z) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'roll) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'pitch) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ControllerDebug>) istream)
  "Deserializes a message object of type '<ControllerDebug>"
    (cl:let ((__sec 0) (__nsec 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 0) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __nsec) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'stamp) (cl:+ (cl:coerce __sec 'cl:double-float) (cl:/ __nsec 1e9))))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'z) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'roll) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'pitch) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ControllerDebug>)))
  "Returns string type for a message object of type '<ControllerDebug>"
  "tauv_msgs/ControllerDebug")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ControllerDebug)))
  "Returns string type for a message object of type 'ControllerDebug"
  "tauv_msgs/ControllerDebug")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ControllerDebug>)))
  "Returns md5sum for a message object of type '<ControllerDebug>"
  "bb322eef3e5c11d51e1cb54450d584b1")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ControllerDebug)))
  "Returns md5sum for a message object of type 'ControllerDebug"
  "bb322eef3e5c11d51e1cb54450d584b1")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ControllerDebug>)))
  "Returns full string definition for message of type '<ControllerDebug>"
  (cl:format cl:nil "time stamp~%~%PIDDebug z~%PIDDebug roll~%PIDDebug pitch~%================================================================================~%MSG: tauv_msgs/PIDDebug~%PIDTuning tuning~%~%float64 value~%float64 setpoint~%float64 error~%float64 proportional~%float64 integral~%float64 derivative~%float64 effort~%================================================================================~%MSG: tauv_msgs/PIDTuning~%string axis~%float64 kp~%float64 ki~%float64 kd~%float64 tau~%float64[2] limits~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ControllerDebug)))
  "Returns full string definition for message of type 'ControllerDebug"
  (cl:format cl:nil "time stamp~%~%PIDDebug z~%PIDDebug roll~%PIDDebug pitch~%================================================================================~%MSG: tauv_msgs/PIDDebug~%PIDTuning tuning~%~%float64 value~%float64 setpoint~%float64 error~%float64 proportional~%float64 integral~%float64 derivative~%float64 effort~%================================================================================~%MSG: tauv_msgs/PIDTuning~%string axis~%float64 kp~%float64 ki~%float64 kd~%float64 tau~%float64[2] limits~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ControllerDebug>))
  (cl:+ 0
     8
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'z))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'roll))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'pitch))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ControllerDebug>))
  "Converts a ROS message object to a list"
  (cl:list 'ControllerDebug
    (cl:cons ':stamp (stamp msg))
    (cl:cons ':z (z msg))
    (cl:cons ':roll (roll msg))
    (cl:cons ':pitch (pitch msg))
))
