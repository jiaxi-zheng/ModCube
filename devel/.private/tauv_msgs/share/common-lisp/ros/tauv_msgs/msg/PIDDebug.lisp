; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude PIDDebug.msg.html

(cl:defclass <PIDDebug> (roslisp-msg-protocol:ros-message)
  ((tuning
    :reader tuning
    :initarg :tuning
    :type tauv_msgs-msg:PIDTuning
    :initform (cl:make-instance 'tauv_msgs-msg:PIDTuning))
   (value
    :reader value
    :initarg :value
    :type cl:float
    :initform 0.0)
   (setpoint
    :reader setpoint
    :initarg :setpoint
    :type cl:float
    :initform 0.0)
   (error
    :reader error
    :initarg :error
    :type cl:float
    :initform 0.0)
   (proportional
    :reader proportional
    :initarg :proportional
    :type cl:float
    :initform 0.0)
   (integral
    :reader integral
    :initarg :integral
    :type cl:float
    :initform 0.0)
   (derivative
    :reader derivative
    :initarg :derivative
    :type cl:float
    :initform 0.0)
   (effort
    :reader effort
    :initarg :effort
    :type cl:float
    :initform 0.0))
)

(cl:defclass PIDDebug (<PIDDebug>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PIDDebug>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PIDDebug)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<PIDDebug> is deprecated: use tauv_msgs-msg:PIDDebug instead.")))

(cl:ensure-generic-function 'tuning-val :lambda-list '(m))
(cl:defmethod tuning-val ((m <PIDDebug>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:tuning-val is deprecated.  Use tauv_msgs-msg:tuning instead.")
  (tuning m))

(cl:ensure-generic-function 'value-val :lambda-list '(m))
(cl:defmethod value-val ((m <PIDDebug>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:value-val is deprecated.  Use tauv_msgs-msg:value instead.")
  (value m))

(cl:ensure-generic-function 'setpoint-val :lambda-list '(m))
(cl:defmethod setpoint-val ((m <PIDDebug>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:setpoint-val is deprecated.  Use tauv_msgs-msg:setpoint instead.")
  (setpoint m))

(cl:ensure-generic-function 'error-val :lambda-list '(m))
(cl:defmethod error-val ((m <PIDDebug>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:error-val is deprecated.  Use tauv_msgs-msg:error instead.")
  (error m))

(cl:ensure-generic-function 'proportional-val :lambda-list '(m))
(cl:defmethod proportional-val ((m <PIDDebug>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:proportional-val is deprecated.  Use tauv_msgs-msg:proportional instead.")
  (proportional m))

(cl:ensure-generic-function 'integral-val :lambda-list '(m))
(cl:defmethod integral-val ((m <PIDDebug>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:integral-val is deprecated.  Use tauv_msgs-msg:integral instead.")
  (integral m))

(cl:ensure-generic-function 'derivative-val :lambda-list '(m))
(cl:defmethod derivative-val ((m <PIDDebug>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:derivative-val is deprecated.  Use tauv_msgs-msg:derivative instead.")
  (derivative m))

(cl:ensure-generic-function 'effort-val :lambda-list '(m))
(cl:defmethod effort-val ((m <PIDDebug>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:effort-val is deprecated.  Use tauv_msgs-msg:effort instead.")
  (effort m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PIDDebug>) ostream)
  "Serializes a message object of type '<PIDDebug>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'tuning) ostream)
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'value))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'setpoint))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'error))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'proportional))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'integral))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'derivative))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'effort))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PIDDebug>) istream)
  "Deserializes a message object of type '<PIDDebug>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'tuning) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'value) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'setpoint) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'error) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'proportional) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'integral) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'derivative) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'effort) (roslisp-utils:decode-double-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PIDDebug>)))
  "Returns string type for a message object of type '<PIDDebug>"
  "tauv_msgs/PIDDebug")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PIDDebug)))
  "Returns string type for a message object of type 'PIDDebug"
  "tauv_msgs/PIDDebug")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PIDDebug>)))
  "Returns md5sum for a message object of type '<PIDDebug>"
  "16ed0a79b3cb076d76fdb91c1c560d6a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PIDDebug)))
  "Returns md5sum for a message object of type 'PIDDebug"
  "16ed0a79b3cb076d76fdb91c1c560d6a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PIDDebug>)))
  "Returns full string definition for message of type '<PIDDebug>"
  (cl:format cl:nil "PIDTuning tuning~%~%float64 value~%float64 setpoint~%float64 error~%float64 proportional~%float64 integral~%float64 derivative~%float64 effort~%================================================================================~%MSG: tauv_msgs/PIDTuning~%string axis~%float64 kp~%float64 ki~%float64 kd~%float64 tau~%float64[2] limits~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PIDDebug)))
  "Returns full string definition for message of type 'PIDDebug"
  (cl:format cl:nil "PIDTuning tuning~%~%float64 value~%float64 setpoint~%float64 error~%float64 proportional~%float64 integral~%float64 derivative~%float64 effort~%================================================================================~%MSG: tauv_msgs/PIDTuning~%string axis~%float64 kp~%float64 ki~%float64 kd~%float64 tau~%float64[2] limits~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PIDDebug>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'tuning))
     8
     8
     8
     8
     8
     8
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PIDDebug>))
  "Converts a ROS message object to a list"
  (cl:list 'PIDDebug
    (cl:cons ':tuning (tuning msg))
    (cl:cons ':value (value msg))
    (cl:cons ':setpoint (setpoint msg))
    (cl:cons ':error (error msg))
    (cl:cons ':proportional (proportional msg))
    (cl:cons ':integral (integral msg))
    (cl:cons ':derivative (derivative msg))
    (cl:cons ':effort (effort msg))
))
