; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude ControllerCommand.msg.html

(cl:defclass <ControllerCommand> (roslisp-msg-protocol:ros-message)
  ((a_x
    :reader a_x
    :initarg :a_x
    :type cl:float
    :initform 0.0)
   (a_y
    :reader a_y
    :initarg :a_y
    :type cl:float
    :initform 0.0)
   (a_z
    :reader a_z
    :initarg :a_z
    :type cl:float
    :initform 0.0)
   (a_roll
    :reader a_roll
    :initarg :a_roll
    :type cl:float
    :initform 0.0)
   (a_pitch
    :reader a_pitch
    :initarg :a_pitch
    :type cl:float
    :initform 0.0)
   (a_yaw
    :reader a_yaw
    :initarg :a_yaw
    :type cl:float
    :initform 0.0)
   (f_x
    :reader f_x
    :initarg :f_x
    :type cl:float
    :initform 0.0)
   (f_y
    :reader f_y
    :initarg :f_y
    :type cl:float
    :initform 0.0)
   (f_z
    :reader f_z
    :initarg :f_z
    :type cl:float
    :initform 0.0)
   (f_roll
    :reader f_roll
    :initarg :f_roll
    :type cl:float
    :initform 0.0)
   (f_pitch
    :reader f_pitch
    :initarg :f_pitch
    :type cl:float
    :initform 0.0)
   (f_yaw
    :reader f_yaw
    :initarg :f_yaw
    :type cl:float
    :initform 0.0)
   (use_f_x
    :reader use_f_x
    :initarg :use_f_x
    :type cl:boolean
    :initform cl:nil)
   (use_f_y
    :reader use_f_y
    :initarg :use_f_y
    :type cl:boolean
    :initform cl:nil)
   (use_f_z
    :reader use_f_z
    :initarg :use_f_z
    :type cl:boolean
    :initform cl:nil)
   (use_f_roll
    :reader use_f_roll
    :initarg :use_f_roll
    :type cl:boolean
    :initform cl:nil)
   (use_f_pitch
    :reader use_f_pitch
    :initarg :use_f_pitch
    :type cl:boolean
    :initform cl:nil)
   (use_f_yaw
    :reader use_f_yaw
    :initarg :use_f_yaw
    :type cl:boolean
    :initform cl:nil)
   (setpoint_z
    :reader setpoint_z
    :initarg :setpoint_z
    :type cl:float
    :initform 0.0)
   (setpoint_roll
    :reader setpoint_roll
    :initarg :setpoint_roll
    :type cl:float
    :initform 0.0)
   (setpoint_pitch
    :reader setpoint_pitch
    :initarg :setpoint_pitch
    :type cl:float
    :initform 0.0)
   (use_setpoint_z
    :reader use_setpoint_z
    :initarg :use_setpoint_z
    :type cl:boolean
    :initform cl:nil)
   (use_setpoint_roll
    :reader use_setpoint_roll
    :initarg :use_setpoint_roll
    :type cl:boolean
    :initform cl:nil)
   (use_setpoint_pitch
    :reader use_setpoint_pitch
    :initarg :use_setpoint_pitch
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass ControllerCommand (<ControllerCommand>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ControllerCommand>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ControllerCommand)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<ControllerCommand> is deprecated: use tauv_msgs-msg:ControllerCommand instead.")))

(cl:ensure-generic-function 'a_x-val :lambda-list '(m))
(cl:defmethod a_x-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:a_x-val is deprecated.  Use tauv_msgs-msg:a_x instead.")
  (a_x m))

(cl:ensure-generic-function 'a_y-val :lambda-list '(m))
(cl:defmethod a_y-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:a_y-val is deprecated.  Use tauv_msgs-msg:a_y instead.")
  (a_y m))

(cl:ensure-generic-function 'a_z-val :lambda-list '(m))
(cl:defmethod a_z-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:a_z-val is deprecated.  Use tauv_msgs-msg:a_z instead.")
  (a_z m))

(cl:ensure-generic-function 'a_roll-val :lambda-list '(m))
(cl:defmethod a_roll-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:a_roll-val is deprecated.  Use tauv_msgs-msg:a_roll instead.")
  (a_roll m))

(cl:ensure-generic-function 'a_pitch-val :lambda-list '(m))
(cl:defmethod a_pitch-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:a_pitch-val is deprecated.  Use tauv_msgs-msg:a_pitch instead.")
  (a_pitch m))

(cl:ensure-generic-function 'a_yaw-val :lambda-list '(m))
(cl:defmethod a_yaw-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:a_yaw-val is deprecated.  Use tauv_msgs-msg:a_yaw instead.")
  (a_yaw m))

(cl:ensure-generic-function 'f_x-val :lambda-list '(m))
(cl:defmethod f_x-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:f_x-val is deprecated.  Use tauv_msgs-msg:f_x instead.")
  (f_x m))

(cl:ensure-generic-function 'f_y-val :lambda-list '(m))
(cl:defmethod f_y-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:f_y-val is deprecated.  Use tauv_msgs-msg:f_y instead.")
  (f_y m))

(cl:ensure-generic-function 'f_z-val :lambda-list '(m))
(cl:defmethod f_z-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:f_z-val is deprecated.  Use tauv_msgs-msg:f_z instead.")
  (f_z m))

(cl:ensure-generic-function 'f_roll-val :lambda-list '(m))
(cl:defmethod f_roll-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:f_roll-val is deprecated.  Use tauv_msgs-msg:f_roll instead.")
  (f_roll m))

(cl:ensure-generic-function 'f_pitch-val :lambda-list '(m))
(cl:defmethod f_pitch-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:f_pitch-val is deprecated.  Use tauv_msgs-msg:f_pitch instead.")
  (f_pitch m))

(cl:ensure-generic-function 'f_yaw-val :lambda-list '(m))
(cl:defmethod f_yaw-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:f_yaw-val is deprecated.  Use tauv_msgs-msg:f_yaw instead.")
  (f_yaw m))

(cl:ensure-generic-function 'use_f_x-val :lambda-list '(m))
(cl:defmethod use_f_x-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:use_f_x-val is deprecated.  Use tauv_msgs-msg:use_f_x instead.")
  (use_f_x m))

(cl:ensure-generic-function 'use_f_y-val :lambda-list '(m))
(cl:defmethod use_f_y-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:use_f_y-val is deprecated.  Use tauv_msgs-msg:use_f_y instead.")
  (use_f_y m))

(cl:ensure-generic-function 'use_f_z-val :lambda-list '(m))
(cl:defmethod use_f_z-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:use_f_z-val is deprecated.  Use tauv_msgs-msg:use_f_z instead.")
  (use_f_z m))

(cl:ensure-generic-function 'use_f_roll-val :lambda-list '(m))
(cl:defmethod use_f_roll-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:use_f_roll-val is deprecated.  Use tauv_msgs-msg:use_f_roll instead.")
  (use_f_roll m))

(cl:ensure-generic-function 'use_f_pitch-val :lambda-list '(m))
(cl:defmethod use_f_pitch-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:use_f_pitch-val is deprecated.  Use tauv_msgs-msg:use_f_pitch instead.")
  (use_f_pitch m))

(cl:ensure-generic-function 'use_f_yaw-val :lambda-list '(m))
(cl:defmethod use_f_yaw-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:use_f_yaw-val is deprecated.  Use tauv_msgs-msg:use_f_yaw instead.")
  (use_f_yaw m))

(cl:ensure-generic-function 'setpoint_z-val :lambda-list '(m))
(cl:defmethod setpoint_z-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:setpoint_z-val is deprecated.  Use tauv_msgs-msg:setpoint_z instead.")
  (setpoint_z m))

(cl:ensure-generic-function 'setpoint_roll-val :lambda-list '(m))
(cl:defmethod setpoint_roll-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:setpoint_roll-val is deprecated.  Use tauv_msgs-msg:setpoint_roll instead.")
  (setpoint_roll m))

(cl:ensure-generic-function 'setpoint_pitch-val :lambda-list '(m))
(cl:defmethod setpoint_pitch-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:setpoint_pitch-val is deprecated.  Use tauv_msgs-msg:setpoint_pitch instead.")
  (setpoint_pitch m))

(cl:ensure-generic-function 'use_setpoint_z-val :lambda-list '(m))
(cl:defmethod use_setpoint_z-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:use_setpoint_z-val is deprecated.  Use tauv_msgs-msg:use_setpoint_z instead.")
  (use_setpoint_z m))

(cl:ensure-generic-function 'use_setpoint_roll-val :lambda-list '(m))
(cl:defmethod use_setpoint_roll-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:use_setpoint_roll-val is deprecated.  Use tauv_msgs-msg:use_setpoint_roll instead.")
  (use_setpoint_roll m))

(cl:ensure-generic-function 'use_setpoint_pitch-val :lambda-list '(m))
(cl:defmethod use_setpoint_pitch-val ((m <ControllerCommand>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:use_setpoint_pitch-val is deprecated.  Use tauv_msgs-msg:use_setpoint_pitch instead.")
  (use_setpoint_pitch m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ControllerCommand>) ostream)
  "Serializes a message object of type '<ControllerCommand>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'a_x))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'a_y))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'a_z))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'a_roll))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'a_pitch))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'a_yaw))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'f_x))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'f_y))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'f_z))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'f_roll))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'f_pitch))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'f_yaw))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'use_f_x) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'use_f_y) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'use_f_z) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'use_f_roll) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'use_f_pitch) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'use_f_yaw) 1 0)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'setpoint_z))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'setpoint_roll))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'setpoint_pitch))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'use_setpoint_z) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'use_setpoint_roll) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'use_setpoint_pitch) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ControllerCommand>) istream)
  "Deserializes a message object of type '<ControllerCommand>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'a_x) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'a_y) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'a_z) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'a_roll) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'a_pitch) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'a_yaw) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'f_x) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'f_y) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'f_z) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'f_roll) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'f_pitch) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'f_yaw) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:slot-value msg 'use_f_x) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'use_f_y) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'use_f_z) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'use_f_roll) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'use_f_pitch) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'use_f_yaw) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'setpoint_z) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'setpoint_roll) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'setpoint_pitch) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:slot-value msg 'use_setpoint_z) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'use_setpoint_roll) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'use_setpoint_pitch) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ControllerCommand>)))
  "Returns string type for a message object of type '<ControllerCommand>"
  "tauv_msgs/ControllerCommand")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ControllerCommand)))
  "Returns string type for a message object of type 'ControllerCommand"
  "tauv_msgs/ControllerCommand")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ControllerCommand>)))
  "Returns md5sum for a message object of type '<ControllerCommand>"
  "ad1b57ce703dafd167a4f28711d03e4e")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ControllerCommand)))
  "Returns md5sum for a message object of type 'ControllerCommand"
  "ad1b57ce703dafd167a4f28711d03e4e")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ControllerCommand>)))
  "Returns full string definition for message of type '<ControllerCommand>"
  (cl:format cl:nil "# Accelerations~%float32 a_x~%float32 a_y~%float32 a_z~%float32 a_roll~%float32 a_pitch~%float32 a_yaw~%~%# Forces~%float32 f_x~%float32 f_y~%float32 f_z~%float32 f_roll~%float32 f_pitch~%float32 f_yaw~%~%# If set, override accelerations and use forces~%bool use_f_x~%bool use_f_y~%bool use_f_z~%bool use_f_roll~%bool use_f_pitch~%bool use_f_yaw~%~%# Setpoints~%float32 setpoint_z~%float32 setpoint_roll~%float32 setpoint_pitch~%~%# If set, override accelerations and forces and use setpoints~%bool use_setpoint_z~%bool use_setpoint_roll~%bool use_setpoint_pitch~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ControllerCommand)))
  "Returns full string definition for message of type 'ControllerCommand"
  (cl:format cl:nil "# Accelerations~%float32 a_x~%float32 a_y~%float32 a_z~%float32 a_roll~%float32 a_pitch~%float32 a_yaw~%~%# Forces~%float32 f_x~%float32 f_y~%float32 f_z~%float32 f_roll~%float32 f_pitch~%float32 f_yaw~%~%# If set, override accelerations and use forces~%bool use_f_x~%bool use_f_y~%bool use_f_z~%bool use_f_roll~%bool use_f_pitch~%bool use_f_yaw~%~%# Setpoints~%float32 setpoint_z~%float32 setpoint_roll~%float32 setpoint_pitch~%~%# If set, override accelerations and forces and use setpoints~%bool use_setpoint_z~%bool use_setpoint_roll~%bool use_setpoint_pitch~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ControllerCommand>))
  (cl:+ 0
     4
     4
     4
     4
     4
     4
     4
     4
     4
     4
     4
     4
     1
     1
     1
     1
     1
     1
     4
     4
     4
     1
     1
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ControllerCommand>))
  "Converts a ROS message object to a list"
  (cl:list 'ControllerCommand
    (cl:cons ':a_x (a_x msg))
    (cl:cons ':a_y (a_y msg))
    (cl:cons ':a_z (a_z msg))
    (cl:cons ':a_roll (a_roll msg))
    (cl:cons ':a_pitch (a_pitch msg))
    (cl:cons ':a_yaw (a_yaw msg))
    (cl:cons ':f_x (f_x msg))
    (cl:cons ':f_y (f_y msg))
    (cl:cons ':f_z (f_z msg))
    (cl:cons ':f_roll (f_roll msg))
    (cl:cons ':f_pitch (f_pitch msg))
    (cl:cons ':f_yaw (f_yaw msg))
    (cl:cons ':use_f_x (use_f_x msg))
    (cl:cons ':use_f_y (use_f_y msg))
    (cl:cons ':use_f_z (use_f_z msg))
    (cl:cons ':use_f_roll (use_f_roll msg))
    (cl:cons ':use_f_pitch (use_f_pitch msg))
    (cl:cons ':use_f_yaw (use_f_yaw msg))
    (cl:cons ':setpoint_z (setpoint_z msg))
    (cl:cons ':setpoint_roll (setpoint_roll msg))
    (cl:cons ':setpoint_pitch (setpoint_pitch msg))
    (cl:cons ':use_setpoint_z (use_setpoint_z msg))
    (cl:cons ':use_setpoint_roll (use_setpoint_roll msg))
    (cl:cons ':use_setpoint_pitch (use_setpoint_pitch msg))
))
