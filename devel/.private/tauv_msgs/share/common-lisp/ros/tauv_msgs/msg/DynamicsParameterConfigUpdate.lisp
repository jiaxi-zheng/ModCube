; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude DynamicsParameterConfigUpdate.msg.html

(cl:defclass <DynamicsParameterConfigUpdate> (roslisp-msg-protocol:ros-message)
  ((name
    :reader name
    :initarg :name
    :type cl:string
    :initform "")
   (update_initial_value
    :reader update_initial_value
    :initarg :update_initial_value
    :type cl:boolean
    :initform cl:nil)
   (initial_value
    :reader initial_value
    :initarg :initial_value
    :type cl:float
    :initform 0.0)
   (update_fixed
    :reader update_fixed
    :initarg :update_fixed
    :type cl:boolean
    :initform cl:nil)
   (fixed
    :reader fixed
    :initarg :fixed
    :type cl:boolean
    :initform cl:nil)
   (update_initial_covariance
    :reader update_initial_covariance
    :initarg :update_initial_covariance
    :type cl:boolean
    :initform cl:nil)
   (initial_covariance
    :reader initial_covariance
    :initarg :initial_covariance
    :type cl:float
    :initform 0.0)
   (update_process_covariance
    :reader update_process_covariance
    :initarg :update_process_covariance
    :type cl:boolean
    :initform cl:nil)
   (process_covariance
    :reader process_covariance
    :initarg :process_covariance
    :type cl:float
    :initform 0.0)
   (update_limits
    :reader update_limits
    :initarg :update_limits
    :type cl:boolean
    :initform cl:nil)
   (limits
    :reader limits
    :initarg :limits
    :type (cl:vector cl:float)
   :initform (cl:make-array 2 :element-type 'cl:float :initial-element 0.0))
   (reset
    :reader reset
    :initarg :reset
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass DynamicsParameterConfigUpdate (<DynamicsParameterConfigUpdate>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <DynamicsParameterConfigUpdate>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'DynamicsParameterConfigUpdate)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<DynamicsParameterConfigUpdate> is deprecated: use tauv_msgs-msg:DynamicsParameterConfigUpdate instead.")))

(cl:ensure-generic-function 'name-val :lambda-list '(m))
(cl:defmethod name-val ((m <DynamicsParameterConfigUpdate>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:name-val is deprecated.  Use tauv_msgs-msg:name instead.")
  (name m))

(cl:ensure-generic-function 'update_initial_value-val :lambda-list '(m))
(cl:defmethod update_initial_value-val ((m <DynamicsParameterConfigUpdate>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:update_initial_value-val is deprecated.  Use tauv_msgs-msg:update_initial_value instead.")
  (update_initial_value m))

(cl:ensure-generic-function 'initial_value-val :lambda-list '(m))
(cl:defmethod initial_value-val ((m <DynamicsParameterConfigUpdate>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:initial_value-val is deprecated.  Use tauv_msgs-msg:initial_value instead.")
  (initial_value m))

(cl:ensure-generic-function 'update_fixed-val :lambda-list '(m))
(cl:defmethod update_fixed-val ((m <DynamicsParameterConfigUpdate>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:update_fixed-val is deprecated.  Use tauv_msgs-msg:update_fixed instead.")
  (update_fixed m))

(cl:ensure-generic-function 'fixed-val :lambda-list '(m))
(cl:defmethod fixed-val ((m <DynamicsParameterConfigUpdate>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:fixed-val is deprecated.  Use tauv_msgs-msg:fixed instead.")
  (fixed m))

(cl:ensure-generic-function 'update_initial_covariance-val :lambda-list '(m))
(cl:defmethod update_initial_covariance-val ((m <DynamicsParameterConfigUpdate>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:update_initial_covariance-val is deprecated.  Use tauv_msgs-msg:update_initial_covariance instead.")
  (update_initial_covariance m))

(cl:ensure-generic-function 'initial_covariance-val :lambda-list '(m))
(cl:defmethod initial_covariance-val ((m <DynamicsParameterConfigUpdate>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:initial_covariance-val is deprecated.  Use tauv_msgs-msg:initial_covariance instead.")
  (initial_covariance m))

(cl:ensure-generic-function 'update_process_covariance-val :lambda-list '(m))
(cl:defmethod update_process_covariance-val ((m <DynamicsParameterConfigUpdate>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:update_process_covariance-val is deprecated.  Use tauv_msgs-msg:update_process_covariance instead.")
  (update_process_covariance m))

(cl:ensure-generic-function 'process_covariance-val :lambda-list '(m))
(cl:defmethod process_covariance-val ((m <DynamicsParameterConfigUpdate>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:process_covariance-val is deprecated.  Use tauv_msgs-msg:process_covariance instead.")
  (process_covariance m))

(cl:ensure-generic-function 'update_limits-val :lambda-list '(m))
(cl:defmethod update_limits-val ((m <DynamicsParameterConfigUpdate>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:update_limits-val is deprecated.  Use tauv_msgs-msg:update_limits instead.")
  (update_limits m))

(cl:ensure-generic-function 'limits-val :lambda-list '(m))
(cl:defmethod limits-val ((m <DynamicsParameterConfigUpdate>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:limits-val is deprecated.  Use tauv_msgs-msg:limits instead.")
  (limits m))

(cl:ensure-generic-function 'reset-val :lambda-list '(m))
(cl:defmethod reset-val ((m <DynamicsParameterConfigUpdate>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:reset-val is deprecated.  Use tauv_msgs-msg:reset instead.")
  (reset m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <DynamicsParameterConfigUpdate>) ostream)
  "Serializes a message object of type '<DynamicsParameterConfigUpdate>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'name))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'update_initial_value) 1 0)) ostream)
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'initial_value))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'update_fixed) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'fixed) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'update_initial_covariance) 1 0)) ostream)
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'initial_covariance))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'update_process_covariance) 1 0)) ostream)
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'process_covariance))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'update_limits) 1 0)) ostream)
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'limits))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'reset) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <DynamicsParameterConfigUpdate>) istream)
  "Deserializes a message object of type '<DynamicsParameterConfigUpdate>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:setf (cl:slot-value msg 'update_initial_value) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'initial_value) (roslisp-utils:decode-double-float-bits bits)))
    (cl:setf (cl:slot-value msg 'update_fixed) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'fixed) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'update_initial_covariance) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'initial_covariance) (roslisp-utils:decode-double-float-bits bits)))
    (cl:setf (cl:slot-value msg 'update_process_covariance) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'process_covariance) (roslisp-utils:decode-double-float-bits bits)))
    (cl:setf (cl:slot-value msg 'update_limits) (cl:not (cl:zerop (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'limits) (cl:make-array 2))
  (cl:let ((vals (cl:slot-value msg 'limits)))
    (cl:dotimes (i 2)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-double-float-bits bits)))))
    (cl:setf (cl:slot-value msg 'reset) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<DynamicsParameterConfigUpdate>)))
  "Returns string type for a message object of type '<DynamicsParameterConfigUpdate>"
  "tauv_msgs/DynamicsParameterConfigUpdate")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DynamicsParameterConfigUpdate)))
  "Returns string type for a message object of type 'DynamicsParameterConfigUpdate"
  "tauv_msgs/DynamicsParameterConfigUpdate")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<DynamicsParameterConfigUpdate>)))
  "Returns md5sum for a message object of type '<DynamicsParameterConfigUpdate>"
  "31294fd8c67bf91ba516e5502711b385")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'DynamicsParameterConfigUpdate)))
  "Returns md5sum for a message object of type 'DynamicsParameterConfigUpdate"
  "31294fd8c67bf91ba516e5502711b385")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<DynamicsParameterConfigUpdate>)))
  "Returns full string definition for message of type '<DynamicsParameterConfigUpdate>"
  (cl:format cl:nil "string name~%~%bool update_initial_value~%float64 initial_value~%~%bool update_fixed~%bool fixed~%~%bool update_initial_covariance~%float64 initial_covariance~%~%bool update_process_covariance~%float64 process_covariance~%~%bool update_limits~%float64[2] limits~%~%bool reset~%~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'DynamicsParameterConfigUpdate)))
  "Returns full string definition for message of type 'DynamicsParameterConfigUpdate"
  (cl:format cl:nil "string name~%~%bool update_initial_value~%float64 initial_value~%~%bool update_fixed~%bool fixed~%~%bool update_initial_covariance~%float64 initial_covariance~%~%bool update_process_covariance~%float64 process_covariance~%~%bool update_limits~%float64[2] limits~%~%bool reset~%~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <DynamicsParameterConfigUpdate>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'name))
     1
     8
     1
     1
     1
     8
     1
     8
     1
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'limits) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <DynamicsParameterConfigUpdate>))
  "Converts a ROS message object to a list"
  (cl:list 'DynamicsParameterConfigUpdate
    (cl:cons ':name (name msg))
    (cl:cons ':update_initial_value (update_initial_value msg))
    (cl:cons ':initial_value (initial_value msg))
    (cl:cons ':update_fixed (update_fixed msg))
    (cl:cons ':fixed (fixed msg))
    (cl:cons ':update_initial_covariance (update_initial_covariance msg))
    (cl:cons ':initial_covariance (initial_covariance msg))
    (cl:cons ':update_process_covariance (update_process_covariance msg))
    (cl:cons ':process_covariance (process_covariance msg))
    (cl:cons ':update_limits (update_limits msg))
    (cl:cons ':limits (limits msg))
    (cl:cons ':reset (reset msg))
))
