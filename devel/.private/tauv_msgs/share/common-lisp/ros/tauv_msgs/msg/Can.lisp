; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude Can.msg.html

(cl:defclass <Can> (roslisp-msg-protocol:ros-message)
  ((voltage_current
    :reader voltage_current
    :initarg :voltage_current
    :type cl:fixnum
    :initform 0)
   (temperature_cabin_current
    :reader temperature_cabin_current
    :initarg :temperature_cabin_current
    :type cl:fixnum
    :initform 0)
   (rm_speed_rpm
    :reader rm_speed_rpm
    :initarg :rm_speed_rpm
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (rm_given_current
    :reader rm_given_current
    :initarg :rm_given_current
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (rm_total_angle
    :reader rm_total_angle
    :initarg :rm_total_angle
    :type (cl:vector cl:integer)
   :initform (cl:make-array 4 :element-type 'cl:integer :initial-element 0))
   (FB_auv_pit
    :reader FB_auv_pit
    :initarg :FB_auv_pit
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (FB_auv_rol
    :reader FB_auv_rol
    :initarg :FB_auv_rol
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (FB_auv_yaw
    :reader FB_auv_yaw
    :initarg :FB_auv_yaw
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (FB_auv_deep
    :reader FB_auv_deep
    :initarg :FB_auv_deep
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (FB_auv_deep_vel
    :reader FB_auv_deep_vel
    :initarg :FB_auv_deep_vel
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (FB_auv_ang_vel_pit
    :reader FB_auv_ang_vel_pit
    :initarg :FB_auv_ang_vel_pit
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (FB_auv_ang_vel_rol
    :reader FB_auv_ang_vel_rol
    :initarg :FB_auv_ang_vel_rol
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (FB_auv_ang_vel_yaw
    :reader FB_auv_ang_vel_yaw
    :initarg :FB_auv_ang_vel_yaw
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (PWM_set
    :reader PWM_set
    :initarg :PWM_set
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 8 :element-type 'cl:fixnum :initial-element 0)))
)

(cl:defclass Can (<Can>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Can>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Can)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<Can> is deprecated: use tauv_msgs-msg:Can instead.")))

(cl:ensure-generic-function 'voltage_current-val :lambda-list '(m))
(cl:defmethod voltage_current-val ((m <Can>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:voltage_current-val is deprecated.  Use tauv_msgs-msg:voltage_current instead.")
  (voltage_current m))

(cl:ensure-generic-function 'temperature_cabin_current-val :lambda-list '(m))
(cl:defmethod temperature_cabin_current-val ((m <Can>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:temperature_cabin_current-val is deprecated.  Use tauv_msgs-msg:temperature_cabin_current instead.")
  (temperature_cabin_current m))

(cl:ensure-generic-function 'rm_speed_rpm-val :lambda-list '(m))
(cl:defmethod rm_speed_rpm-val ((m <Can>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:rm_speed_rpm-val is deprecated.  Use tauv_msgs-msg:rm_speed_rpm instead.")
  (rm_speed_rpm m))

(cl:ensure-generic-function 'rm_given_current-val :lambda-list '(m))
(cl:defmethod rm_given_current-val ((m <Can>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:rm_given_current-val is deprecated.  Use tauv_msgs-msg:rm_given_current instead.")
  (rm_given_current m))

(cl:ensure-generic-function 'rm_total_angle-val :lambda-list '(m))
(cl:defmethod rm_total_angle-val ((m <Can>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:rm_total_angle-val is deprecated.  Use tauv_msgs-msg:rm_total_angle instead.")
  (rm_total_angle m))

(cl:ensure-generic-function 'FB_auv_pit-val :lambda-list '(m))
(cl:defmethod FB_auv_pit-val ((m <Can>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:FB_auv_pit-val is deprecated.  Use tauv_msgs-msg:FB_auv_pit instead.")
  (FB_auv_pit m))

(cl:ensure-generic-function 'FB_auv_rol-val :lambda-list '(m))
(cl:defmethod FB_auv_rol-val ((m <Can>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:FB_auv_rol-val is deprecated.  Use tauv_msgs-msg:FB_auv_rol instead.")
  (FB_auv_rol m))

(cl:ensure-generic-function 'FB_auv_yaw-val :lambda-list '(m))
(cl:defmethod FB_auv_yaw-val ((m <Can>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:FB_auv_yaw-val is deprecated.  Use tauv_msgs-msg:FB_auv_yaw instead.")
  (FB_auv_yaw m))

(cl:ensure-generic-function 'FB_auv_deep-val :lambda-list '(m))
(cl:defmethod FB_auv_deep-val ((m <Can>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:FB_auv_deep-val is deprecated.  Use tauv_msgs-msg:FB_auv_deep instead.")
  (FB_auv_deep m))

(cl:ensure-generic-function 'FB_auv_deep_vel-val :lambda-list '(m))
(cl:defmethod FB_auv_deep_vel-val ((m <Can>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:FB_auv_deep_vel-val is deprecated.  Use tauv_msgs-msg:FB_auv_deep_vel instead.")
  (FB_auv_deep_vel m))

(cl:ensure-generic-function 'FB_auv_ang_vel_pit-val :lambda-list '(m))
(cl:defmethod FB_auv_ang_vel_pit-val ((m <Can>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:FB_auv_ang_vel_pit-val is deprecated.  Use tauv_msgs-msg:FB_auv_ang_vel_pit instead.")
  (FB_auv_ang_vel_pit m))

(cl:ensure-generic-function 'FB_auv_ang_vel_rol-val :lambda-list '(m))
(cl:defmethod FB_auv_ang_vel_rol-val ((m <Can>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:FB_auv_ang_vel_rol-val is deprecated.  Use tauv_msgs-msg:FB_auv_ang_vel_rol instead.")
  (FB_auv_ang_vel_rol m))

(cl:ensure-generic-function 'FB_auv_ang_vel_yaw-val :lambda-list '(m))
(cl:defmethod FB_auv_ang_vel_yaw-val ((m <Can>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:FB_auv_ang_vel_yaw-val is deprecated.  Use tauv_msgs-msg:FB_auv_ang_vel_yaw instead.")
  (FB_auv_ang_vel_yaw m))

(cl:ensure-generic-function 'PWM_set-val :lambda-list '(m))
(cl:defmethod PWM_set-val ((m <Can>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:PWM_set-val is deprecated.  Use tauv_msgs-msg:PWM_set instead.")
  (PWM_set m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Can>) ostream)
  "Serializes a message object of type '<Can>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'voltage_current)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'temperature_cabin_current)) ostream)
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'rm_speed_rpm))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'rm_given_current))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    ))
   (cl:slot-value msg 'rm_total_angle))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'FB_auv_pit))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'FB_auv_rol))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'FB_auv_yaw))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'FB_auv_deep))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'FB_auv_deep_vel))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'FB_auv_ang_vel_pit))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'FB_auv_ang_vel_rol))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'FB_auv_ang_vel_yaw))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'PWM_set))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Can>) istream)
  "Deserializes a message object of type '<Can>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'voltage_current)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'temperature_cabin_current)) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'rm_speed_rpm) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'rm_speed_rpm)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  (cl:setf (cl:slot-value msg 'rm_given_current) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'rm_given_current)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  (cl:setf (cl:slot-value msg 'rm_total_angle) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'rm_total_angle)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))))
  (cl:setf (cl:slot-value msg 'FB_auv_pit) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'FB_auv_pit)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  (cl:setf (cl:slot-value msg 'FB_auv_rol) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'FB_auv_rol)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  (cl:setf (cl:slot-value msg 'FB_auv_yaw) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'FB_auv_yaw)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  (cl:setf (cl:slot-value msg 'FB_auv_deep) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'FB_auv_deep)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  (cl:setf (cl:slot-value msg 'FB_auv_deep_vel) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'FB_auv_deep_vel)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  (cl:setf (cl:slot-value msg 'FB_auv_ang_vel_pit) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'FB_auv_ang_vel_pit)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  (cl:setf (cl:slot-value msg 'FB_auv_ang_vel_rol) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'FB_auv_ang_vel_rol)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  (cl:setf (cl:slot-value msg 'FB_auv_ang_vel_yaw) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'FB_auv_ang_vel_yaw)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  (cl:setf (cl:slot-value msg 'PWM_set) (cl:make-array 8))
  (cl:let ((vals (cl:slot-value msg 'PWM_set)))
    (cl:dotimes (i 8)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Can>)))
  "Returns string type for a message object of type '<Can>"
  "tauv_msgs/Can")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Can)))
  "Returns string type for a message object of type 'Can"
  "tauv_msgs/Can")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Can>)))
  "Returns md5sum for a message object of type '<Can>"
  "38928799346a1e1fb02c8f1ad6011cc1")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Can)))
  "Returns md5sum for a message object of type 'Can"
  "38928799346a1e1fb02c8f1ad6011cc1")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Can>)))
  "Returns full string definition for message of type '<Can>"
  (cl:format cl:nil "uint8 voltage_current~%uint8 temperature_cabin_current~%~%int16[4] rm_speed_rpm~%int16[4] rm_given_current~%int32[4] rm_total_angle~%~%int16[4] FB_auv_pit~%int16[4] FB_auv_rol~%int16[4] FB_auv_yaw~%int16[4] FB_auv_deep~%int16[4] FB_auv_deep_vel~%int16[4] FB_auv_ang_vel_pit~%int16[4] FB_auv_ang_vel_rol~%int16[4] FB_auv_ang_vel_yaw~%~%int16[8] PWM_set~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Can)))
  "Returns full string definition for message of type 'Can"
  (cl:format cl:nil "uint8 voltage_current~%uint8 temperature_cabin_current~%~%int16[4] rm_speed_rpm~%int16[4] rm_given_current~%int32[4] rm_total_angle~%~%int16[4] FB_auv_pit~%int16[4] FB_auv_rol~%int16[4] FB_auv_yaw~%int16[4] FB_auv_deep~%int16[4] FB_auv_deep_vel~%int16[4] FB_auv_ang_vel_pit~%int16[4] FB_auv_ang_vel_rol~%int16[4] FB_auv_ang_vel_yaw~%~%int16[8] PWM_set~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Can>))
  (cl:+ 0
     1
     1
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'rm_speed_rpm) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'rm_given_current) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'rm_total_angle) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'FB_auv_pit) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'FB_auv_rol) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'FB_auv_yaw) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'FB_auv_deep) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'FB_auv_deep_vel) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'FB_auv_ang_vel_pit) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'FB_auv_ang_vel_rol) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'FB_auv_ang_vel_yaw) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'PWM_set) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Can>))
  "Converts a ROS message object to a list"
  (cl:list 'Can
    (cl:cons ':voltage_current (voltage_current msg))
    (cl:cons ':temperature_cabin_current (temperature_cabin_current msg))
    (cl:cons ':rm_speed_rpm (rm_speed_rpm msg))
    (cl:cons ':rm_given_current (rm_given_current msg))
    (cl:cons ':rm_total_angle (rm_total_angle msg))
    (cl:cons ':FB_auv_pit (FB_auv_pit msg))
    (cl:cons ':FB_auv_rol (FB_auv_rol msg))
    (cl:cons ':FB_auv_yaw (FB_auv_yaw msg))
    (cl:cons ':FB_auv_deep (FB_auv_deep msg))
    (cl:cons ':FB_auv_deep_vel (FB_auv_deep_vel msg))
    (cl:cons ':FB_auv_ang_vel_pit (FB_auv_ang_vel_pit msg))
    (cl:cons ':FB_auv_ang_vel_rol (FB_auv_ang_vel_rol msg))
    (cl:cons ':FB_auv_ang_vel_yaw (FB_auv_ang_vel_yaw msg))
    (cl:cons ':PWM_set (PWM_set msg))
))
