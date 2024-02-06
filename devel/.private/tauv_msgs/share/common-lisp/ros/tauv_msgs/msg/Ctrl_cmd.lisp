; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude Ctrl_cmd.msg.html

(cl:defclass <Ctrl_cmd> (roslisp-msg-protocol:ros-message)
  ((Ctrl_vel_X
    :reader Ctrl_vel_X
    :initarg :Ctrl_vel_X
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Ctrl_vel_Y
    :reader Ctrl_vel_Y
    :initarg :Ctrl_vel_Y
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Ctrl_vel_Z
    :reader Ctrl_vel_Z
    :initarg :Ctrl_vel_Z
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Ctrl_fixed_Z
    :reader Ctrl_fixed_Z
    :initarg :Ctrl_fixed_Z
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Ctrl_vel_Rol
    :reader Ctrl_vel_Rol
    :initarg :Ctrl_vel_Rol
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Ctrl_vel_Pit
    :reader Ctrl_vel_Pit
    :initarg :Ctrl_vel_Pit
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Ctrl_vel_Yaw
    :reader Ctrl_vel_Yaw
    :initarg :Ctrl_vel_Yaw
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Ctrl_fixed_Yaw
    :reader Ctrl_fixed_Yaw
    :initarg :Ctrl_fixed_Yaw
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Ctrl_pivot_1
    :reader Ctrl_pivot_1
    :initarg :Ctrl_pivot_1
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Ctrl_pivot_2
    :reader Ctrl_pivot_2
    :initarg :Ctrl_pivot_2
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Ctrl_pivot_3
    :reader Ctrl_pivot_3
    :initarg :Ctrl_pivot_3
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Ctrl_pivot_4
    :reader Ctrl_pivot_4
    :initarg :Ctrl_pivot_4
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Ctrl_emagnet_1
    :reader Ctrl_emagnet_1
    :initarg :Ctrl_emagnet_1
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Ctrl_emagnet_2
    :reader Ctrl_emagnet_2
    :initarg :Ctrl_emagnet_2
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Ctrl_emagnet_3
    :reader Ctrl_emagnet_3
    :initarg :Ctrl_emagnet_3
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Ctrl_emagnet_4
    :reader Ctrl_emagnet_4
    :initarg :Ctrl_emagnet_4
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Ctrl_arm_joint_1
    :reader Ctrl_arm_joint_1
    :initarg :Ctrl_arm_joint_1
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Ctrl_arm_joint_2
    :reader Ctrl_arm_joint_2
    :initarg :Ctrl_arm_joint_2
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Joy_Button_Y
    :reader Joy_Button_Y
    :initarg :Joy_Button_Y
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Joy_Button_X
    :reader Joy_Button_X
    :initarg :Joy_Button_X
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Joy_Button_A
    :reader Joy_Button_A
    :initarg :Joy_Button_A
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Joy_Button_B
    :reader Joy_Button_B
    :initarg :Joy_Button_B
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Joy_Button_LB
    :reader Joy_Button_LB
    :initarg :Joy_Button_LB
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Joy_Button_RB
    :reader Joy_Button_RB
    :initarg :Joy_Button_RB
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Joy_Button_STICK_LEFT
    :reader Joy_Button_STICK_LEFT
    :initarg :Joy_Button_STICK_LEFT
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Joy_Button_STICK_RIGHT
    :reader Joy_Button_STICK_RIGHT
    :initarg :Joy_Button_STICK_RIGHT
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0))
   (Reset_pwm
    :reader Reset_pwm
    :initarg :Reset_pwm
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 4 :element-type 'cl:fixnum :initial-element 0)))
)

(cl:defclass Ctrl_cmd (<Ctrl_cmd>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Ctrl_cmd>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Ctrl_cmd)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<Ctrl_cmd> is deprecated: use tauv_msgs-msg:Ctrl_cmd instead.")))

(cl:ensure-generic-function 'Ctrl_vel_X-val :lambda-list '(m))
(cl:defmethod Ctrl_vel_X-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Ctrl_vel_X-val is deprecated.  Use tauv_msgs-msg:Ctrl_vel_X instead.")
  (Ctrl_vel_X m))

(cl:ensure-generic-function 'Ctrl_vel_Y-val :lambda-list '(m))
(cl:defmethod Ctrl_vel_Y-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Ctrl_vel_Y-val is deprecated.  Use tauv_msgs-msg:Ctrl_vel_Y instead.")
  (Ctrl_vel_Y m))

(cl:ensure-generic-function 'Ctrl_vel_Z-val :lambda-list '(m))
(cl:defmethod Ctrl_vel_Z-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Ctrl_vel_Z-val is deprecated.  Use tauv_msgs-msg:Ctrl_vel_Z instead.")
  (Ctrl_vel_Z m))

(cl:ensure-generic-function 'Ctrl_fixed_Z-val :lambda-list '(m))
(cl:defmethod Ctrl_fixed_Z-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Ctrl_fixed_Z-val is deprecated.  Use tauv_msgs-msg:Ctrl_fixed_Z instead.")
  (Ctrl_fixed_Z m))

(cl:ensure-generic-function 'Ctrl_vel_Rol-val :lambda-list '(m))
(cl:defmethod Ctrl_vel_Rol-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Ctrl_vel_Rol-val is deprecated.  Use tauv_msgs-msg:Ctrl_vel_Rol instead.")
  (Ctrl_vel_Rol m))

(cl:ensure-generic-function 'Ctrl_vel_Pit-val :lambda-list '(m))
(cl:defmethod Ctrl_vel_Pit-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Ctrl_vel_Pit-val is deprecated.  Use tauv_msgs-msg:Ctrl_vel_Pit instead.")
  (Ctrl_vel_Pit m))

(cl:ensure-generic-function 'Ctrl_vel_Yaw-val :lambda-list '(m))
(cl:defmethod Ctrl_vel_Yaw-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Ctrl_vel_Yaw-val is deprecated.  Use tauv_msgs-msg:Ctrl_vel_Yaw instead.")
  (Ctrl_vel_Yaw m))

(cl:ensure-generic-function 'Ctrl_fixed_Yaw-val :lambda-list '(m))
(cl:defmethod Ctrl_fixed_Yaw-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Ctrl_fixed_Yaw-val is deprecated.  Use tauv_msgs-msg:Ctrl_fixed_Yaw instead.")
  (Ctrl_fixed_Yaw m))

(cl:ensure-generic-function 'Ctrl_pivot_1-val :lambda-list '(m))
(cl:defmethod Ctrl_pivot_1-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Ctrl_pivot_1-val is deprecated.  Use tauv_msgs-msg:Ctrl_pivot_1 instead.")
  (Ctrl_pivot_1 m))

(cl:ensure-generic-function 'Ctrl_pivot_2-val :lambda-list '(m))
(cl:defmethod Ctrl_pivot_2-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Ctrl_pivot_2-val is deprecated.  Use tauv_msgs-msg:Ctrl_pivot_2 instead.")
  (Ctrl_pivot_2 m))

(cl:ensure-generic-function 'Ctrl_pivot_3-val :lambda-list '(m))
(cl:defmethod Ctrl_pivot_3-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Ctrl_pivot_3-val is deprecated.  Use tauv_msgs-msg:Ctrl_pivot_3 instead.")
  (Ctrl_pivot_3 m))

(cl:ensure-generic-function 'Ctrl_pivot_4-val :lambda-list '(m))
(cl:defmethod Ctrl_pivot_4-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Ctrl_pivot_4-val is deprecated.  Use tauv_msgs-msg:Ctrl_pivot_4 instead.")
  (Ctrl_pivot_4 m))

(cl:ensure-generic-function 'Ctrl_emagnet_1-val :lambda-list '(m))
(cl:defmethod Ctrl_emagnet_1-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Ctrl_emagnet_1-val is deprecated.  Use tauv_msgs-msg:Ctrl_emagnet_1 instead.")
  (Ctrl_emagnet_1 m))

(cl:ensure-generic-function 'Ctrl_emagnet_2-val :lambda-list '(m))
(cl:defmethod Ctrl_emagnet_2-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Ctrl_emagnet_2-val is deprecated.  Use tauv_msgs-msg:Ctrl_emagnet_2 instead.")
  (Ctrl_emagnet_2 m))

(cl:ensure-generic-function 'Ctrl_emagnet_3-val :lambda-list '(m))
(cl:defmethod Ctrl_emagnet_3-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Ctrl_emagnet_3-val is deprecated.  Use tauv_msgs-msg:Ctrl_emagnet_3 instead.")
  (Ctrl_emagnet_3 m))

(cl:ensure-generic-function 'Ctrl_emagnet_4-val :lambda-list '(m))
(cl:defmethod Ctrl_emagnet_4-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Ctrl_emagnet_4-val is deprecated.  Use tauv_msgs-msg:Ctrl_emagnet_4 instead.")
  (Ctrl_emagnet_4 m))

(cl:ensure-generic-function 'Ctrl_arm_joint_1-val :lambda-list '(m))
(cl:defmethod Ctrl_arm_joint_1-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Ctrl_arm_joint_1-val is deprecated.  Use tauv_msgs-msg:Ctrl_arm_joint_1 instead.")
  (Ctrl_arm_joint_1 m))

(cl:ensure-generic-function 'Ctrl_arm_joint_2-val :lambda-list '(m))
(cl:defmethod Ctrl_arm_joint_2-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Ctrl_arm_joint_2-val is deprecated.  Use tauv_msgs-msg:Ctrl_arm_joint_2 instead.")
  (Ctrl_arm_joint_2 m))

(cl:ensure-generic-function 'Joy_Button_Y-val :lambda-list '(m))
(cl:defmethod Joy_Button_Y-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Joy_Button_Y-val is deprecated.  Use tauv_msgs-msg:Joy_Button_Y instead.")
  (Joy_Button_Y m))

(cl:ensure-generic-function 'Joy_Button_X-val :lambda-list '(m))
(cl:defmethod Joy_Button_X-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Joy_Button_X-val is deprecated.  Use tauv_msgs-msg:Joy_Button_X instead.")
  (Joy_Button_X m))

(cl:ensure-generic-function 'Joy_Button_A-val :lambda-list '(m))
(cl:defmethod Joy_Button_A-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Joy_Button_A-val is deprecated.  Use tauv_msgs-msg:Joy_Button_A instead.")
  (Joy_Button_A m))

(cl:ensure-generic-function 'Joy_Button_B-val :lambda-list '(m))
(cl:defmethod Joy_Button_B-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Joy_Button_B-val is deprecated.  Use tauv_msgs-msg:Joy_Button_B instead.")
  (Joy_Button_B m))

(cl:ensure-generic-function 'Joy_Button_LB-val :lambda-list '(m))
(cl:defmethod Joy_Button_LB-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Joy_Button_LB-val is deprecated.  Use tauv_msgs-msg:Joy_Button_LB instead.")
  (Joy_Button_LB m))

(cl:ensure-generic-function 'Joy_Button_RB-val :lambda-list '(m))
(cl:defmethod Joy_Button_RB-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Joy_Button_RB-val is deprecated.  Use tauv_msgs-msg:Joy_Button_RB instead.")
  (Joy_Button_RB m))

(cl:ensure-generic-function 'Joy_Button_STICK_LEFT-val :lambda-list '(m))
(cl:defmethod Joy_Button_STICK_LEFT-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Joy_Button_STICK_LEFT-val is deprecated.  Use tauv_msgs-msg:Joy_Button_STICK_LEFT instead.")
  (Joy_Button_STICK_LEFT m))

(cl:ensure-generic-function 'Joy_Button_STICK_RIGHT-val :lambda-list '(m))
(cl:defmethod Joy_Button_STICK_RIGHT-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Joy_Button_STICK_RIGHT-val is deprecated.  Use tauv_msgs-msg:Joy_Button_STICK_RIGHT instead.")
  (Joy_Button_STICK_RIGHT m))

(cl:ensure-generic-function 'Reset_pwm-val :lambda-list '(m))
(cl:defmethod Reset_pwm-val ((m <Ctrl_cmd>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:Reset_pwm-val is deprecated.  Use tauv_msgs-msg:Reset_pwm instead.")
  (Reset_pwm m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Ctrl_cmd>) ostream)
  "Serializes a message object of type '<Ctrl_cmd>"
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'Ctrl_vel_X))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'Ctrl_vel_Y))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'Ctrl_vel_Z))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'Ctrl_fixed_Z))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'Ctrl_vel_Rol))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'Ctrl_vel_Pit))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'Ctrl_vel_Yaw))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'Ctrl_fixed_Yaw))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'Ctrl_pivot_1))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'Ctrl_pivot_2))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'Ctrl_pivot_3))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'Ctrl_pivot_4))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'Ctrl_emagnet_1))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'Ctrl_emagnet_2))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'Ctrl_emagnet_3))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'Ctrl_emagnet_4))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'Ctrl_arm_joint_1))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'Ctrl_arm_joint_2))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'Joy_Button_Y))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'Joy_Button_X))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'Joy_Button_A))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'Joy_Button_B))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'Joy_Button_LB))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'Joy_Button_RB))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'Joy_Button_STICK_LEFT))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'Joy_Button_STICK_RIGHT))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    ))
   (cl:slot-value msg 'Reset_pwm))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Ctrl_cmd>) istream)
  "Deserializes a message object of type '<Ctrl_cmd>"
  (cl:setf (cl:slot-value msg 'Ctrl_vel_X) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Ctrl_vel_X)))
    (cl:dotimes (i 4)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Ctrl_vel_Y) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Ctrl_vel_Y)))
    (cl:dotimes (i 4)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Ctrl_vel_Z) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Ctrl_vel_Z)))
    (cl:dotimes (i 4)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Ctrl_fixed_Z) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Ctrl_fixed_Z)))
    (cl:dotimes (i 4)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Ctrl_vel_Rol) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Ctrl_vel_Rol)))
    (cl:dotimes (i 4)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Ctrl_vel_Pit) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Ctrl_vel_Pit)))
    (cl:dotimes (i 4)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Ctrl_vel_Yaw) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Ctrl_vel_Yaw)))
    (cl:dotimes (i 4)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Ctrl_fixed_Yaw) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Ctrl_fixed_Yaw)))
    (cl:dotimes (i 4)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Ctrl_pivot_1) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Ctrl_pivot_1)))
    (cl:dotimes (i 4)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Ctrl_pivot_2) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Ctrl_pivot_2)))
    (cl:dotimes (i 4)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Ctrl_pivot_3) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Ctrl_pivot_3)))
    (cl:dotimes (i 4)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Ctrl_pivot_4) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Ctrl_pivot_4)))
    (cl:dotimes (i 4)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Ctrl_emagnet_1) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Ctrl_emagnet_1)))
    (cl:dotimes (i 4)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Ctrl_emagnet_2) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Ctrl_emagnet_2)))
    (cl:dotimes (i 4)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Ctrl_emagnet_3) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Ctrl_emagnet_3)))
    (cl:dotimes (i 4)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Ctrl_emagnet_4) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Ctrl_emagnet_4)))
    (cl:dotimes (i 4)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Ctrl_arm_joint_1) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Ctrl_arm_joint_1)))
    (cl:dotimes (i 4)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Ctrl_arm_joint_2) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Ctrl_arm_joint_2)))
    (cl:dotimes (i 4)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Joy_Button_Y) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Joy_Button_Y)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  (cl:setf (cl:slot-value msg 'Joy_Button_X) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Joy_Button_X)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  (cl:setf (cl:slot-value msg 'Joy_Button_A) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Joy_Button_A)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  (cl:setf (cl:slot-value msg 'Joy_Button_B) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Joy_Button_B)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  (cl:setf (cl:slot-value msg 'Joy_Button_LB) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Joy_Button_LB)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  (cl:setf (cl:slot-value msg 'Joy_Button_RB) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Joy_Button_RB)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  (cl:setf (cl:slot-value msg 'Joy_Button_STICK_LEFT) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Joy_Button_STICK_LEFT)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  (cl:setf (cl:slot-value msg 'Joy_Button_STICK_RIGHT) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Joy_Button_STICK_RIGHT)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  (cl:setf (cl:slot-value msg 'Reset_pwm) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'Reset_pwm)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Ctrl_cmd>)))
  "Returns string type for a message object of type '<Ctrl_cmd>"
  "tauv_msgs/Ctrl_cmd")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Ctrl_cmd)))
  "Returns string type for a message object of type 'Ctrl_cmd"
  "tauv_msgs/Ctrl_cmd")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Ctrl_cmd>)))
  "Returns md5sum for a message object of type '<Ctrl_cmd>"
  "7a746ffc64e1d5a26fb5d6cba04e63c2")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Ctrl_cmd)))
  "Returns md5sum for a message object of type 'Ctrl_cmd"
  "7a746ffc64e1d5a26fb5d6cba04e63c2")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Ctrl_cmd>)))
  "Returns full string definition for message of type '<Ctrl_cmd>"
  (cl:format cl:nil "~%uint8[4] Ctrl_vel_X~%uint8[4] Ctrl_vel_Y~%uint8[4] Ctrl_vel_Z~%uint8[4] Ctrl_fixed_Z~%uint8[4] Ctrl_vel_Rol~%uint8[4] Ctrl_vel_Pit~%uint8[4] Ctrl_vel_Yaw~%uint8[4] Ctrl_fixed_Yaw~%~%uint8[4] Ctrl_pivot_1~%uint8[4] Ctrl_pivot_2~%uint8[4] Ctrl_pivot_3~%uint8[4] Ctrl_pivot_4~%~%uint8[4] Ctrl_emagnet_1~%uint8[4] Ctrl_emagnet_2~%uint8[4] Ctrl_emagnet_3~%uint8[4] Ctrl_emagnet_4~%~%uint8[4] Ctrl_arm_joint_1~%uint8[4] Ctrl_arm_joint_2~%~%int16[4] Joy_Button_Y~%int16[4] Joy_Button_X~%int16[4] Joy_Button_A~%int16[4] Joy_Button_B~%int16[4] Joy_Button_LB~%int16[4] Joy_Button_RB~%int16[4] Joy_Button_STICK_LEFT~%int16[4] Joy_Button_STICK_RIGHT~%int16[4] Reset_pwm~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Ctrl_cmd)))
  "Returns full string definition for message of type 'Ctrl_cmd"
  (cl:format cl:nil "~%uint8[4] Ctrl_vel_X~%uint8[4] Ctrl_vel_Y~%uint8[4] Ctrl_vel_Z~%uint8[4] Ctrl_fixed_Z~%uint8[4] Ctrl_vel_Rol~%uint8[4] Ctrl_vel_Pit~%uint8[4] Ctrl_vel_Yaw~%uint8[4] Ctrl_fixed_Yaw~%~%uint8[4] Ctrl_pivot_1~%uint8[4] Ctrl_pivot_2~%uint8[4] Ctrl_pivot_3~%uint8[4] Ctrl_pivot_4~%~%uint8[4] Ctrl_emagnet_1~%uint8[4] Ctrl_emagnet_2~%uint8[4] Ctrl_emagnet_3~%uint8[4] Ctrl_emagnet_4~%~%uint8[4] Ctrl_arm_joint_1~%uint8[4] Ctrl_arm_joint_2~%~%int16[4] Joy_Button_Y~%int16[4] Joy_Button_X~%int16[4] Joy_Button_A~%int16[4] Joy_Button_B~%int16[4] Joy_Button_LB~%int16[4] Joy_Button_RB~%int16[4] Joy_Button_STICK_LEFT~%int16[4] Joy_Button_STICK_RIGHT~%int16[4] Reset_pwm~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Ctrl_cmd>))
  (cl:+ 0
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Ctrl_vel_X) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Ctrl_vel_Y) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Ctrl_vel_Z) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Ctrl_fixed_Z) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Ctrl_vel_Rol) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Ctrl_vel_Pit) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Ctrl_vel_Yaw) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Ctrl_fixed_Yaw) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Ctrl_pivot_1) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Ctrl_pivot_2) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Ctrl_pivot_3) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Ctrl_pivot_4) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Ctrl_emagnet_1) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Ctrl_emagnet_2) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Ctrl_emagnet_3) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Ctrl_emagnet_4) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Ctrl_arm_joint_1) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Ctrl_arm_joint_2) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Joy_Button_Y) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Joy_Button_X) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Joy_Button_A) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Joy_Button_B) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Joy_Button_LB) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Joy_Button_RB) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Joy_Button_STICK_LEFT) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Joy_Button_STICK_RIGHT) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Reset_pwm) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Ctrl_cmd>))
  "Converts a ROS message object to a list"
  (cl:list 'Ctrl_cmd
    (cl:cons ':Ctrl_vel_X (Ctrl_vel_X msg))
    (cl:cons ':Ctrl_vel_Y (Ctrl_vel_Y msg))
    (cl:cons ':Ctrl_vel_Z (Ctrl_vel_Z msg))
    (cl:cons ':Ctrl_fixed_Z (Ctrl_fixed_Z msg))
    (cl:cons ':Ctrl_vel_Rol (Ctrl_vel_Rol msg))
    (cl:cons ':Ctrl_vel_Pit (Ctrl_vel_Pit msg))
    (cl:cons ':Ctrl_vel_Yaw (Ctrl_vel_Yaw msg))
    (cl:cons ':Ctrl_fixed_Yaw (Ctrl_fixed_Yaw msg))
    (cl:cons ':Ctrl_pivot_1 (Ctrl_pivot_1 msg))
    (cl:cons ':Ctrl_pivot_2 (Ctrl_pivot_2 msg))
    (cl:cons ':Ctrl_pivot_3 (Ctrl_pivot_3 msg))
    (cl:cons ':Ctrl_pivot_4 (Ctrl_pivot_4 msg))
    (cl:cons ':Ctrl_emagnet_1 (Ctrl_emagnet_1 msg))
    (cl:cons ':Ctrl_emagnet_2 (Ctrl_emagnet_2 msg))
    (cl:cons ':Ctrl_emagnet_3 (Ctrl_emagnet_3 msg))
    (cl:cons ':Ctrl_emagnet_4 (Ctrl_emagnet_4 msg))
    (cl:cons ':Ctrl_arm_joint_1 (Ctrl_arm_joint_1 msg))
    (cl:cons ':Ctrl_arm_joint_2 (Ctrl_arm_joint_2 msg))
    (cl:cons ':Joy_Button_Y (Joy_Button_Y msg))
    (cl:cons ':Joy_Button_X (Joy_Button_X msg))
    (cl:cons ':Joy_Button_A (Joy_Button_A msg))
    (cl:cons ':Joy_Button_B (Joy_Button_B msg))
    (cl:cons ':Joy_Button_LB (Joy_Button_LB msg))
    (cl:cons ':Joy_Button_RB (Joy_Button_RB msg))
    (cl:cons ':Joy_Button_STICK_LEFT (Joy_Button_STICK_LEFT msg))
    (cl:cons ':Joy_Button_STICK_RIGHT (Joy_Button_STICK_RIGHT msg))
    (cl:cons ':Reset_pwm (Reset_pwm msg))
))
