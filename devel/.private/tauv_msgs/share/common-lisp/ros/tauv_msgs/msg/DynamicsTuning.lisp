; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude DynamicsTuning.msg.html

(cl:defclass <DynamicsTuning> (roslisp-msg-protocol:ros-message)
  ((update_mass
    :reader update_mass
    :initarg :update_mass
    :type cl:boolean
    :initform cl:nil)
   (mass
    :reader mass
    :initarg :mass
    :type cl:float
    :initform 0.0)
   (update_volume
    :reader update_volume
    :initarg :update_volume
    :type cl:boolean
    :initform cl:nil)
   (volume
    :reader volume
    :initarg :volume
    :type cl:float
    :initform 0.0)
   (update_water_density
    :reader update_water_density
    :initarg :update_water_density
    :type cl:boolean
    :initform cl:nil)
   (water_density
    :reader water_density
    :initarg :water_density
    :type cl:float
    :initform 0.0)
   (update_center_of_gravity
    :reader update_center_of_gravity
    :initarg :update_center_of_gravity
    :type cl:boolean
    :initform cl:nil)
   (center_of_gravity
    :reader center_of_gravity
    :initarg :center_of_gravity
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (update_center_of_buoyancy
    :reader update_center_of_buoyancy
    :initarg :update_center_of_buoyancy
    :type cl:boolean
    :initform cl:nil)
   (center_of_buoyancy
    :reader center_of_buoyancy
    :initarg :center_of_buoyancy
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (update_moments
    :reader update_moments
    :initarg :update_moments
    :type cl:boolean
    :initform cl:nil)
   (moments
    :reader moments
    :initarg :moments
    :type (cl:vector cl:float)
   :initform (cl:make-array 6 :element-type 'cl:float :initial-element 0.0))
   (update_linear_damping
    :reader update_linear_damping
    :initarg :update_linear_damping
    :type cl:boolean
    :initform cl:nil)
   (linear_damping
    :reader linear_damping
    :initarg :linear_damping
    :type (cl:vector cl:float)
   :initform (cl:make-array 6 :element-type 'cl:float :initial-element 0.0))
   (update_quadratic_damping
    :reader update_quadratic_damping
    :initarg :update_quadratic_damping
    :type cl:boolean
    :initform cl:nil)
   (quadratic_damping
    :reader quadratic_damping
    :initarg :quadratic_damping
    :type (cl:vector cl:float)
   :initform (cl:make-array 6 :element-type 'cl:float :initial-element 0.0))
   (update_added_mass
    :reader update_added_mass
    :initarg :update_added_mass
    :type cl:boolean
    :initform cl:nil)
   (added_mass
    :reader added_mass
    :initarg :added_mass
    :type (cl:vector cl:float)
   :initform (cl:make-array 6 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass DynamicsTuning (<DynamicsTuning>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <DynamicsTuning>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'DynamicsTuning)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<DynamicsTuning> is deprecated: use tauv_msgs-msg:DynamicsTuning instead.")))

(cl:ensure-generic-function 'update_mass-val :lambda-list '(m))
(cl:defmethod update_mass-val ((m <DynamicsTuning>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:update_mass-val is deprecated.  Use tauv_msgs-msg:update_mass instead.")
  (update_mass m))

(cl:ensure-generic-function 'mass-val :lambda-list '(m))
(cl:defmethod mass-val ((m <DynamicsTuning>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:mass-val is deprecated.  Use tauv_msgs-msg:mass instead.")
  (mass m))

(cl:ensure-generic-function 'update_volume-val :lambda-list '(m))
(cl:defmethod update_volume-val ((m <DynamicsTuning>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:update_volume-val is deprecated.  Use tauv_msgs-msg:update_volume instead.")
  (update_volume m))

(cl:ensure-generic-function 'volume-val :lambda-list '(m))
(cl:defmethod volume-val ((m <DynamicsTuning>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:volume-val is deprecated.  Use tauv_msgs-msg:volume instead.")
  (volume m))

(cl:ensure-generic-function 'update_water_density-val :lambda-list '(m))
(cl:defmethod update_water_density-val ((m <DynamicsTuning>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:update_water_density-val is deprecated.  Use tauv_msgs-msg:update_water_density instead.")
  (update_water_density m))

(cl:ensure-generic-function 'water_density-val :lambda-list '(m))
(cl:defmethod water_density-val ((m <DynamicsTuning>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:water_density-val is deprecated.  Use tauv_msgs-msg:water_density instead.")
  (water_density m))

(cl:ensure-generic-function 'update_center_of_gravity-val :lambda-list '(m))
(cl:defmethod update_center_of_gravity-val ((m <DynamicsTuning>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:update_center_of_gravity-val is deprecated.  Use tauv_msgs-msg:update_center_of_gravity instead.")
  (update_center_of_gravity m))

(cl:ensure-generic-function 'center_of_gravity-val :lambda-list '(m))
(cl:defmethod center_of_gravity-val ((m <DynamicsTuning>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:center_of_gravity-val is deprecated.  Use tauv_msgs-msg:center_of_gravity instead.")
  (center_of_gravity m))

(cl:ensure-generic-function 'update_center_of_buoyancy-val :lambda-list '(m))
(cl:defmethod update_center_of_buoyancy-val ((m <DynamicsTuning>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:update_center_of_buoyancy-val is deprecated.  Use tauv_msgs-msg:update_center_of_buoyancy instead.")
  (update_center_of_buoyancy m))

(cl:ensure-generic-function 'center_of_buoyancy-val :lambda-list '(m))
(cl:defmethod center_of_buoyancy-val ((m <DynamicsTuning>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:center_of_buoyancy-val is deprecated.  Use tauv_msgs-msg:center_of_buoyancy instead.")
  (center_of_buoyancy m))

(cl:ensure-generic-function 'update_moments-val :lambda-list '(m))
(cl:defmethod update_moments-val ((m <DynamicsTuning>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:update_moments-val is deprecated.  Use tauv_msgs-msg:update_moments instead.")
  (update_moments m))

(cl:ensure-generic-function 'moments-val :lambda-list '(m))
(cl:defmethod moments-val ((m <DynamicsTuning>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:moments-val is deprecated.  Use tauv_msgs-msg:moments instead.")
  (moments m))

(cl:ensure-generic-function 'update_linear_damping-val :lambda-list '(m))
(cl:defmethod update_linear_damping-val ((m <DynamicsTuning>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:update_linear_damping-val is deprecated.  Use tauv_msgs-msg:update_linear_damping instead.")
  (update_linear_damping m))

(cl:ensure-generic-function 'linear_damping-val :lambda-list '(m))
(cl:defmethod linear_damping-val ((m <DynamicsTuning>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:linear_damping-val is deprecated.  Use tauv_msgs-msg:linear_damping instead.")
  (linear_damping m))

(cl:ensure-generic-function 'update_quadratic_damping-val :lambda-list '(m))
(cl:defmethod update_quadratic_damping-val ((m <DynamicsTuning>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:update_quadratic_damping-val is deprecated.  Use tauv_msgs-msg:update_quadratic_damping instead.")
  (update_quadratic_damping m))

(cl:ensure-generic-function 'quadratic_damping-val :lambda-list '(m))
(cl:defmethod quadratic_damping-val ((m <DynamicsTuning>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:quadratic_damping-val is deprecated.  Use tauv_msgs-msg:quadratic_damping instead.")
  (quadratic_damping m))

(cl:ensure-generic-function 'update_added_mass-val :lambda-list '(m))
(cl:defmethod update_added_mass-val ((m <DynamicsTuning>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:update_added_mass-val is deprecated.  Use tauv_msgs-msg:update_added_mass instead.")
  (update_added_mass m))

(cl:ensure-generic-function 'added_mass-val :lambda-list '(m))
(cl:defmethod added_mass-val ((m <DynamicsTuning>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:added_mass-val is deprecated.  Use tauv_msgs-msg:added_mass instead.")
  (added_mass m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <DynamicsTuning>) ostream)
  "Serializes a message object of type '<DynamicsTuning>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'update_mass) 1 0)) ostream)
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'mass))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'update_volume) 1 0)) ostream)
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'volume))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'update_water_density) 1 0)) ostream)
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'water_density))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'update_center_of_gravity) 1 0)) ostream)
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'center_of_gravity))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'update_center_of_buoyancy) 1 0)) ostream)
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'center_of_buoyancy))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'update_moments) 1 0)) ostream)
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'moments))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'update_linear_damping) 1 0)) ostream)
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'linear_damping))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'update_quadratic_damping) 1 0)) ostream)
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'quadratic_damping))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'update_added_mass) 1 0)) ostream)
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'added_mass))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <DynamicsTuning>) istream)
  "Deserializes a message object of type '<DynamicsTuning>"
    (cl:setf (cl:slot-value msg 'update_mass) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'mass) (roslisp-utils:decode-double-float-bits bits)))
    (cl:setf (cl:slot-value msg 'update_volume) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'volume) (roslisp-utils:decode-double-float-bits bits)))
    (cl:setf (cl:slot-value msg 'update_water_density) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'water_density) (roslisp-utils:decode-double-float-bits bits)))
    (cl:setf (cl:slot-value msg 'update_center_of_gravity) (cl:not (cl:zerop (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'center_of_gravity) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'center_of_gravity)))
    (cl:dotimes (i 3)
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
    (cl:setf (cl:slot-value msg 'update_center_of_buoyancy) (cl:not (cl:zerop (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'center_of_buoyancy) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'center_of_buoyancy)))
    (cl:dotimes (i 3)
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
    (cl:setf (cl:slot-value msg 'update_moments) (cl:not (cl:zerop (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'moments) (cl:make-array 6))
  (cl:let ((vals (cl:slot-value msg 'moments)))
    (cl:dotimes (i 6)
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
    (cl:setf (cl:slot-value msg 'update_linear_damping) (cl:not (cl:zerop (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'linear_damping) (cl:make-array 6))
  (cl:let ((vals (cl:slot-value msg 'linear_damping)))
    (cl:dotimes (i 6)
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
    (cl:setf (cl:slot-value msg 'update_quadratic_damping) (cl:not (cl:zerop (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'quadratic_damping) (cl:make-array 6))
  (cl:let ((vals (cl:slot-value msg 'quadratic_damping)))
    (cl:dotimes (i 6)
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
    (cl:setf (cl:slot-value msg 'update_added_mass) (cl:not (cl:zerop (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'added_mass) (cl:make-array 6))
  (cl:let ((vals (cl:slot-value msg 'added_mass)))
    (cl:dotimes (i 6)
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
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<DynamicsTuning>)))
  "Returns string type for a message object of type '<DynamicsTuning>"
  "tauv_msgs/DynamicsTuning")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DynamicsTuning)))
  "Returns string type for a message object of type 'DynamicsTuning"
  "tauv_msgs/DynamicsTuning")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<DynamicsTuning>)))
  "Returns md5sum for a message object of type '<DynamicsTuning>"
  "1657e74b53352c5e93a01a5d1743eeaa")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'DynamicsTuning)))
  "Returns md5sum for a message object of type 'DynamicsTuning"
  "1657e74b53352c5e93a01a5d1743eeaa")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<DynamicsTuning>)))
  "Returns full string definition for message of type '<DynamicsTuning>"
  (cl:format cl:nil "bool update_mass~%float64 mass~%bool update_volume~%float64 volume~%bool update_water_density~%float64 water_density~%bool update_center_of_gravity~%float64[3] center_of_gravity~%bool update_center_of_buoyancy~%float64[3] center_of_buoyancy~%bool update_moments~%float64[6] moments~%bool update_linear_damping~%float64[6] linear_damping~%bool update_quadratic_damping~%float64[6] quadratic_damping~%bool update_added_mass~%float64[6] added_mass~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'DynamicsTuning)))
  "Returns full string definition for message of type 'DynamicsTuning"
  (cl:format cl:nil "bool update_mass~%float64 mass~%bool update_volume~%float64 volume~%bool update_water_density~%float64 water_density~%bool update_center_of_gravity~%float64[3] center_of_gravity~%bool update_center_of_buoyancy~%float64[3] center_of_buoyancy~%bool update_moments~%float64[6] moments~%bool update_linear_damping~%float64[6] linear_damping~%bool update_quadratic_damping~%float64[6] quadratic_damping~%bool update_added_mass~%float64[6] added_mass~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <DynamicsTuning>))
  (cl:+ 0
     1
     8
     1
     8
     1
     8
     1
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'center_of_gravity) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     1
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'center_of_buoyancy) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     1
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'moments) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     1
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'linear_damping) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     1
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'quadratic_damping) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     1
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'added_mass) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <DynamicsTuning>))
  "Converts a ROS message object to a list"
  (cl:list 'DynamicsTuning
    (cl:cons ':update_mass (update_mass msg))
    (cl:cons ':mass (mass msg))
    (cl:cons ':update_volume (update_volume msg))
    (cl:cons ':volume (volume msg))
    (cl:cons ':update_water_density (update_water_density msg))
    (cl:cons ':water_density (water_density msg))
    (cl:cons ':update_center_of_gravity (update_center_of_gravity msg))
    (cl:cons ':center_of_gravity (center_of_gravity msg))
    (cl:cons ':update_center_of_buoyancy (update_center_of_buoyancy msg))
    (cl:cons ':center_of_buoyancy (center_of_buoyancy msg))
    (cl:cons ':update_moments (update_moments msg))
    (cl:cons ':moments (moments msg))
    (cl:cons ':update_linear_damping (update_linear_damping msg))
    (cl:cons ':linear_damping (linear_damping msg))
    (cl:cons ':update_quadratic_damping (update_quadratic_damping msg))
    (cl:cons ':quadratic_damping (quadratic_damping msg))
    (cl:cons ':update_added_mass (update_added_mass msg))
    (cl:cons ':added_mass (added_mass msg))
))
