; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude NavigationState.msg.html

(cl:defclass <NavigationState> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (position
    :reader position
    :initarg :position
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (linear_velocity
    :reader linear_velocity
    :initarg :linear_velocity
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (linear_acceleration
    :reader linear_acceleration
    :initarg :linear_acceleration
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (orientation
    :reader orientation
    :initarg :orientation
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (euler_velocity
    :reader euler_velocity
    :initarg :euler_velocity
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (euler_acceleration
    :reader euler_acceleration
    :initarg :euler_acceleration
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3)))
)

(cl:defclass NavigationState (<NavigationState>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <NavigationState>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'NavigationState)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<NavigationState> is deprecated: use tauv_msgs-msg:NavigationState instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <NavigationState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:header-val is deprecated.  Use tauv_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'position-val :lambda-list '(m))
(cl:defmethod position-val ((m <NavigationState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:position-val is deprecated.  Use tauv_msgs-msg:position instead.")
  (position m))

(cl:ensure-generic-function 'linear_velocity-val :lambda-list '(m))
(cl:defmethod linear_velocity-val ((m <NavigationState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:linear_velocity-val is deprecated.  Use tauv_msgs-msg:linear_velocity instead.")
  (linear_velocity m))

(cl:ensure-generic-function 'linear_acceleration-val :lambda-list '(m))
(cl:defmethod linear_acceleration-val ((m <NavigationState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:linear_acceleration-val is deprecated.  Use tauv_msgs-msg:linear_acceleration instead.")
  (linear_acceleration m))

(cl:ensure-generic-function 'orientation-val :lambda-list '(m))
(cl:defmethod orientation-val ((m <NavigationState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:orientation-val is deprecated.  Use tauv_msgs-msg:orientation instead.")
  (orientation m))

(cl:ensure-generic-function 'euler_velocity-val :lambda-list '(m))
(cl:defmethod euler_velocity-val ((m <NavigationState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:euler_velocity-val is deprecated.  Use tauv_msgs-msg:euler_velocity instead.")
  (euler_velocity m))

(cl:ensure-generic-function 'euler_acceleration-val :lambda-list '(m))
(cl:defmethod euler_acceleration-val ((m <NavigationState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:euler_acceleration-val is deprecated.  Use tauv_msgs-msg:euler_acceleration instead.")
  (euler_acceleration m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <NavigationState>) ostream)
  "Serializes a message object of type '<NavigationState>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'position) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'linear_velocity) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'linear_acceleration) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'orientation) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'euler_velocity) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'euler_acceleration) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <NavigationState>) istream)
  "Deserializes a message object of type '<NavigationState>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'position) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'linear_velocity) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'linear_acceleration) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'orientation) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'euler_velocity) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'euler_acceleration) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<NavigationState>)))
  "Returns string type for a message object of type '<NavigationState>"
  "tauv_msgs/NavigationState")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'NavigationState)))
  "Returns string type for a message object of type 'NavigationState"
  "tauv_msgs/NavigationState")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<NavigationState>)))
  "Returns md5sum for a message object of type '<NavigationState>"
  "3507e091ce71f01fa9dea0061304746a")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'NavigationState)))
  "Returns md5sum for a message object of type 'NavigationState"
  "3507e091ce71f01fa9dea0061304746a")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<NavigationState>)))
  "Returns full string definition for message of type '<NavigationState>"
  (cl:format cl:nil "Header header~%~%geometry_msgs/Vector3 position~%geometry_msgs/Vector3 linear_velocity      # body~%geometry_msgs/Vector3 linear_acceleration  # body~%geometry_msgs/Vector3 orientation          # (r, p, y)~%geometry_msgs/Vector3 euler_velocity       # (dr/dt, dp/dt, dy/dt)~%geometry_msgs/Vector3 euler_acceleration   # (d2r/dt2, d2p/dt2, d2y/dt2)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'NavigationState)))
  "Returns full string definition for message of type 'NavigationState"
  (cl:format cl:nil "Header header~%~%geometry_msgs/Vector3 position~%geometry_msgs/Vector3 linear_velocity      # body~%geometry_msgs/Vector3 linear_acceleration  # body~%geometry_msgs/Vector3 orientation          # (r, p, y)~%geometry_msgs/Vector3 euler_velocity       # (dr/dt, dp/dt, dy/dt)~%geometry_msgs/Vector3 euler_acceleration   # (d2r/dt2, d2p/dt2, d2y/dt2)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <NavigationState>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'position))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'linear_velocity))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'linear_acceleration))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'orientation))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'euler_velocity))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'euler_acceleration))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <NavigationState>))
  "Converts a ROS message object to a list"
  (cl:list 'NavigationState
    (cl:cons ':header (header msg))
    (cl:cons ':position (position msg))
    (cl:cons ':linear_velocity (linear_velocity msg))
    (cl:cons ':linear_acceleration (linear_acceleration msg))
    (cl:cons ':orientation (orientation msg))
    (cl:cons ':euler_velocity (euler_velocity msg))
    (cl:cons ':euler_acceleration (euler_acceleration msg))
))
