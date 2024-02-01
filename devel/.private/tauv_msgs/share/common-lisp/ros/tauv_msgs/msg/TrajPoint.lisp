; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude TrajPoint.msg.html

(cl:defclass <TrajPoint> (roslisp-msg-protocol:ros-message)
  ((pose
    :reader pose
    :initarg :pose
    :type geometry_msgs-msg:Pose
    :initform (cl:make-instance 'geometry_msgs-msg:Pose))
   (twist
    :reader twist
    :initarg :twist
    :type geometry_msgs-msg:Twist
    :initform (cl:make-instance 'geometry_msgs-msg:Twist)))
)

(cl:defclass TrajPoint (<TrajPoint>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TrajPoint>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TrajPoint)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<TrajPoint> is deprecated: use tauv_msgs-msg:TrajPoint instead.")))

(cl:ensure-generic-function 'pose-val :lambda-list '(m))
(cl:defmethod pose-val ((m <TrajPoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:pose-val is deprecated.  Use tauv_msgs-msg:pose instead.")
  (pose m))

(cl:ensure-generic-function 'twist-val :lambda-list '(m))
(cl:defmethod twist-val ((m <TrajPoint>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:twist-val is deprecated.  Use tauv_msgs-msg:twist instead.")
  (twist m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TrajPoint>) ostream)
  "Serializes a message object of type '<TrajPoint>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'pose) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'twist) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TrajPoint>) istream)
  "Deserializes a message object of type '<TrajPoint>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'pose) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'twist) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TrajPoint>)))
  "Returns string type for a message object of type '<TrajPoint>"
  "tauv_msgs/TrajPoint")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TrajPoint)))
  "Returns string type for a message object of type 'TrajPoint"
  "tauv_msgs/TrajPoint")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TrajPoint>)))
  "Returns md5sum for a message object of type '<TrajPoint>"
  "c79f0d88a7597db980a56d7ac144c654")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TrajPoint)))
  "Returns md5sum for a message object of type 'TrajPoint"
  "c79f0d88a7597db980a56d7ac144c654")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TrajPoint>)))
  "Returns full string definition for message of type '<TrajPoint>"
  (cl:format cl:nil "geometry_msgs/Pose pose     # world frame~%geometry_msgs/Twist twist   # world frame~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: geometry_msgs/Twist~%# This expresses velocity in free space broken into its linear and angular parts.~%Vector3  linear~%Vector3  angular~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TrajPoint)))
  "Returns full string definition for message of type 'TrajPoint"
  (cl:format cl:nil "geometry_msgs/Pose pose     # world frame~%geometry_msgs/Twist twist   # world frame~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: geometry_msgs/Twist~%# This expresses velocity in free space broken into its linear and angular parts.~%Vector3  linear~%Vector3  angular~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TrajPoint>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'pose))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'twist))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TrajPoint>))
  "Converts a ROS message object to a list"
  (cl:list 'TrajPoint
    (cl:cons ':pose (pose msg))
    (cl:cons ':twist (twist msg))
))
