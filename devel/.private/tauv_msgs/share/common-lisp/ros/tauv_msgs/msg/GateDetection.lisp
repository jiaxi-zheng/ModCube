; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude GateDetection.msg.html

(cl:defclass <GateDetection> (roslisp-msg-protocol:ros-message)
  ((leftCorner
    :reader leftCorner
    :initarg :leftCorner
    :type geometry_msgs-msg:Point
    :initform (cl:make-instance 'geometry_msgs-msg:Point))
   (rightCorner
    :reader rightCorner
    :initarg :rightCorner
    :type geometry_msgs-msg:Point
    :initform (cl:make-instance 'geometry_msgs-msg:Point))
   (centerPoint
    :reader centerPoint
    :initarg :centerPoint
    :type geometry_msgs-msg:Point
    :initform (cl:make-instance 'geometry_msgs-msg:Point)))
)

(cl:defclass GateDetection (<GateDetection>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GateDetection>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GateDetection)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<GateDetection> is deprecated: use tauv_msgs-msg:GateDetection instead.")))

(cl:ensure-generic-function 'leftCorner-val :lambda-list '(m))
(cl:defmethod leftCorner-val ((m <GateDetection>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:leftCorner-val is deprecated.  Use tauv_msgs-msg:leftCorner instead.")
  (leftCorner m))

(cl:ensure-generic-function 'rightCorner-val :lambda-list '(m))
(cl:defmethod rightCorner-val ((m <GateDetection>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:rightCorner-val is deprecated.  Use tauv_msgs-msg:rightCorner instead.")
  (rightCorner m))

(cl:ensure-generic-function 'centerPoint-val :lambda-list '(m))
(cl:defmethod centerPoint-val ((m <GateDetection>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:centerPoint-val is deprecated.  Use tauv_msgs-msg:centerPoint instead.")
  (centerPoint m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GateDetection>) ostream)
  "Serializes a message object of type '<GateDetection>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'leftCorner) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'rightCorner) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'centerPoint) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GateDetection>) istream)
  "Deserializes a message object of type '<GateDetection>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'leftCorner) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'rightCorner) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'centerPoint) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GateDetection>)))
  "Returns string type for a message object of type '<GateDetection>"
  "tauv_msgs/GateDetection")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GateDetection)))
  "Returns string type for a message object of type 'GateDetection"
  "tauv_msgs/GateDetection")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GateDetection>)))
  "Returns md5sum for a message object of type '<GateDetection>"
  "8a67a29e701c68c820bd45950230c212")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GateDetection)))
  "Returns md5sum for a message object of type 'GateDetection"
  "8a67a29e701c68c820bd45950230c212")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GateDetection>)))
  "Returns full string definition for message of type '<GateDetection>"
  (cl:format cl:nil "geometry_msgs/Point leftCorner~%geometry_msgs/Point rightCorner~%geometry_msgs/Point centerPoint~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GateDetection)))
  "Returns full string definition for message of type 'GateDetection"
  (cl:format cl:nil "geometry_msgs/Point leftCorner~%geometry_msgs/Point rightCorner~%geometry_msgs/Point centerPoint~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GateDetection>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'leftCorner))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'rightCorner))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'centerPoint))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GateDetection>))
  "Converts a ROS message object to a list"
  (cl:list 'GateDetection
    (cl:cons ':leftCorner (leftCorner msg))
    (cl:cons ':rightCorner (rightCorner msg))
    (cl:cons ':centerPoint (centerPoint msg))
))
