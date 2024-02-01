; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude RegisterMeasurement.msg.html

(cl:defclass <RegisterMeasurement> (roslisp-msg-protocol:ros-message)
  ((pg_meas
    :reader pg_meas
    :initarg :pg_meas
    :type tauv_msgs-msg:PoseGraphMeasurement
    :initform (cl:make-instance 'tauv_msgs-msg:PoseGraphMeasurement)))
)

(cl:defclass RegisterMeasurement (<RegisterMeasurement>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RegisterMeasurement>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RegisterMeasurement)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<RegisterMeasurement> is deprecated: use tauv_msgs-msg:RegisterMeasurement instead.")))

(cl:ensure-generic-function 'pg_meas-val :lambda-list '(m))
(cl:defmethod pg_meas-val ((m <RegisterMeasurement>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:pg_meas-val is deprecated.  Use tauv_msgs-msg:pg_meas instead.")
  (pg_meas m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RegisterMeasurement>) ostream)
  "Serializes a message object of type '<RegisterMeasurement>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'pg_meas) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RegisterMeasurement>) istream)
  "Deserializes a message object of type '<RegisterMeasurement>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'pg_meas) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RegisterMeasurement>)))
  "Returns string type for a message object of type '<RegisterMeasurement>"
  "tauv_msgs/RegisterMeasurement")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RegisterMeasurement)))
  "Returns string type for a message object of type 'RegisterMeasurement"
  "tauv_msgs/RegisterMeasurement")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RegisterMeasurement>)))
  "Returns md5sum for a message object of type '<RegisterMeasurement>"
  "b355dd17bfdad2a0499de8384660e7ff")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RegisterMeasurement)))
  "Returns md5sum for a message object of type 'RegisterMeasurement"
  "b355dd17bfdad2a0499de8384660e7ff")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RegisterMeasurement>)))
  "Returns full string definition for message of type '<RegisterMeasurement>"
  (cl:format cl:nil "PoseGraphMeasurement pg_meas~%================================================================================~%MSG: tauv_msgs/PoseGraphMeasurement~%Header header~%uint32 landmark_id~%geometry_msgs/Point position~%~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RegisterMeasurement)))
  "Returns full string definition for message of type 'RegisterMeasurement"
  (cl:format cl:nil "PoseGraphMeasurement pg_meas~%================================================================================~%MSG: tauv_msgs/PoseGraphMeasurement~%Header header~%uint32 landmark_id~%geometry_msgs/Point position~%~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RegisterMeasurement>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'pg_meas))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RegisterMeasurement>))
  "Converts a ROS message object to a list"
  (cl:list 'RegisterMeasurement
    (cl:cons ':pg_meas (pg_meas msg))
))
