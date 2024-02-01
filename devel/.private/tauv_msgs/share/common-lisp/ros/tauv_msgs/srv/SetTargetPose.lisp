; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-srv)


;//! \htmlinclude SetTargetPose-request.msg.html

(cl:defclass <SetTargetPose-request> (roslisp-msg-protocol:ros-message)
  ((pose
    :reader pose
    :initarg :pose
    :type geometry_msgs-msg:Pose
    :initform (cl:make-instance 'geometry_msgs-msg:Pose)))
)

(cl:defclass SetTargetPose-request (<SetTargetPose-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetTargetPose-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetTargetPose-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<SetTargetPose-request> is deprecated: use tauv_msgs-srv:SetTargetPose-request instead.")))

(cl:ensure-generic-function 'pose-val :lambda-list '(m))
(cl:defmethod pose-val ((m <SetTargetPose-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:pose-val is deprecated.  Use tauv_msgs-srv:pose instead.")
  (pose m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetTargetPose-request>) ostream)
  "Serializes a message object of type '<SetTargetPose-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'pose) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetTargetPose-request>) istream)
  "Deserializes a message object of type '<SetTargetPose-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'pose) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetTargetPose-request>)))
  "Returns string type for a service object of type '<SetTargetPose-request>"
  "tauv_msgs/SetTargetPoseRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetTargetPose-request)))
  "Returns string type for a service object of type 'SetTargetPose-request"
  "tauv_msgs/SetTargetPoseRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetTargetPose-request>)))
  "Returns md5sum for a message object of type '<SetTargetPose-request>"
  "b66f4c8c2fa81b98544cf0f3e10950ee")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetTargetPose-request)))
  "Returns md5sum for a message object of type 'SetTargetPose-request"
  "b66f4c8c2fa81b98544cf0f3e10950ee")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetTargetPose-request>)))
  "Returns full string definition for message of type '<SetTargetPose-request>"
  (cl:format cl:nil "geometry_msgs/Pose pose~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetTargetPose-request)))
  "Returns full string definition for message of type 'SetTargetPose-request"
  (cl:format cl:nil "geometry_msgs/Pose pose~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetTargetPose-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'pose))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetTargetPose-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetTargetPose-request
    (cl:cons ':pose (pose msg))
))
;//! \htmlinclude SetTargetPose-response.msg.html

(cl:defclass <SetTargetPose-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass SetTargetPose-response (<SetTargetPose-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetTargetPose-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetTargetPose-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<SetTargetPose-response> is deprecated: use tauv_msgs-srv:SetTargetPose-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <SetTargetPose-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:success-val is deprecated.  Use tauv_msgs-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetTargetPose-response>) ostream)
  "Serializes a message object of type '<SetTargetPose-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetTargetPose-response>) istream)
  "Deserializes a message object of type '<SetTargetPose-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetTargetPose-response>)))
  "Returns string type for a service object of type '<SetTargetPose-response>"
  "tauv_msgs/SetTargetPoseResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetTargetPose-response)))
  "Returns string type for a service object of type 'SetTargetPose-response"
  "tauv_msgs/SetTargetPoseResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetTargetPose-response>)))
  "Returns md5sum for a message object of type '<SetTargetPose-response>"
  "b66f4c8c2fa81b98544cf0f3e10950ee")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetTargetPose-response)))
  "Returns md5sum for a message object of type 'SetTargetPose-response"
  "b66f4c8c2fa81b98544cf0f3e10950ee")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetTargetPose-response>)))
  "Returns full string definition for message of type '<SetTargetPose-response>"
  (cl:format cl:nil "bool success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetTargetPose-response)))
  "Returns full string definition for message of type 'SetTargetPose-response"
  (cl:format cl:nil "bool success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetTargetPose-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetTargetPose-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetTargetPose-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetTargetPose)))
  'SetTargetPose-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetTargetPose)))
  'SetTargetPose-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetTargetPose)))
  "Returns string type for a service object of type '<SetTargetPose>"
  "tauv_msgs/SetTargetPose")