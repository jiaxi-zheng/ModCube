; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-srv)


;//! \htmlinclude SetPose-request.msg.html

(cl:defclass <SetPose-request> (roslisp-msg-protocol:ros-message)
  ((position
    :reader position
    :initarg :position
    :type geometry_msgs-msg:Vector3
    :initform (cl:make-instance 'geometry_msgs-msg:Vector3))
   (yaw
    :reader yaw
    :initarg :yaw
    :type cl:float
    :initform 0.0))
)

(cl:defclass SetPose-request (<SetPose-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetPose-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetPose-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<SetPose-request> is deprecated: use tauv_msgs-srv:SetPose-request instead.")))

(cl:ensure-generic-function 'position-val :lambda-list '(m))
(cl:defmethod position-val ((m <SetPose-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:position-val is deprecated.  Use tauv_msgs-srv:position instead.")
  (position m))

(cl:ensure-generic-function 'yaw-val :lambda-list '(m))
(cl:defmethod yaw-val ((m <SetPose-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:yaw-val is deprecated.  Use tauv_msgs-srv:yaw instead.")
  (yaw m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetPose-request>) ostream)
  "Serializes a message object of type '<SetPose-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'position) ostream)
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'yaw))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetPose-request>) istream)
  "Deserializes a message object of type '<SetPose-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'position) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'yaw) (roslisp-utils:decode-double-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetPose-request>)))
  "Returns string type for a service object of type '<SetPose-request>"
  "tauv_msgs/SetPoseRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetPose-request)))
  "Returns string type for a service object of type 'SetPose-request"
  "tauv_msgs/SetPoseRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetPose-request>)))
  "Returns md5sum for a message object of type '<SetPose-request>"
  "c0694c76e8e4d6ff81999164dc53cd23")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetPose-request)))
  "Returns md5sum for a message object of type 'SetPose-request"
  "c0694c76e8e4d6ff81999164dc53cd23")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetPose-request>)))
  "Returns full string definition for message of type '<SetPose-request>"
  (cl:format cl:nil "geometry_msgs/Vector3 position~%float64 yaw~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetPose-request)))
  "Returns full string definition for message of type 'SetPose-request"
  (cl:format cl:nil "geometry_msgs/Vector3 position~%float64 yaw~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetPose-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'position))
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetPose-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetPose-request
    (cl:cons ':position (position msg))
    (cl:cons ':yaw (yaw msg))
))
;//! \htmlinclude SetPose-response.msg.html

(cl:defclass <SetPose-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass SetPose-response (<SetPose-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetPose-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetPose-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<SetPose-response> is deprecated: use tauv_msgs-srv:SetPose-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <SetPose-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:success-val is deprecated.  Use tauv_msgs-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetPose-response>) ostream)
  "Serializes a message object of type '<SetPose-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetPose-response>) istream)
  "Deserializes a message object of type '<SetPose-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetPose-response>)))
  "Returns string type for a service object of type '<SetPose-response>"
  "tauv_msgs/SetPoseResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetPose-response)))
  "Returns string type for a service object of type 'SetPose-response"
  "tauv_msgs/SetPoseResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetPose-response>)))
  "Returns md5sum for a message object of type '<SetPose-response>"
  "c0694c76e8e4d6ff81999164dc53cd23")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetPose-response)))
  "Returns md5sum for a message object of type 'SetPose-response"
  "c0694c76e8e4d6ff81999164dc53cd23")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetPose-response>)))
  "Returns full string definition for message of type '<SetPose-response>"
  (cl:format cl:nil "bool success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetPose-response)))
  "Returns full string definition for message of type 'SetPose-response"
  (cl:format cl:nil "bool success~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetPose-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetPose-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetPose-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetPose)))
  'SetPose-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetPose)))
  'SetPose-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetPose)))
  "Returns string type for a service object of type '<SetPose>"
  "tauv_msgs/SetPose")