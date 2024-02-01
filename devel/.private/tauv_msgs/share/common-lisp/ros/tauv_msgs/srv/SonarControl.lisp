; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-srv)


;//! \htmlinclude SonarControl-request.msg.html

(cl:defclass <SonarControl-request> (roslisp-msg-protocol:ros-message)
  ((op
    :reader op
    :initarg :op
    :type cl:fixnum
    :initform 0))
)

(cl:defclass SonarControl-request (<SonarControl-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SonarControl-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SonarControl-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<SonarControl-request> is deprecated: use tauv_msgs-srv:SonarControl-request instead.")))

(cl:ensure-generic-function 'op-val :lambda-list '(m))
(cl:defmethod op-val ((m <SonarControl-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:op-val is deprecated.  Use tauv_msgs-srv:op instead.")
  (op m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SonarControl-request>) ostream)
  "Serializes a message object of type '<SonarControl-request>"
  (cl:let* ((signed (cl:slot-value msg 'op)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SonarControl-request>) istream)
  "Deserializes a message object of type '<SonarControl-request>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'op) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SonarControl-request>)))
  "Returns string type for a service object of type '<SonarControl-request>"
  "tauv_msgs/SonarControlRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SonarControl-request)))
  "Returns string type for a service object of type 'SonarControl-request"
  "tauv_msgs/SonarControlRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SonarControl-request>)))
  "Returns md5sum for a message object of type '<SonarControl-request>"
  "cdbd07758f57cd8fbf9c7b8e225a19cf")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SonarControl-request)))
  "Returns md5sum for a message object of type 'SonarControl-request"
  "cdbd07758f57cd8fbf9c7b8e225a19cf")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SonarControl-request>)))
  "Returns full string definition for message of type '<SonarControl-request>"
  (cl:format cl:nil "int8 op~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SonarControl-request)))
  "Returns full string definition for message of type 'SonarControl-request"
  (cl:format cl:nil "int8 op~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SonarControl-request>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SonarControl-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SonarControl-request
    (cl:cons ':op (op msg))
))
;//! \htmlinclude SonarControl-response.msg.html

(cl:defclass <SonarControl-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass SonarControl-response (<SonarControl-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SonarControl-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SonarControl-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<SonarControl-response> is deprecated: use tauv_msgs-srv:SonarControl-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <SonarControl-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:success-val is deprecated.  Use tauv_msgs-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SonarControl-response>) ostream)
  "Serializes a message object of type '<SonarControl-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SonarControl-response>) istream)
  "Deserializes a message object of type '<SonarControl-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SonarControl-response>)))
  "Returns string type for a service object of type '<SonarControl-response>"
  "tauv_msgs/SonarControlResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SonarControl-response)))
  "Returns string type for a service object of type 'SonarControl-response"
  "tauv_msgs/SonarControlResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SonarControl-response>)))
  "Returns md5sum for a message object of type '<SonarControl-response>"
  "cdbd07758f57cd8fbf9c7b8e225a19cf")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SonarControl-response)))
  "Returns md5sum for a message object of type 'SonarControl-response"
  "cdbd07758f57cd8fbf9c7b8e225a19cf")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SonarControl-response>)))
  "Returns full string definition for message of type '<SonarControl-response>"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SonarControl-response)))
  "Returns full string definition for message of type 'SonarControl-response"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SonarControl-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SonarControl-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SonarControl-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SonarControl)))
  'SonarControl-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SonarControl)))
  'SonarControl-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SonarControl)))
  "Returns string type for a service object of type '<SonarControl>"
  "tauv_msgs/SonarControl")