; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-srv)


;//! \htmlinclude RunMission-request.msg.html

(cl:defclass <RunMission-request> (roslisp-msg-protocol:ros-message)
  ((mission_name
    :reader mission_name
    :initarg :mission_name
    :type cl:string
    :initform ""))
)

(cl:defclass RunMission-request (<RunMission-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RunMission-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RunMission-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<RunMission-request> is deprecated: use tauv_msgs-srv:RunMission-request instead.")))

(cl:ensure-generic-function 'mission_name-val :lambda-list '(m))
(cl:defmethod mission_name-val ((m <RunMission-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:mission_name-val is deprecated.  Use tauv_msgs-srv:mission_name instead.")
  (mission_name m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RunMission-request>) ostream)
  "Serializes a message object of type '<RunMission-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'mission_name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'mission_name))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RunMission-request>) istream)
  "Deserializes a message object of type '<RunMission-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'mission_name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'mission_name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RunMission-request>)))
  "Returns string type for a service object of type '<RunMission-request>"
  "tauv_msgs/RunMissionRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RunMission-request)))
  "Returns string type for a service object of type 'RunMission-request"
  "tauv_msgs/RunMissionRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RunMission-request>)))
  "Returns md5sum for a message object of type '<RunMission-request>"
  "9f981363eccc116598d821fe063b0990")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RunMission-request)))
  "Returns md5sum for a message object of type 'RunMission-request"
  "9f981363eccc116598d821fe063b0990")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RunMission-request>)))
  "Returns full string definition for message of type '<RunMission-request>"
  (cl:format cl:nil "string mission_name~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RunMission-request)))
  "Returns full string definition for message of type 'RunMission-request"
  (cl:format cl:nil "string mission_name~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RunMission-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'mission_name))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RunMission-request>))
  "Converts a ROS message object to a list"
  (cl:list 'RunMission-request
    (cl:cons ':mission_name (mission_name msg))
))
;//! \htmlinclude RunMission-response.msg.html

(cl:defclass <RunMission-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil)
   (message
    :reader message
    :initarg :message
    :type cl:string
    :initform ""))
)

(cl:defclass RunMission-response (<RunMission-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RunMission-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RunMission-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<RunMission-response> is deprecated: use tauv_msgs-srv:RunMission-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <RunMission-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:success-val is deprecated.  Use tauv_msgs-srv:success instead.")
  (success m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <RunMission-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:message-val is deprecated.  Use tauv_msgs-srv:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RunMission-response>) ostream)
  "Serializes a message object of type '<RunMission-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RunMission-response>) istream)
  "Deserializes a message object of type '<RunMission-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'message) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'message) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RunMission-response>)))
  "Returns string type for a service object of type '<RunMission-response>"
  "tauv_msgs/RunMissionResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RunMission-response)))
  "Returns string type for a service object of type 'RunMission-response"
  "tauv_msgs/RunMissionResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RunMission-response>)))
  "Returns md5sum for a message object of type '<RunMission-response>"
  "9f981363eccc116598d821fe063b0990")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RunMission-response)))
  "Returns md5sum for a message object of type 'RunMission-response"
  "9f981363eccc116598d821fe063b0990")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RunMission-response>)))
  "Returns full string definition for message of type '<RunMission-response>"
  (cl:format cl:nil "bool success~%string message~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RunMission-response)))
  "Returns full string definition for message of type 'RunMission-response"
  (cl:format cl:nil "bool success~%string message~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RunMission-response>))
  (cl:+ 0
     1
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RunMission-response>))
  "Converts a ROS message object to a list"
  (cl:list 'RunMission-response
    (cl:cons ':success (success msg))
    (cl:cons ':message (message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'RunMission)))
  'RunMission-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'RunMission)))
  'RunMission-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RunMission)))
  "Returns string type for a service object of type '<RunMission>"
  "tauv_msgs/RunMission")