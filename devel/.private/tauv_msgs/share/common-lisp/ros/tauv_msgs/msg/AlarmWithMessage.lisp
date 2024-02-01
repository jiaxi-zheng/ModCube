; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude AlarmWithMessage.msg.html

(cl:defclass <AlarmWithMessage> (roslisp-msg-protocol:ros-message)
  ((id
    :reader id
    :initarg :id
    :type cl:integer
    :initform 0)
   (set
    :reader set
    :initarg :set
    :type cl:boolean
    :initform cl:nil)
   (message
    :reader message
    :initarg :message
    :type cl:string
    :initform ""))
)

(cl:defclass AlarmWithMessage (<AlarmWithMessage>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <AlarmWithMessage>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'AlarmWithMessage)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<AlarmWithMessage> is deprecated: use tauv_msgs-msg:AlarmWithMessage instead.")))

(cl:ensure-generic-function 'id-val :lambda-list '(m))
(cl:defmethod id-val ((m <AlarmWithMessage>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:id-val is deprecated.  Use tauv_msgs-msg:id instead.")
  (id m))

(cl:ensure-generic-function 'set-val :lambda-list '(m))
(cl:defmethod set-val ((m <AlarmWithMessage>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:set-val is deprecated.  Use tauv_msgs-msg:set instead.")
  (set m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <AlarmWithMessage>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:message-val is deprecated.  Use tauv_msgs-msg:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <AlarmWithMessage>) ostream)
  "Serializes a message object of type '<AlarmWithMessage>"
  (cl:let* ((signed (cl:slot-value msg 'id)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'set) 1 0)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <AlarmWithMessage>) istream)
  "Deserializes a message object of type '<AlarmWithMessage>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'id) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:setf (cl:slot-value msg 'set) (cl:not (cl:zerop (cl:read-byte istream))))
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<AlarmWithMessage>)))
  "Returns string type for a message object of type '<AlarmWithMessage>"
  "tauv_msgs/AlarmWithMessage")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'AlarmWithMessage)))
  "Returns string type for a message object of type 'AlarmWithMessage"
  "tauv_msgs/AlarmWithMessage")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<AlarmWithMessage>)))
  "Returns md5sum for a message object of type '<AlarmWithMessage>"
  "2b0df1cdd443c9ac99920553ac2eb8d8")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'AlarmWithMessage)))
  "Returns md5sum for a message object of type 'AlarmWithMessage"
  "2b0df1cdd443c9ac99920553ac2eb8d8")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<AlarmWithMessage>)))
  "Returns full string definition for message of type '<AlarmWithMessage>"
  (cl:format cl:nil "int32 id            # ID of the alarm~%bool set            # True = set, False = Cleared~%string message      # Readable message~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'AlarmWithMessage)))
  "Returns full string definition for message of type 'AlarmWithMessage"
  (cl:format cl:nil "int32 id            # ID of the alarm~%bool set            # True = set, False = Cleared~%string message      # Readable message~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <AlarmWithMessage>))
  (cl:+ 0
     4
     1
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <AlarmWithMessage>))
  "Converts a ROS message object to a list"
  (cl:list 'AlarmWithMessage
    (cl:cons ':id (id msg))
    (cl:cons ':set (set msg))
    (cl:cons ':message (message msg))
))
