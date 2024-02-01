; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude Message.msg.html

(cl:defclass <Message> (roslisp-msg-protocol:ros-message)
  ((stamp
    :reader stamp
    :initarg :stamp
    :type cl:real
    :initform 0)
   (color_code_256
    :reader color_code_256
    :initarg :color_code_256
    :type cl:fixnum
    :initform 0)
   (severity
    :reader severity
    :initarg :severity
    :type cl:fixnum
    :initform 0)
   (message
    :reader message
    :initarg :message
    :type cl:string
    :initform ""))
)

(cl:defclass Message (<Message>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Message>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Message)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<Message> is deprecated: use tauv_msgs-msg:Message instead.")))

(cl:ensure-generic-function 'stamp-val :lambda-list '(m))
(cl:defmethod stamp-val ((m <Message>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:stamp-val is deprecated.  Use tauv_msgs-msg:stamp instead.")
  (stamp m))

(cl:ensure-generic-function 'color_code_256-val :lambda-list '(m))
(cl:defmethod color_code_256-val ((m <Message>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:color_code_256-val is deprecated.  Use tauv_msgs-msg:color_code_256 instead.")
  (color_code_256 m))

(cl:ensure-generic-function 'severity-val :lambda-list '(m))
(cl:defmethod severity-val ((m <Message>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:severity-val is deprecated.  Use tauv_msgs-msg:severity instead.")
  (severity m))

(cl:ensure-generic-function 'message-val :lambda-list '(m))
(cl:defmethod message-val ((m <Message>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:message-val is deprecated.  Use tauv_msgs-msg:message instead.")
  (message m))
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql '<Message>)))
    "Constants for message type '<Message>"
  '((:ERROR . 0)
    (:WARNING . 1)
    (:INFO . 2)
    (:DEBUG . 3))
)
(cl:defmethod roslisp-msg-protocol:symbol-codes ((msg-type (cl:eql 'Message)))
    "Constants for message type 'Message"
  '((:ERROR . 0)
    (:WARNING . 1)
    (:INFO . 2)
    (:DEBUG . 3))
)
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Message>) ostream)
  "Serializes a message object of type '<Message>"
  (cl:let ((__sec (cl:floor (cl:slot-value msg 'stamp)))
        (__nsec (cl:round (cl:* 1e9 (cl:- (cl:slot-value msg 'stamp) (cl:floor (cl:slot-value msg 'stamp)))))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 0) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __nsec) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'color_code_256)) ostream)
  (cl:let* ((signed (cl:slot-value msg 'severity)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Message>) istream)
  "Deserializes a message object of type '<Message>"
    (cl:let ((__sec 0) (__nsec 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 0) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __nsec) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'stamp) (cl:+ (cl:coerce __sec 'cl:double-float) (cl:/ __nsec 1e9))))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'color_code_256)) (cl:read-byte istream))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'severity) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Message>)))
  "Returns string type for a message object of type '<Message>"
  "tauv_msgs/Message")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Message)))
  "Returns string type for a message object of type 'Message"
  "tauv_msgs/Message")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Message>)))
  "Returns md5sum for a message object of type '<Message>"
  "9553ea1e0a9cfe897226edaceb34218e")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Message)))
  "Returns md5sum for a message object of type 'Message"
  "9553ea1e0a9cfe897226edaceb34218e")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Message>)))
  "Returns full string definition for message of type '<Message>"
  (cl:format cl:nil "int8 ERROR=0~%int8 WARNING=1~%int8 INFO=2~%int8 DEBUG=3~%~%time stamp         # time stamp of the message~%uint8 color_code_256   # color code of the message~%int8 severity       # severity level~%string message      # message~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Message)))
  "Returns full string definition for message of type 'Message"
  (cl:format cl:nil "int8 ERROR=0~%int8 WARNING=1~%int8 INFO=2~%int8 DEBUG=3~%~%time stamp         # time stamp of the message~%uint8 color_code_256   # color code of the message~%int8 severity       # severity level~%string message      # message~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Message>))
  (cl:+ 0
     8
     1
     1
     4 (cl:length (cl:slot-value msg 'message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Message>))
  "Converts a ROS message object to a list"
  (cl:list 'Message
    (cl:cons ':stamp (stamp msg))
    (cl:cons ':color_code_256 (color_code_256 msg))
    (cl:cons ':severity (severity msg))
    (cl:cons ':message (message msg))
))
