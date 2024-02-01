; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude AlarmReport.msg.html

(cl:defclass <AlarmReport> (roslisp-msg-protocol:ros-message)
  ((stamp
    :reader stamp
    :initarg :stamp
    :type cl:real
    :initform 0)
   (active_alarms
    :reader active_alarms
    :initarg :active_alarms
    :type (cl:vector cl:integer)
   :initform (cl:make-array 0 :element-type 'cl:integer :initial-element 0)))
)

(cl:defclass AlarmReport (<AlarmReport>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <AlarmReport>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'AlarmReport)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<AlarmReport> is deprecated: use tauv_msgs-msg:AlarmReport instead.")))

(cl:ensure-generic-function 'stamp-val :lambda-list '(m))
(cl:defmethod stamp-val ((m <AlarmReport>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:stamp-val is deprecated.  Use tauv_msgs-msg:stamp instead.")
  (stamp m))

(cl:ensure-generic-function 'active_alarms-val :lambda-list '(m))
(cl:defmethod active_alarms-val ((m <AlarmReport>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:active_alarms-val is deprecated.  Use tauv_msgs-msg:active_alarms instead.")
  (active_alarms m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <AlarmReport>) ostream)
  "Serializes a message object of type '<AlarmReport>"
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
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'active_alarms))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    ))
   (cl:slot-value msg 'active_alarms))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <AlarmReport>) istream)
  "Deserializes a message object of type '<AlarmReport>"
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
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'active_alarms) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'active_alarms)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296)))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<AlarmReport>)))
  "Returns string type for a message object of type '<AlarmReport>"
  "tauv_msgs/AlarmReport")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'AlarmReport)))
  "Returns string type for a message object of type 'AlarmReport"
  "tauv_msgs/AlarmReport")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<AlarmReport>)))
  "Returns md5sum for a message object of type '<AlarmReport>"
  "6041271f37a12a54ca5b8c77ba39eab9")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'AlarmReport)))
  "Returns md5sum for a message object of type 'AlarmReport"
  "6041271f37a12a54ca5b8c77ba39eab9")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<AlarmReport>)))
  "Returns full string definition for message of type '<AlarmReport>"
  (cl:format cl:nil "time stamp~%int32[] active_alarms~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'AlarmReport)))
  "Returns full string definition for message of type 'AlarmReport"
  (cl:format cl:nil "time stamp~%int32[] active_alarms~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <AlarmReport>))
  (cl:+ 0
     8
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'active_alarms) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <AlarmReport>))
  "Converts a ROS message object to a list"
  (cl:list 'AlarmReport
    (cl:cons ':stamp (stamp msg))
    (cl:cons ':active_alarms (active_alarms msg))
))
