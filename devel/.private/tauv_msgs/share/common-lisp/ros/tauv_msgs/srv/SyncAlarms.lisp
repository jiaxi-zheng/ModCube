; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-srv)


;//! \htmlinclude SyncAlarms-request.msg.html

(cl:defclass <SyncAlarms-request> (roslisp-msg-protocol:ros-message)
  ((diff
    :reader diff
    :initarg :diff
    :type (cl:vector tauv_msgs-msg:AlarmWithMessage)
   :initform (cl:make-array 0 :element-type 'tauv_msgs-msg:AlarmWithMessage :initial-element (cl:make-instance 'tauv_msgs-msg:AlarmWithMessage))))
)

(cl:defclass SyncAlarms-request (<SyncAlarms-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SyncAlarms-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SyncAlarms-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<SyncAlarms-request> is deprecated: use tauv_msgs-srv:SyncAlarms-request instead.")))

(cl:ensure-generic-function 'diff-val :lambda-list '(m))
(cl:defmethod diff-val ((m <SyncAlarms-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:diff-val is deprecated.  Use tauv_msgs-srv:diff instead.")
  (diff m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SyncAlarms-request>) ostream)
  "Serializes a message object of type '<SyncAlarms-request>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'diff))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'diff))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SyncAlarms-request>) istream)
  "Deserializes a message object of type '<SyncAlarms-request>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'diff) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'diff)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'tauv_msgs-msg:AlarmWithMessage))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SyncAlarms-request>)))
  "Returns string type for a service object of type '<SyncAlarms-request>"
  "tauv_msgs/SyncAlarmsRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SyncAlarms-request)))
  "Returns string type for a service object of type 'SyncAlarms-request"
  "tauv_msgs/SyncAlarmsRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SyncAlarms-request>)))
  "Returns md5sum for a message object of type '<SyncAlarms-request>"
  "54b1739021e723bf57d59b6622adc3ef")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SyncAlarms-request)))
  "Returns md5sum for a message object of type 'SyncAlarms-request"
  "54b1739021e723bf57d59b6622adc3ef")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SyncAlarms-request>)))
  "Returns full string definition for message of type '<SyncAlarms-request>"
  (cl:format cl:nil "# Note: Angular velocities outside of yaw (z axis) are currently unused.~%~%tauv_msgs/AlarmWithMessage[] diff~%~%~%================================================================================~%MSG: tauv_msgs/AlarmWithMessage~%int32 id            # ID of the alarm~%bool set            # True = set, False = Cleared~%string message      # Readable message~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SyncAlarms-request)))
  "Returns full string definition for message of type 'SyncAlarms-request"
  (cl:format cl:nil "# Note: Angular velocities outside of yaw (z axis) are currently unused.~%~%tauv_msgs/AlarmWithMessage[] diff~%~%~%================================================================================~%MSG: tauv_msgs/AlarmWithMessage~%int32 id            # ID of the alarm~%bool set            # True = set, False = Cleared~%string message      # Readable message~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SyncAlarms-request>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'diff) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SyncAlarms-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SyncAlarms-request
    (cl:cons ':diff (diff msg))
))
;//! \htmlinclude SyncAlarms-response.msg.html

(cl:defclass <SyncAlarms-response> (roslisp-msg-protocol:ros-message)
  ((stamp
    :reader stamp
    :initarg :stamp
    :type cl:real
    :initform 0)
   (active_alarms
    :reader active_alarms
    :initarg :active_alarms
    :type (cl:vector cl:integer)
   :initform (cl:make-array 0 :element-type 'cl:integer :initial-element 0))
   (success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass SyncAlarms-response (<SyncAlarms-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SyncAlarms-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SyncAlarms-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<SyncAlarms-response> is deprecated: use tauv_msgs-srv:SyncAlarms-response instead.")))

(cl:ensure-generic-function 'stamp-val :lambda-list '(m))
(cl:defmethod stamp-val ((m <SyncAlarms-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:stamp-val is deprecated.  Use tauv_msgs-srv:stamp instead.")
  (stamp m))

(cl:ensure-generic-function 'active_alarms-val :lambda-list '(m))
(cl:defmethod active_alarms-val ((m <SyncAlarms-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:active_alarms-val is deprecated.  Use tauv_msgs-srv:active_alarms instead.")
  (active_alarms m))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <SyncAlarms-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:success-val is deprecated.  Use tauv_msgs-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SyncAlarms-response>) ostream)
  "Serializes a message object of type '<SyncAlarms-response>"
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
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SyncAlarms-response>) istream)
  "Deserializes a message object of type '<SyncAlarms-response>"
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
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SyncAlarms-response>)))
  "Returns string type for a service object of type '<SyncAlarms-response>"
  "tauv_msgs/SyncAlarmsResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SyncAlarms-response)))
  "Returns string type for a service object of type 'SyncAlarms-response"
  "tauv_msgs/SyncAlarmsResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SyncAlarms-response>)))
  "Returns md5sum for a message object of type '<SyncAlarms-response>"
  "54b1739021e723bf57d59b6622adc3ef")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SyncAlarms-response)))
  "Returns md5sum for a message object of type 'SyncAlarms-response"
  "54b1739021e723bf57d59b6622adc3ef")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SyncAlarms-response>)))
  "Returns full string definition for message of type '<SyncAlarms-response>"
  (cl:format cl:nil "~%time stamp~%int32[] active_alarms~%bool success  # false indicates some sort of failure~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SyncAlarms-response)))
  "Returns full string definition for message of type 'SyncAlarms-response"
  (cl:format cl:nil "~%time stamp~%int32[] active_alarms~%bool success  # false indicates some sort of failure~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SyncAlarms-response>))
  (cl:+ 0
     8
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'active_alarms) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SyncAlarms-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SyncAlarms-response
    (cl:cons ':stamp (stamp msg))
    (cl:cons ':active_alarms (active_alarms msg))
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SyncAlarms)))
  'SyncAlarms-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SyncAlarms)))
  'SyncAlarms-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SyncAlarms)))
  "Returns string type for a service object of type '<SyncAlarms>"
  "tauv_msgs/SyncAlarms")