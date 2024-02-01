; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-srv)


;//! \htmlinclude MapFindOne-request.msg.html

(cl:defclass <MapFindOne-request> (roslisp-msg-protocol:ros-message)
  ((tag
    :reader tag
    :initarg :tag
    :type cl:string
    :initform ""))
)

(cl:defclass MapFindOne-request (<MapFindOne-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <MapFindOne-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'MapFindOne-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<MapFindOne-request> is deprecated: use tauv_msgs-srv:MapFindOne-request instead.")))

(cl:ensure-generic-function 'tag-val :lambda-list '(m))
(cl:defmethod tag-val ((m <MapFindOne-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:tag-val is deprecated.  Use tauv_msgs-srv:tag instead.")
  (tag m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <MapFindOne-request>) ostream)
  "Serializes a message object of type '<MapFindOne-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'tag))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'tag))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <MapFindOne-request>) istream)
  "Deserializes a message object of type '<MapFindOne-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'tag) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'tag) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<MapFindOne-request>)))
  "Returns string type for a service object of type '<MapFindOne-request>"
  "tauv_msgs/MapFindOneRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MapFindOne-request)))
  "Returns string type for a service object of type 'MapFindOne-request"
  "tauv_msgs/MapFindOneRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<MapFindOne-request>)))
  "Returns md5sum for a message object of type '<MapFindOne-request>"
  "0b9e59f5c45dc6444e155df33a798059")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'MapFindOne-request)))
  "Returns md5sum for a message object of type 'MapFindOne-request"
  "0b9e59f5c45dc6444e155df33a798059")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<MapFindOne-request>)))
  "Returns full string definition for message of type '<MapFindOne-request>"
  (cl:format cl:nil "string tag~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'MapFindOne-request)))
  "Returns full string definition for message of type 'MapFindOne-request"
  (cl:format cl:nil "string tag~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <MapFindOne-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'tag))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <MapFindOne-request>))
  "Converts a ROS message object to a list"
  (cl:list 'MapFindOne-request
    (cl:cons ':tag (tag msg))
))
;//! \htmlinclude MapFindOne-response.msg.html

(cl:defclass <MapFindOne-response> (roslisp-msg-protocol:ros-message)
  ((detection
    :reader detection
    :initarg :detection
    :type tauv_msgs-msg:FeatureDetection
    :initform (cl:make-instance 'tauv_msgs-msg:FeatureDetection))
   (success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass MapFindOne-response (<MapFindOne-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <MapFindOne-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'MapFindOne-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<MapFindOne-response> is deprecated: use tauv_msgs-srv:MapFindOne-response instead.")))

(cl:ensure-generic-function 'detection-val :lambda-list '(m))
(cl:defmethod detection-val ((m <MapFindOne-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:detection-val is deprecated.  Use tauv_msgs-srv:detection instead.")
  (detection m))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <MapFindOne-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:success-val is deprecated.  Use tauv_msgs-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <MapFindOne-response>) ostream)
  "Serializes a message object of type '<MapFindOne-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'detection) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <MapFindOne-response>) istream)
  "Deserializes a message object of type '<MapFindOne-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'detection) istream)
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<MapFindOne-response>)))
  "Returns string type for a service object of type '<MapFindOne-response>"
  "tauv_msgs/MapFindOneResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MapFindOne-response)))
  "Returns string type for a service object of type 'MapFindOne-response"
  "tauv_msgs/MapFindOneResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<MapFindOne-response>)))
  "Returns md5sum for a message object of type '<MapFindOne-response>"
  "0b9e59f5c45dc6444e155df33a798059")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'MapFindOne-response)))
  "Returns md5sum for a message object of type 'MapFindOne-response"
  "0b9e59f5c45dc6444e155df33a798059")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<MapFindOne-response>)))
  "Returns full string definition for message of type '<MapFindOne-response>"
  (cl:format cl:nil "tauv_msgs/FeatureDetection detection~%bool success~%~%================================================================================~%MSG: tauv_msgs/FeatureDetection~%Header header~%geometry_msgs/Point position #SE2 msgs will only use x y~%geometry_msgs/Point orientation #SE2 msgs will only use x (theta)~%string tag~%float64 confidence~%bool SE2~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'MapFindOne-response)))
  "Returns full string definition for message of type 'MapFindOne-response"
  (cl:format cl:nil "tauv_msgs/FeatureDetection detection~%bool success~%~%================================================================================~%MSG: tauv_msgs/FeatureDetection~%Header header~%geometry_msgs/Point position #SE2 msgs will only use x y~%geometry_msgs/Point orientation #SE2 msgs will only use x (theta)~%string tag~%float64 confidence~%bool SE2~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <MapFindOne-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'detection))
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <MapFindOne-response>))
  "Converts a ROS message object to a list"
  (cl:list 'MapFindOne-response
    (cl:cons ':detection (detection msg))
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'MapFindOne)))
  'MapFindOne-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'MapFindOne)))
  'MapFindOne-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MapFindOne)))
  "Returns string type for a service object of type '<MapFindOne>"
  "tauv_msgs/MapFindOne")