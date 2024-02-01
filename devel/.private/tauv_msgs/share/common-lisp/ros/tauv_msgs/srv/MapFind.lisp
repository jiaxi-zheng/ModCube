; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-srv)


;//! \htmlinclude MapFind-request.msg.html

(cl:defclass <MapFind-request> (roslisp-msg-protocol:ros-message)
  ((tag
    :reader tag
    :initarg :tag
    :type cl:string
    :initform ""))
)

(cl:defclass MapFind-request (<MapFind-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <MapFind-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'MapFind-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<MapFind-request> is deprecated: use tauv_msgs-srv:MapFind-request instead.")))

(cl:ensure-generic-function 'tag-val :lambda-list '(m))
(cl:defmethod tag-val ((m <MapFind-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:tag-val is deprecated.  Use tauv_msgs-srv:tag instead.")
  (tag m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <MapFind-request>) ostream)
  "Serializes a message object of type '<MapFind-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'tag))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'tag))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <MapFind-request>) istream)
  "Deserializes a message object of type '<MapFind-request>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<MapFind-request>)))
  "Returns string type for a service object of type '<MapFind-request>"
  "tauv_msgs/MapFindRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MapFind-request)))
  "Returns string type for a service object of type 'MapFind-request"
  "tauv_msgs/MapFindRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<MapFind-request>)))
  "Returns md5sum for a message object of type '<MapFind-request>"
  "c4b23e80f3f361fabb0b381682792779")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'MapFind-request)))
  "Returns md5sum for a message object of type 'MapFind-request"
  "c4b23e80f3f361fabb0b381682792779")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<MapFind-request>)))
  "Returns full string definition for message of type '<MapFind-request>"
  (cl:format cl:nil "string tag~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'MapFind-request)))
  "Returns full string definition for message of type 'MapFind-request"
  (cl:format cl:nil "string tag~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <MapFind-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'tag))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <MapFind-request>))
  "Converts a ROS message object to a list"
  (cl:list 'MapFind-request
    (cl:cons ':tag (tag msg))
))
;//! \htmlinclude MapFind-response.msg.html

(cl:defclass <MapFind-response> (roslisp-msg-protocol:ros-message)
  ((detections
    :reader detections
    :initarg :detections
    :type (cl:vector tauv_msgs-msg:FeatureDetection)
   :initform (cl:make-array 0 :element-type 'tauv_msgs-msg:FeatureDetection :initial-element (cl:make-instance 'tauv_msgs-msg:FeatureDetection)))
   (success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass MapFind-response (<MapFind-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <MapFind-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'MapFind-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<MapFind-response> is deprecated: use tauv_msgs-srv:MapFind-response instead.")))

(cl:ensure-generic-function 'detections-val :lambda-list '(m))
(cl:defmethod detections-val ((m <MapFind-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:detections-val is deprecated.  Use tauv_msgs-srv:detections instead.")
  (detections m))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <MapFind-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:success-val is deprecated.  Use tauv_msgs-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <MapFind-response>) ostream)
  "Serializes a message object of type '<MapFind-response>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'detections))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'detections))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <MapFind-response>) istream)
  "Deserializes a message object of type '<MapFind-response>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'detections) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'detections)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'tauv_msgs-msg:FeatureDetection))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<MapFind-response>)))
  "Returns string type for a service object of type '<MapFind-response>"
  "tauv_msgs/MapFindResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MapFind-response)))
  "Returns string type for a service object of type 'MapFind-response"
  "tauv_msgs/MapFindResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<MapFind-response>)))
  "Returns md5sum for a message object of type '<MapFind-response>"
  "c4b23e80f3f361fabb0b381682792779")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'MapFind-response)))
  "Returns md5sum for a message object of type 'MapFind-response"
  "c4b23e80f3f361fabb0b381682792779")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<MapFind-response>)))
  "Returns full string definition for message of type '<MapFind-response>"
  (cl:format cl:nil "tauv_msgs/FeatureDetection[] detections~%bool success~%~%================================================================================~%MSG: tauv_msgs/FeatureDetection~%Header header~%geometry_msgs/Point position #SE2 msgs will only use x y~%geometry_msgs/Point orientation #SE2 msgs will only use x (theta)~%string tag~%float64 confidence~%bool SE2~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'MapFind-response)))
  "Returns full string definition for message of type 'MapFind-response"
  (cl:format cl:nil "tauv_msgs/FeatureDetection[] detections~%bool success~%~%================================================================================~%MSG: tauv_msgs/FeatureDetection~%Header header~%geometry_msgs/Point position #SE2 msgs will only use x y~%geometry_msgs/Point orientation #SE2 msgs will only use x (theta)~%string tag~%float64 confidence~%bool SE2~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <MapFind-response>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'detections) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <MapFind-response>))
  "Converts a ROS message object to a list"
  (cl:list 'MapFind-response
    (cl:cons ':detections (detections msg))
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'MapFind)))
  'MapFind-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'MapFind)))
  'MapFind-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MapFind)))
  "Returns string type for a service object of type '<MapFind>"
  "tauv_msgs/MapFind")