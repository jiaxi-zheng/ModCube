; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-srv)


;//! \htmlinclude FeatureDetectionsSync-request.msg.html

(cl:defclass <FeatureDetectionsSync-request> (roslisp-msg-protocol:ros-message)
  ((detections
    :reader detections
    :initarg :detections
    :type tauv_msgs-msg:FeatureDetections
    :initform (cl:make-instance 'tauv_msgs-msg:FeatureDetections)))
)

(cl:defclass FeatureDetectionsSync-request (<FeatureDetectionsSync-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <FeatureDetectionsSync-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'FeatureDetectionsSync-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<FeatureDetectionsSync-request> is deprecated: use tauv_msgs-srv:FeatureDetectionsSync-request instead.")))

(cl:ensure-generic-function 'detections-val :lambda-list '(m))
(cl:defmethod detections-val ((m <FeatureDetectionsSync-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:detections-val is deprecated.  Use tauv_msgs-srv:detections instead.")
  (detections m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <FeatureDetectionsSync-request>) ostream)
  "Serializes a message object of type '<FeatureDetectionsSync-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'detections) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <FeatureDetectionsSync-request>) istream)
  "Deserializes a message object of type '<FeatureDetectionsSync-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'detections) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<FeatureDetectionsSync-request>)))
  "Returns string type for a service object of type '<FeatureDetectionsSync-request>"
  "tauv_msgs/FeatureDetectionsSyncRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'FeatureDetectionsSync-request)))
  "Returns string type for a service object of type 'FeatureDetectionsSync-request"
  "tauv_msgs/FeatureDetectionsSyncRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<FeatureDetectionsSync-request>)))
  "Returns md5sum for a message object of type '<FeatureDetectionsSync-request>"
  "2da451629f8ca3f1c8b6e832ba66c6c5")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'FeatureDetectionsSync-request)))
  "Returns md5sum for a message object of type 'FeatureDetectionsSync-request"
  "2da451629f8ca3f1c8b6e832ba66c6c5")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<FeatureDetectionsSync-request>)))
  "Returns full string definition for message of type '<FeatureDetectionsSync-request>"
  (cl:format cl:nil "tauv_msgs/FeatureDetections detections~%~%================================================================================~%MSG: tauv_msgs/FeatureDetections~%FeatureDetection[] detections~%string detector_tag~%================================================================================~%MSG: tauv_msgs/FeatureDetection~%Header header~%geometry_msgs/Point position #SE2 msgs will only use x y~%geometry_msgs/Point orientation #SE2 msgs will only use x (theta)~%string tag~%float64 confidence~%bool SE2~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'FeatureDetectionsSync-request)))
  "Returns full string definition for message of type 'FeatureDetectionsSync-request"
  (cl:format cl:nil "tauv_msgs/FeatureDetections detections~%~%================================================================================~%MSG: tauv_msgs/FeatureDetections~%FeatureDetection[] detections~%string detector_tag~%================================================================================~%MSG: tauv_msgs/FeatureDetection~%Header header~%geometry_msgs/Point position #SE2 msgs will only use x y~%geometry_msgs/Point orientation #SE2 msgs will only use x (theta)~%string tag~%float64 confidence~%bool SE2~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <FeatureDetectionsSync-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'detections))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <FeatureDetectionsSync-request>))
  "Converts a ROS message object to a list"
  (cl:list 'FeatureDetectionsSync-request
    (cl:cons ':detections (detections msg))
))
;//! \htmlinclude FeatureDetectionsSync-response.msg.html

(cl:defclass <FeatureDetectionsSync-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass FeatureDetectionsSync-response (<FeatureDetectionsSync-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <FeatureDetectionsSync-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'FeatureDetectionsSync-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<FeatureDetectionsSync-response> is deprecated: use tauv_msgs-srv:FeatureDetectionsSync-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <FeatureDetectionsSync-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:success-val is deprecated.  Use tauv_msgs-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <FeatureDetectionsSync-response>) ostream)
  "Serializes a message object of type '<FeatureDetectionsSync-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <FeatureDetectionsSync-response>) istream)
  "Deserializes a message object of type '<FeatureDetectionsSync-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<FeatureDetectionsSync-response>)))
  "Returns string type for a service object of type '<FeatureDetectionsSync-response>"
  "tauv_msgs/FeatureDetectionsSyncResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'FeatureDetectionsSync-response)))
  "Returns string type for a service object of type 'FeatureDetectionsSync-response"
  "tauv_msgs/FeatureDetectionsSyncResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<FeatureDetectionsSync-response>)))
  "Returns md5sum for a message object of type '<FeatureDetectionsSync-response>"
  "2da451629f8ca3f1c8b6e832ba66c6c5")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'FeatureDetectionsSync-response)))
  "Returns md5sum for a message object of type 'FeatureDetectionsSync-response"
  "2da451629f8ca3f1c8b6e832ba66c6c5")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<FeatureDetectionsSync-response>)))
  "Returns full string definition for message of type '<FeatureDetectionsSync-response>"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'FeatureDetectionsSync-response)))
  "Returns full string definition for message of type 'FeatureDetectionsSync-response"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <FeatureDetectionsSync-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <FeatureDetectionsSync-response>))
  "Converts a ROS message object to a list"
  (cl:list 'FeatureDetectionsSync-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'FeatureDetectionsSync)))
  'FeatureDetectionsSync-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'FeatureDetectionsSync)))
  'FeatureDetectionsSync-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'FeatureDetectionsSync)))
  "Returns string type for a service object of type '<FeatureDetectionsSync>"
  "tauv_msgs/FeatureDetectionsSync")