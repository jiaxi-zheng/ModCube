; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude FeatureDetections.msg.html

(cl:defclass <FeatureDetections> (roslisp-msg-protocol:ros-message)
  ((detections
    :reader detections
    :initarg :detections
    :type (cl:vector tauv_msgs-msg:FeatureDetection)
   :initform (cl:make-array 0 :element-type 'tauv_msgs-msg:FeatureDetection :initial-element (cl:make-instance 'tauv_msgs-msg:FeatureDetection)))
   (detector_tag
    :reader detector_tag
    :initarg :detector_tag
    :type cl:string
    :initform ""))
)

(cl:defclass FeatureDetections (<FeatureDetections>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <FeatureDetections>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'FeatureDetections)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<FeatureDetections> is deprecated: use tauv_msgs-msg:FeatureDetections instead.")))

(cl:ensure-generic-function 'detections-val :lambda-list '(m))
(cl:defmethod detections-val ((m <FeatureDetections>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:detections-val is deprecated.  Use tauv_msgs-msg:detections instead.")
  (detections m))

(cl:ensure-generic-function 'detector_tag-val :lambda-list '(m))
(cl:defmethod detector_tag-val ((m <FeatureDetections>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:detector_tag-val is deprecated.  Use tauv_msgs-msg:detector_tag instead.")
  (detector_tag m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <FeatureDetections>) ostream)
  "Serializes a message object of type '<FeatureDetections>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'detections))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'detections))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'detector_tag))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'detector_tag))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <FeatureDetections>) istream)
  "Deserializes a message object of type '<FeatureDetections>"
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
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'detector_tag) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'detector_tag) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<FeatureDetections>)))
  "Returns string type for a message object of type '<FeatureDetections>"
  "tauv_msgs/FeatureDetections")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'FeatureDetections)))
  "Returns string type for a message object of type 'FeatureDetections"
  "tauv_msgs/FeatureDetections")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<FeatureDetections>)))
  "Returns md5sum for a message object of type '<FeatureDetections>"
  "b198f96e11b160e3e0b3e1f890a3f57d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'FeatureDetections)))
  "Returns md5sum for a message object of type 'FeatureDetections"
  "b198f96e11b160e3e0b3e1f890a3f57d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<FeatureDetections>)))
  "Returns full string definition for message of type '<FeatureDetections>"
  (cl:format cl:nil "FeatureDetection[] detections~%string detector_tag~%================================================================================~%MSG: tauv_msgs/FeatureDetection~%Header header~%geometry_msgs/Point position #SE2 msgs will only use x y~%geometry_msgs/Point orientation #SE2 msgs will only use x (theta)~%string tag~%float64 confidence~%bool SE2~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'FeatureDetections)))
  "Returns full string definition for message of type 'FeatureDetections"
  (cl:format cl:nil "FeatureDetection[] detections~%string detector_tag~%================================================================================~%MSG: tauv_msgs/FeatureDetection~%Header header~%geometry_msgs/Point position #SE2 msgs will only use x y~%geometry_msgs/Point orientation #SE2 msgs will only use x (theta)~%string tag~%float64 confidence~%bool SE2~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <FeatureDetections>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'detections) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:length (cl:slot-value msg 'detector_tag))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <FeatureDetections>))
  "Converts a ROS message object to a list"
  (cl:list 'FeatureDetections
    (cl:cons ':detections (detections msg))
    (cl:cons ':detector_tag (detector_tag msg))
))
