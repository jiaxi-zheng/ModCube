; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-srv)


;//! \htmlinclude TuneController-request.msg.html

(cl:defclass <TuneController-request> (roslisp-msg-protocol:ros-message)
  ((tunings
    :reader tunings
    :initarg :tunings
    :type (cl:vector tauv_msgs-msg:PIDTuning)
   :initform (cl:make-array 0 :element-type 'tauv_msgs-msg:PIDTuning :initial-element (cl:make-instance 'tauv_msgs-msg:PIDTuning))))
)

(cl:defclass TuneController-request (<TuneController-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TuneController-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TuneController-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<TuneController-request> is deprecated: use tauv_msgs-srv:TuneController-request instead.")))

(cl:ensure-generic-function 'tunings-val :lambda-list '(m))
(cl:defmethod tunings-val ((m <TuneController-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:tunings-val is deprecated.  Use tauv_msgs-srv:tunings instead.")
  (tunings m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TuneController-request>) ostream)
  "Serializes a message object of type '<TuneController-request>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'tunings))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'tunings))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TuneController-request>) istream)
  "Deserializes a message object of type '<TuneController-request>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'tunings) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'tunings)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'tauv_msgs-msg:PIDTuning))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TuneController-request>)))
  "Returns string type for a service object of type '<TuneController-request>"
  "tauv_msgs/TuneControllerRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TuneController-request)))
  "Returns string type for a service object of type 'TuneController-request"
  "tauv_msgs/TuneControllerRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TuneController-request>)))
  "Returns md5sum for a message object of type '<TuneController-request>"
  "c6a95158ee66091a3c801f9968586b2d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TuneController-request)))
  "Returns md5sum for a message object of type 'TuneController-request"
  "c6a95158ee66091a3c801f9968586b2d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TuneController-request>)))
  "Returns full string definition for message of type '<TuneController-request>"
  (cl:format cl:nil "tauv_msgs/PIDTuning[] tunings~%~%================================================================================~%MSG: tauv_msgs/PIDTuning~%string axis~%float64 kp~%float64 ki~%float64 kd~%float64 tau~%float64[2] limits~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TuneController-request)))
  "Returns full string definition for message of type 'TuneController-request"
  (cl:format cl:nil "tauv_msgs/PIDTuning[] tunings~%~%================================================================================~%MSG: tauv_msgs/PIDTuning~%string axis~%float64 kp~%float64 ki~%float64 kd~%float64 tau~%float64[2] limits~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TuneController-request>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'tunings) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TuneController-request>))
  "Converts a ROS message object to a list"
  (cl:list 'TuneController-request
    (cl:cons ':tunings (tunings msg))
))
;//! \htmlinclude TuneController-response.msg.html

(cl:defclass <TuneController-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass TuneController-response (<TuneController-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TuneController-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TuneController-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<TuneController-response> is deprecated: use tauv_msgs-srv:TuneController-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <TuneController-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:success-val is deprecated.  Use tauv_msgs-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TuneController-response>) ostream)
  "Serializes a message object of type '<TuneController-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TuneController-response>) istream)
  "Deserializes a message object of type '<TuneController-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TuneController-response>)))
  "Returns string type for a service object of type '<TuneController-response>"
  "tauv_msgs/TuneControllerResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TuneController-response)))
  "Returns string type for a service object of type 'TuneController-response"
  "tauv_msgs/TuneControllerResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TuneController-response>)))
  "Returns md5sum for a message object of type '<TuneController-response>"
  "c6a95158ee66091a3c801f9968586b2d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TuneController-response)))
  "Returns md5sum for a message object of type 'TuneController-response"
  "c6a95158ee66091a3c801f9968586b2d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TuneController-response>)))
  "Returns full string definition for message of type '<TuneController-response>"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TuneController-response)))
  "Returns full string definition for message of type 'TuneController-response"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TuneController-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TuneController-response>))
  "Converts a ROS message object to a list"
  (cl:list 'TuneController-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'TuneController)))
  'TuneController-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'TuneController)))
  'TuneController-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TuneController)))
  "Returns string type for a service object of type '<TuneController>"
  "tauv_msgs/TuneController")