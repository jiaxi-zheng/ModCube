; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-srv)


;//! \htmlinclude TunePIDPlanner-request.msg.html

(cl:defclass <TunePIDPlanner-request> (roslisp-msg-protocol:ros-message)
  ((tunings
    :reader tunings
    :initarg :tunings
    :type (cl:vector tauv_msgs-msg:PIDTuning)
   :initform (cl:make-array 0 :element-type 'tauv_msgs-msg:PIDTuning :initial-element (cl:make-instance 'tauv_msgs-msg:PIDTuning))))
)

(cl:defclass TunePIDPlanner-request (<TunePIDPlanner-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TunePIDPlanner-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TunePIDPlanner-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<TunePIDPlanner-request> is deprecated: use tauv_msgs-srv:TunePIDPlanner-request instead.")))

(cl:ensure-generic-function 'tunings-val :lambda-list '(m))
(cl:defmethod tunings-val ((m <TunePIDPlanner-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:tunings-val is deprecated.  Use tauv_msgs-srv:tunings instead.")
  (tunings m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TunePIDPlanner-request>) ostream)
  "Serializes a message object of type '<TunePIDPlanner-request>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'tunings))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'tunings))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TunePIDPlanner-request>) istream)
  "Deserializes a message object of type '<TunePIDPlanner-request>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TunePIDPlanner-request>)))
  "Returns string type for a service object of type '<TunePIDPlanner-request>"
  "tauv_msgs/TunePIDPlannerRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TunePIDPlanner-request)))
  "Returns string type for a service object of type 'TunePIDPlanner-request"
  "tauv_msgs/TunePIDPlannerRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TunePIDPlanner-request>)))
  "Returns md5sum for a message object of type '<TunePIDPlanner-request>"
  "c6a95158ee66091a3c801f9968586b2d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TunePIDPlanner-request)))
  "Returns md5sum for a message object of type 'TunePIDPlanner-request"
  "c6a95158ee66091a3c801f9968586b2d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TunePIDPlanner-request>)))
  "Returns full string definition for message of type '<TunePIDPlanner-request>"
  (cl:format cl:nil "tauv_msgs/PIDTuning[] tunings~%~%================================================================================~%MSG: tauv_msgs/PIDTuning~%string axis~%float64 kp~%float64 ki~%float64 kd~%float64 tau~%float64[2] limits~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TunePIDPlanner-request)))
  "Returns full string definition for message of type 'TunePIDPlanner-request"
  (cl:format cl:nil "tauv_msgs/PIDTuning[] tunings~%~%================================================================================~%MSG: tauv_msgs/PIDTuning~%string axis~%float64 kp~%float64 ki~%float64 kd~%float64 tau~%float64[2] limits~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TunePIDPlanner-request>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'tunings) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TunePIDPlanner-request>))
  "Converts a ROS message object to a list"
  (cl:list 'TunePIDPlanner-request
    (cl:cons ':tunings (tunings msg))
))
;//! \htmlinclude TunePIDPlanner-response.msg.html

(cl:defclass <TunePIDPlanner-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass TunePIDPlanner-response (<TunePIDPlanner-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TunePIDPlanner-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TunePIDPlanner-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<TunePIDPlanner-response> is deprecated: use tauv_msgs-srv:TunePIDPlanner-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <TunePIDPlanner-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:success-val is deprecated.  Use tauv_msgs-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TunePIDPlanner-response>) ostream)
  "Serializes a message object of type '<TunePIDPlanner-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TunePIDPlanner-response>) istream)
  "Deserializes a message object of type '<TunePIDPlanner-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TunePIDPlanner-response>)))
  "Returns string type for a service object of type '<TunePIDPlanner-response>"
  "tauv_msgs/TunePIDPlannerResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TunePIDPlanner-response)))
  "Returns string type for a service object of type 'TunePIDPlanner-response"
  "tauv_msgs/TunePIDPlannerResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TunePIDPlanner-response>)))
  "Returns md5sum for a message object of type '<TunePIDPlanner-response>"
  "c6a95158ee66091a3c801f9968586b2d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TunePIDPlanner-response)))
  "Returns md5sum for a message object of type 'TunePIDPlanner-response"
  "c6a95158ee66091a3c801f9968586b2d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TunePIDPlanner-response>)))
  "Returns full string definition for message of type '<TunePIDPlanner-response>"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TunePIDPlanner-response)))
  "Returns full string definition for message of type 'TunePIDPlanner-response"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TunePIDPlanner-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TunePIDPlanner-response>))
  "Converts a ROS message object to a list"
  (cl:list 'TunePIDPlanner-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'TunePIDPlanner)))
  'TunePIDPlanner-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'TunePIDPlanner)))
  'TunePIDPlanner-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TunePIDPlanner)))
  "Returns string type for a service object of type '<TunePIDPlanner>"
  "tauv_msgs/TunePIDPlanner")