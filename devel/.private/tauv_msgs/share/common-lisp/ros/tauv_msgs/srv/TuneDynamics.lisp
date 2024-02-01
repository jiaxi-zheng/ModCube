; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-srv)


;//! \htmlinclude TuneDynamics-request.msg.html

(cl:defclass <TuneDynamics-request> (roslisp-msg-protocol:ros-message)
  ((tuning
    :reader tuning
    :initarg :tuning
    :type tauv_msgs-msg:DynamicsTuning
    :initform (cl:make-instance 'tauv_msgs-msg:DynamicsTuning)))
)

(cl:defclass TuneDynamics-request (<TuneDynamics-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TuneDynamics-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TuneDynamics-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<TuneDynamics-request> is deprecated: use tauv_msgs-srv:TuneDynamics-request instead.")))

(cl:ensure-generic-function 'tuning-val :lambda-list '(m))
(cl:defmethod tuning-val ((m <TuneDynamics-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:tuning-val is deprecated.  Use tauv_msgs-srv:tuning instead.")
  (tuning m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TuneDynamics-request>) ostream)
  "Serializes a message object of type '<TuneDynamics-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'tuning) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TuneDynamics-request>) istream)
  "Deserializes a message object of type '<TuneDynamics-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'tuning) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TuneDynamics-request>)))
  "Returns string type for a service object of type '<TuneDynamics-request>"
  "tauv_msgs/TuneDynamicsRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TuneDynamics-request)))
  "Returns string type for a service object of type 'TuneDynamics-request"
  "tauv_msgs/TuneDynamicsRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TuneDynamics-request>)))
  "Returns md5sum for a message object of type '<TuneDynamics-request>"
  "7d0bd5eafc7372a83c5502516a094dbc")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TuneDynamics-request)))
  "Returns md5sum for a message object of type 'TuneDynamics-request"
  "7d0bd5eafc7372a83c5502516a094dbc")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TuneDynamics-request>)))
  "Returns full string definition for message of type '<TuneDynamics-request>"
  (cl:format cl:nil "tauv_msgs/DynamicsTuning tuning~%~%================================================================================~%MSG: tauv_msgs/DynamicsTuning~%bool update_mass~%float64 mass~%bool update_volume~%float64 volume~%bool update_water_density~%float64 water_density~%bool update_center_of_gravity~%float64[3] center_of_gravity~%bool update_center_of_buoyancy~%float64[3] center_of_buoyancy~%bool update_moments~%float64[6] moments~%bool update_linear_damping~%float64[6] linear_damping~%bool update_quadratic_damping~%float64[6] quadratic_damping~%bool update_added_mass~%float64[6] added_mass~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TuneDynamics-request)))
  "Returns full string definition for message of type 'TuneDynamics-request"
  (cl:format cl:nil "tauv_msgs/DynamicsTuning tuning~%~%================================================================================~%MSG: tauv_msgs/DynamicsTuning~%bool update_mass~%float64 mass~%bool update_volume~%float64 volume~%bool update_water_density~%float64 water_density~%bool update_center_of_gravity~%float64[3] center_of_gravity~%bool update_center_of_buoyancy~%float64[3] center_of_buoyancy~%bool update_moments~%float64[6] moments~%bool update_linear_damping~%float64[6] linear_damping~%bool update_quadratic_damping~%float64[6] quadratic_damping~%bool update_added_mass~%float64[6] added_mass~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TuneDynamics-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'tuning))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TuneDynamics-request>))
  "Converts a ROS message object to a list"
  (cl:list 'TuneDynamics-request
    (cl:cons ':tuning (tuning msg))
))
;//! \htmlinclude TuneDynamics-response.msg.html

(cl:defclass <TuneDynamics-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass TuneDynamics-response (<TuneDynamics-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <TuneDynamics-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'TuneDynamics-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<TuneDynamics-response> is deprecated: use tauv_msgs-srv:TuneDynamics-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <TuneDynamics-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:success-val is deprecated.  Use tauv_msgs-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <TuneDynamics-response>) ostream)
  "Serializes a message object of type '<TuneDynamics-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <TuneDynamics-response>) istream)
  "Deserializes a message object of type '<TuneDynamics-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<TuneDynamics-response>)))
  "Returns string type for a service object of type '<TuneDynamics-response>"
  "tauv_msgs/TuneDynamicsResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TuneDynamics-response)))
  "Returns string type for a service object of type 'TuneDynamics-response"
  "tauv_msgs/TuneDynamicsResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<TuneDynamics-response>)))
  "Returns md5sum for a message object of type '<TuneDynamics-response>"
  "7d0bd5eafc7372a83c5502516a094dbc")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'TuneDynamics-response)))
  "Returns md5sum for a message object of type 'TuneDynamics-response"
  "7d0bd5eafc7372a83c5502516a094dbc")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<TuneDynamics-response>)))
  "Returns full string definition for message of type '<TuneDynamics-response>"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'TuneDynamics-response)))
  "Returns full string definition for message of type 'TuneDynamics-response"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <TuneDynamics-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <TuneDynamics-response>))
  "Converts a ROS message object to a list"
  (cl:list 'TuneDynamics-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'TuneDynamics)))
  'TuneDynamics-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'TuneDynamics)))
  'TuneDynamics-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'TuneDynamics)))
  "Returns string type for a service object of type '<TuneDynamics>"
  "tauv_msgs/TuneDynamics")