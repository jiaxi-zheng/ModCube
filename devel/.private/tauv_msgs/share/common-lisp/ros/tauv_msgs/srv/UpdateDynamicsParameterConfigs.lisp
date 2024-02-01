; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-srv)


;//! \htmlinclude UpdateDynamicsParameterConfigs-request.msg.html

(cl:defclass <UpdateDynamicsParameterConfigs-request> (roslisp-msg-protocol:ros-message)
  ((updates
    :reader updates
    :initarg :updates
    :type (cl:vector tauv_msgs-msg:DynamicsParameterConfigUpdate)
   :initform (cl:make-array 0 :element-type 'tauv_msgs-msg:DynamicsParameterConfigUpdate :initial-element (cl:make-instance 'tauv_msgs-msg:DynamicsParameterConfigUpdate))))
)

(cl:defclass UpdateDynamicsParameterConfigs-request (<UpdateDynamicsParameterConfigs-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <UpdateDynamicsParameterConfigs-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'UpdateDynamicsParameterConfigs-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<UpdateDynamicsParameterConfigs-request> is deprecated: use tauv_msgs-srv:UpdateDynamicsParameterConfigs-request instead.")))

(cl:ensure-generic-function 'updates-val :lambda-list '(m))
(cl:defmethod updates-val ((m <UpdateDynamicsParameterConfigs-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:updates-val is deprecated.  Use tauv_msgs-srv:updates instead.")
  (updates m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <UpdateDynamicsParameterConfigs-request>) ostream)
  "Serializes a message object of type '<UpdateDynamicsParameterConfigs-request>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'updates))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'updates))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <UpdateDynamicsParameterConfigs-request>) istream)
  "Deserializes a message object of type '<UpdateDynamicsParameterConfigs-request>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'updates) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'updates)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'tauv_msgs-msg:DynamicsParameterConfigUpdate))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<UpdateDynamicsParameterConfigs-request>)))
  "Returns string type for a service object of type '<UpdateDynamicsParameterConfigs-request>"
  "tauv_msgs/UpdateDynamicsParameterConfigsRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UpdateDynamicsParameterConfigs-request)))
  "Returns string type for a service object of type 'UpdateDynamicsParameterConfigs-request"
  "tauv_msgs/UpdateDynamicsParameterConfigsRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<UpdateDynamicsParameterConfigs-request>)))
  "Returns md5sum for a message object of type '<UpdateDynamicsParameterConfigs-request>"
  "613592a768bbac2700fe9ff9c5b0cffe")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'UpdateDynamicsParameterConfigs-request)))
  "Returns md5sum for a message object of type 'UpdateDynamicsParameterConfigs-request"
  "613592a768bbac2700fe9ff9c5b0cffe")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<UpdateDynamicsParameterConfigs-request>)))
  "Returns full string definition for message of type '<UpdateDynamicsParameterConfigs-request>"
  (cl:format cl:nil "tauv_msgs/DynamicsParameterConfigUpdate[] updates~%~%================================================================================~%MSG: tauv_msgs/DynamicsParameterConfigUpdate~%string name~%~%bool update_initial_value~%float64 initial_value~%~%bool update_fixed~%bool fixed~%~%bool update_initial_covariance~%float64 initial_covariance~%~%bool update_process_covariance~%float64 process_covariance~%~%bool update_limits~%float64[2] limits~%~%bool reset~%~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'UpdateDynamicsParameterConfigs-request)))
  "Returns full string definition for message of type 'UpdateDynamicsParameterConfigs-request"
  (cl:format cl:nil "tauv_msgs/DynamicsParameterConfigUpdate[] updates~%~%================================================================================~%MSG: tauv_msgs/DynamicsParameterConfigUpdate~%string name~%~%bool update_initial_value~%float64 initial_value~%~%bool update_fixed~%bool fixed~%~%bool update_initial_covariance~%float64 initial_covariance~%~%bool update_process_covariance~%float64 process_covariance~%~%bool update_limits~%float64[2] limits~%~%bool reset~%~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <UpdateDynamicsParameterConfigs-request>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'updates) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <UpdateDynamicsParameterConfigs-request>))
  "Converts a ROS message object to a list"
  (cl:list 'UpdateDynamicsParameterConfigs-request
    (cl:cons ':updates (updates msg))
))
;//! \htmlinclude UpdateDynamicsParameterConfigs-response.msg.html

(cl:defclass <UpdateDynamicsParameterConfigs-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass UpdateDynamicsParameterConfigs-response (<UpdateDynamicsParameterConfigs-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <UpdateDynamicsParameterConfigs-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'UpdateDynamicsParameterConfigs-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-srv:<UpdateDynamicsParameterConfigs-response> is deprecated: use tauv_msgs-srv:UpdateDynamicsParameterConfigs-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <UpdateDynamicsParameterConfigs-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-srv:success-val is deprecated.  Use tauv_msgs-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <UpdateDynamicsParameterConfigs-response>) ostream)
  "Serializes a message object of type '<UpdateDynamicsParameterConfigs-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <UpdateDynamicsParameterConfigs-response>) istream)
  "Deserializes a message object of type '<UpdateDynamicsParameterConfigs-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<UpdateDynamicsParameterConfigs-response>)))
  "Returns string type for a service object of type '<UpdateDynamicsParameterConfigs-response>"
  "tauv_msgs/UpdateDynamicsParameterConfigsResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UpdateDynamicsParameterConfigs-response)))
  "Returns string type for a service object of type 'UpdateDynamicsParameterConfigs-response"
  "tauv_msgs/UpdateDynamicsParameterConfigsResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<UpdateDynamicsParameterConfigs-response>)))
  "Returns md5sum for a message object of type '<UpdateDynamicsParameterConfigs-response>"
  "613592a768bbac2700fe9ff9c5b0cffe")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'UpdateDynamicsParameterConfigs-response)))
  "Returns md5sum for a message object of type 'UpdateDynamicsParameterConfigs-response"
  "613592a768bbac2700fe9ff9c5b0cffe")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<UpdateDynamicsParameterConfigs-response>)))
  "Returns full string definition for message of type '<UpdateDynamicsParameterConfigs-response>"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'UpdateDynamicsParameterConfigs-response)))
  "Returns full string definition for message of type 'UpdateDynamicsParameterConfigs-response"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <UpdateDynamicsParameterConfigs-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <UpdateDynamicsParameterConfigs-response>))
  "Converts a ROS message object to a list"
  (cl:list 'UpdateDynamicsParameterConfigs-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'UpdateDynamicsParameterConfigs)))
  'UpdateDynamicsParameterConfigs-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'UpdateDynamicsParameterConfigs)))
  'UpdateDynamicsParameterConfigs-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'UpdateDynamicsParameterConfigs)))
  "Returns string type for a service object of type '<UpdateDynamicsParameterConfigs>"
  "tauv_msgs/UpdateDynamicsParameterConfigs")