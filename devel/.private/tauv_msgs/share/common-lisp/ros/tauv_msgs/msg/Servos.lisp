; Auto-generated. Do not edit!


(cl:in-package tauv_msgs-msg)


;//! \htmlinclude Servos.msg.html

(cl:defclass <Servos> (roslisp-msg-protocol:ros-message)
  ((targets
    :reader targets
    :initarg :targets
    :type (cl:vector cl:integer)
   :initform (cl:make-array 4 :element-type 'cl:integer :initial-element 0)))
)

(cl:defclass Servos (<Servos>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Servos>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Servos)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tauv_msgs-msg:<Servos> is deprecated: use tauv_msgs-msg:Servos instead.")))

(cl:ensure-generic-function 'targets-val :lambda-list '(m))
(cl:defmethod targets-val ((m <Servos>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tauv_msgs-msg:targets-val is deprecated.  Use tauv_msgs-msg:targets instead.")
  (targets m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Servos>) ostream)
  "Serializes a message object of type '<Servos>"
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 18446744073709551616) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) unsigned) ostream)
    ))
   (cl:slot-value msg 'targets))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Servos>) istream)
  "Deserializes a message object of type '<Servos>"
  (cl:setf (cl:slot-value msg 'targets) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'targets)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 9223372036854775808) unsigned (cl:- unsigned 18446744073709551616))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Servos>)))
  "Returns string type for a message object of type '<Servos>"
  "tauv_msgs/Servos")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Servos)))
  "Returns string type for a message object of type 'Servos"
  "tauv_msgs/Servos")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Servos>)))
  "Returns md5sum for a message object of type '<Servos>"
  "e8f52576b380ad554e7b8eee4133713d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Servos)))
  "Returns md5sum for a message object of type 'Servos"
  "e8f52576b380ad554e7b8eee4133713d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Servos>)))
  "Returns full string definition for message of type '<Servos>"
  (cl:format cl:nil "int64[4] targets~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Servos)))
  "Returns full string definition for message of type 'Servos"
  (cl:format cl:nil "int64[4] targets~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Servos>))
  (cl:+ 0
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'targets) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Servos>))
  "Converts a ROS message object to a list"
  (cl:list 'Servos
    (cl:cons ':targets (targets msg))
))
