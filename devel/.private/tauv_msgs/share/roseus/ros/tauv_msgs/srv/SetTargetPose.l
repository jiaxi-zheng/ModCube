;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::SetTargetPose)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'SetTargetPose (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::SETTARGETPOSE")
  (make-package "TAUV_MSGS::SETTARGETPOSE"))
(unless (find-package "TAUV_MSGS::SETTARGETPOSEREQUEST")
  (make-package "TAUV_MSGS::SETTARGETPOSEREQUEST"))
(unless (find-package "TAUV_MSGS::SETTARGETPOSERESPONSE")
  (make-package "TAUV_MSGS::SETTARGETPOSERESPONSE"))

(in-package "ROS")

(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))




(defclass tauv_msgs::SetTargetPoseRequest
  :super ros::object
  :slots (_pose ))

(defmethod tauv_msgs::SetTargetPoseRequest
  (:init
   (&key
    ((:pose __pose) (instance geometry_msgs::Pose :init))
    )
   (send-super :init)
   (setq _pose __pose)
   self)
  (:pose
   (&rest __pose)
   (if (keywordp (car __pose))
       (send* _pose __pose)
     (progn
       (if __pose (setq _pose (car __pose)))
       _pose)))
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Pose _pose
    (send _pose :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Pose _pose
       (send _pose :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Pose _pose
     (send _pose :deserialize buf ptr-) (incf ptr- (send _pose :serialization-length))
   ;;
   self)
  )

(defclass tauv_msgs::SetTargetPoseResponse
  :super ros::object
  :slots (_success ))

(defmethod tauv_msgs::SetTargetPoseResponse
  (:init
   (&key
    ((:success __success) nil)
    )
   (send-super :init)
   (setq _success __success)
   self)
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:serialization-length
   ()
   (+
    ;; bool _success
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass tauv_msgs::SetTargetPose
  :super ros::object
  :slots ())

(setf (get tauv_msgs::SetTargetPose :md5sum-) "b66f4c8c2fa81b98544cf0f3e10950ee")
(setf (get tauv_msgs::SetTargetPose :datatype-) "tauv_msgs/SetTargetPose")
(setf (get tauv_msgs::SetTargetPose :request) tauv_msgs::SetTargetPoseRequest)
(setf (get tauv_msgs::SetTargetPose :response) tauv_msgs::SetTargetPoseResponse)

(defmethod tauv_msgs::SetTargetPoseRequest
  (:response () (instance tauv_msgs::SetTargetPoseResponse :init)))

(setf (get tauv_msgs::SetTargetPoseRequest :md5sum-) "b66f4c8c2fa81b98544cf0f3e10950ee")
(setf (get tauv_msgs::SetTargetPoseRequest :datatype-) "tauv_msgs/SetTargetPoseRequest")
(setf (get tauv_msgs::SetTargetPoseRequest :definition-)
      "geometry_msgs/Pose pose

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
---
bool success

")

(setf (get tauv_msgs::SetTargetPoseResponse :md5sum-) "b66f4c8c2fa81b98544cf0f3e10950ee")
(setf (get tauv_msgs::SetTargetPoseResponse :datatype-) "tauv_msgs/SetTargetPoseResponse")
(setf (get tauv_msgs::SetTargetPoseResponse :definition-)
      "geometry_msgs/Pose pose

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
---
bool success

")



(provide :tauv_msgs/SetTargetPose "b66f4c8c2fa81b98544cf0f3e10950ee")


