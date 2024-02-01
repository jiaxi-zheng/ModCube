;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::SetPose)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'SetPose (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::SETPOSE")
  (make-package "TAUV_MSGS::SETPOSE"))
(unless (find-package "TAUV_MSGS::SETPOSEREQUEST")
  (make-package "TAUV_MSGS::SETPOSEREQUEST"))
(unless (find-package "TAUV_MSGS::SETPOSERESPONSE")
  (make-package "TAUV_MSGS::SETPOSERESPONSE"))

(in-package "ROS")

(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))




(defclass tauv_msgs::SetPoseRequest
  :super ros::object
  :slots (_position _yaw ))

(defmethod tauv_msgs::SetPoseRequest
  (:init
   (&key
    ((:position __position) (instance geometry_msgs::Vector3 :init))
    ((:yaw __yaw) 0.0)
    )
   (send-super :init)
   (setq _position __position)
   (setq _yaw (float __yaw))
   self)
  (:position
   (&rest __position)
   (if (keywordp (car __position))
       (send* _position __position)
     (progn
       (if __position (setq _position (car __position)))
       _position)))
  (:yaw
   (&optional __yaw)
   (if __yaw (setq _yaw __yaw)) _yaw)
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Vector3 _position
    (send _position :serialization-length)
    ;; float64 _yaw
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Vector3 _position
       (send _position :serialize s)
     ;; float64 _yaw
       (sys::poke _yaw (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Vector3 _position
     (send _position :deserialize buf ptr-) (incf ptr- (send _position :serialization-length))
   ;; float64 _yaw
     (setq _yaw (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(defclass tauv_msgs::SetPoseResponse
  :super ros::object
  :slots (_success ))

(defmethod tauv_msgs::SetPoseResponse
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

(defclass tauv_msgs::SetPose
  :super ros::object
  :slots ())

(setf (get tauv_msgs::SetPose :md5sum-) "c0694c76e8e4d6ff81999164dc53cd23")
(setf (get tauv_msgs::SetPose :datatype-) "tauv_msgs/SetPose")
(setf (get tauv_msgs::SetPose :request) tauv_msgs::SetPoseRequest)
(setf (get tauv_msgs::SetPose :response) tauv_msgs::SetPoseResponse)

(defmethod tauv_msgs::SetPoseRequest
  (:response () (instance tauv_msgs::SetPoseResponse :init)))

(setf (get tauv_msgs::SetPoseRequest :md5sum-) "c0694c76e8e4d6ff81999164dc53cd23")
(setf (get tauv_msgs::SetPoseRequest :datatype-) "tauv_msgs/SetPoseRequest")
(setf (get tauv_msgs::SetPoseRequest :definition-)
      "geometry_msgs/Vector3 position
float64 yaw

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
---
bool success

")

(setf (get tauv_msgs::SetPoseResponse :md5sum-) "c0694c76e8e4d6ff81999164dc53cd23")
(setf (get tauv_msgs::SetPoseResponse :datatype-) "tauv_msgs/SetPoseResponse")
(setf (get tauv_msgs::SetPoseResponse :definition-)
      "geometry_msgs/Vector3 position
float64 yaw

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
---
bool success

")



(provide :tauv_msgs/SetPose "c0694c76e8e4d6ff81999164dc53cd23")


