;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::TrajPoint)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'TrajPoint (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::TRAJPOINT")
  (make-package "TAUV_MSGS::TRAJPOINT"))

(in-package "ROS")
;;//! \htmlinclude TrajPoint.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass tauv_msgs::TrajPoint
  :super ros::object
  :slots (_pose _twist ))

(defmethod tauv_msgs::TrajPoint
  (:init
   (&key
    ((:pose __pose) (instance geometry_msgs::Pose :init))
    ((:twist __twist) (instance geometry_msgs::Twist :init))
    )
   (send-super :init)
   (setq _pose __pose)
   (setq _twist __twist)
   self)
  (:pose
   (&rest __pose)
   (if (keywordp (car __pose))
       (send* _pose __pose)
     (progn
       (if __pose (setq _pose (car __pose)))
       _pose)))
  (:twist
   (&rest __twist)
   (if (keywordp (car __twist))
       (send* _twist __twist)
     (progn
       (if __twist (setq _twist (car __twist)))
       _twist)))
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Pose _pose
    (send _pose :serialization-length)
    ;; geometry_msgs/Twist _twist
    (send _twist :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Pose _pose
       (send _pose :serialize s)
     ;; geometry_msgs/Twist _twist
       (send _twist :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Pose _pose
     (send _pose :deserialize buf ptr-) (incf ptr- (send _pose :serialization-length))
   ;; geometry_msgs/Twist _twist
     (send _twist :deserialize buf ptr-) (incf ptr- (send _twist :serialization-length))
   ;;
   self)
  )

(setf (get tauv_msgs::TrajPoint :md5sum-) "c79f0d88a7597db980a56d7ac144c654")
(setf (get tauv_msgs::TrajPoint :datatype-) "tauv_msgs/TrajPoint")
(setf (get tauv_msgs::TrajPoint :definition-)
      "geometry_msgs/Pose pose     # world frame
geometry_msgs/Twist twist   # world frame
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

================================================================================
MSG: geometry_msgs/Twist
# This expresses velocity in free space broken into its linear and angular parts.
Vector3  linear
Vector3  angular

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
float64 z
")



(provide :tauv_msgs/TrajPoint "c79f0d88a7597db980a56d7ac144c654")


