;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::RegisterMeasurement)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'RegisterMeasurement (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::REGISTERMEASUREMENT")
  (make-package "TAUV_MSGS::REGISTERMEASUREMENT"))

(in-package "ROS")
;;//! \htmlinclude RegisterMeasurement.msg.html


(defclass tauv_msgs::RegisterMeasurement
  :super ros::object
  :slots (_pg_meas ))

(defmethod tauv_msgs::RegisterMeasurement
  (:init
   (&key
    ((:pg_meas __pg_meas) (instance tauv_msgs::PoseGraphMeasurement :init))
    )
   (send-super :init)
   (setq _pg_meas __pg_meas)
   self)
  (:pg_meas
   (&rest __pg_meas)
   (if (keywordp (car __pg_meas))
       (send* _pg_meas __pg_meas)
     (progn
       (if __pg_meas (setq _pg_meas (car __pg_meas)))
       _pg_meas)))
  (:serialization-length
   ()
   (+
    ;; tauv_msgs/PoseGraphMeasurement _pg_meas
    (send _pg_meas :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; tauv_msgs/PoseGraphMeasurement _pg_meas
       (send _pg_meas :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; tauv_msgs/PoseGraphMeasurement _pg_meas
     (send _pg_meas :deserialize buf ptr-) (incf ptr- (send _pg_meas :serialization-length))
   ;;
   self)
  )

(setf (get tauv_msgs::RegisterMeasurement :md5sum-) "b355dd17bfdad2a0499de8384660e7ff")
(setf (get tauv_msgs::RegisterMeasurement :datatype-) "tauv_msgs/RegisterMeasurement")
(setf (get tauv_msgs::RegisterMeasurement :definition-)
      "PoseGraphMeasurement pg_meas
================================================================================
MSG: tauv_msgs/PoseGraphMeasurement
Header header
uint32 landmark_id
geometry_msgs/Point position


================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

")



(provide :tauv_msgs/RegisterMeasurement "b355dd17bfdad2a0499de8384660e7ff")


