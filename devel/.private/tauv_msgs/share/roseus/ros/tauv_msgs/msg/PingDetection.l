;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::PingDetection)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'PingDetection (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::PINGDETECTION")
  (make-package "TAUV_MSGS::PINGDETECTION"))

(in-package "ROS")
;;//! \htmlinclude PingDetection.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass tauv_msgs::PingDetection
  :super ros::object
  :slots (_header _frequency _direction ))

(defmethod tauv_msgs::PingDetection
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:frequency __frequency) 0.0)
    ((:direction __direction) (instance geometry_msgs::Vector3 :init))
    )
   (send-super :init)
   (setq _header __header)
   (setq _frequency (float __frequency))
   (setq _direction __direction)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:frequency
   (&optional __frequency)
   (if __frequency (setq _frequency __frequency)) _frequency)
  (:direction
   (&rest __direction)
   (if (keywordp (car __direction))
       (send* _direction __direction)
     (progn
       (if __direction (setq _direction (car __direction)))
       _direction)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; float64 _frequency
    8
    ;; geometry_msgs/Vector3 _direction
    (send _direction :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; float64 _frequency
       (sys::poke _frequency (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; geometry_msgs/Vector3 _direction
       (send _direction :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; float64 _frequency
     (setq _frequency (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; geometry_msgs/Vector3 _direction
     (send _direction :deserialize buf ptr-) (incf ptr- (send _direction :serialization-length))
   ;;
   self)
  )

(setf (get tauv_msgs::PingDetection :md5sum-) "a9abd7bac39db1726e1408962ec95f05")
(setf (get tauv_msgs::PingDetection :datatype-) "tauv_msgs/PingDetection")
(setf (get tauv_msgs::PingDetection :definition-)
      "Header header
float64 frequency
geometry_msgs/Vector3 direction
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



(provide :tauv_msgs/PingDetection "a9abd7bac39db1726e1408962ec95f05")


