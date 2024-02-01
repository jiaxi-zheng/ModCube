;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::FluidDepth)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'FluidDepth (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::FLUIDDEPTH")
  (make-package "TAUV_MSGS::FLUIDDEPTH"))

(in-package "ROS")
;;//! \htmlinclude FluidDepth.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass tauv_msgs::FluidDepth
  :super ros::object
  :slots (_header _depth ))

(defmethod tauv_msgs::FluidDepth
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:depth __depth) 0.0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _depth (float __depth))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:depth
   (&optional __depth)
   (if __depth (setq _depth __depth)) _depth)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; float64 _depth
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; float64 _depth
       (sys::poke _depth (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; float64 _depth
     (setq _depth (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(setf (get tauv_msgs::FluidDepth :md5sum-) "b5547e1d41aa6dc176f0964997ace01b")
(setf (get tauv_msgs::FluidDepth :datatype-) "tauv_msgs/FluidDepth")
(setf (get tauv_msgs::FluidDepth :definition-)
      "Header header

float64 depth

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

")



(provide :tauv_msgs/FluidDepth "b5547e1d41aa6dc176f0964997ace01b")


