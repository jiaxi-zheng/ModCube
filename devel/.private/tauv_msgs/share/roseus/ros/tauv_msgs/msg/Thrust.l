;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::Thrust)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'Thrust (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::THRUST")
  (make-package "TAUV_MSGS::THRUST"))

(in-package "ROS")
;;//! \htmlinclude Thrust.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass tauv_msgs::Thrust
  :super ros::object
  :slots (_header _thrusts ))

(defmethod tauv_msgs::Thrust
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:thrusts __thrusts) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _header __header)
   (setq _thrusts __thrusts)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:thrusts
   (&optional __thrusts)
   (if __thrusts (setq _thrusts __thrusts)) _thrusts)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; float64[] _thrusts
    (* 8    (length _thrusts)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; float64[] _thrusts
     (write-long (length _thrusts) s)
     (dotimes (i (length _thrusts))
       (sys::poke (elt _thrusts i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; float64[] _thrusts
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _thrusts (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _thrusts i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;;
   self)
  )

(setf (get tauv_msgs::Thrust :md5sum-) "dad92a8d25b7ce836a9750ca65604332")
(setf (get tauv_msgs::Thrust :datatype-) "tauv_msgs/Thrust")
(setf (get tauv_msgs::Thrust :definition-)
      "Header header

float64[] thrusts
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



(provide :tauv_msgs/Thrust "dad92a8d25b7ce836a9750ca65604332")


