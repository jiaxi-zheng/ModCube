;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::Battery)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'Battery (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::BATTERY")
  (make-package "TAUV_MSGS::BATTERY"))

(in-package "ROS")
;;//! \htmlinclude Battery.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass tauv_msgs::Battery
  :super ros::object
  :slots (_header _state_of_charge _voltage _average_current _remaining_capacity _full_capacity ))

(defmethod tauv_msgs::Battery
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:state_of_charge __state_of_charge) 0.0)
    ((:voltage __voltage) 0.0)
    ((:average_current __average_current) 0.0)
    ((:remaining_capacity __remaining_capacity) 0.0)
    ((:full_capacity __full_capacity) 0.0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _state_of_charge (float __state_of_charge))
   (setq _voltage (float __voltage))
   (setq _average_current (float __average_current))
   (setq _remaining_capacity (float __remaining_capacity))
   (setq _full_capacity (float __full_capacity))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:state_of_charge
   (&optional __state_of_charge)
   (if __state_of_charge (setq _state_of_charge __state_of_charge)) _state_of_charge)
  (:voltage
   (&optional __voltage)
   (if __voltage (setq _voltage __voltage)) _voltage)
  (:average_current
   (&optional __average_current)
   (if __average_current (setq _average_current __average_current)) _average_current)
  (:remaining_capacity
   (&optional __remaining_capacity)
   (if __remaining_capacity (setq _remaining_capacity __remaining_capacity)) _remaining_capacity)
  (:full_capacity
   (&optional __full_capacity)
   (if __full_capacity (setq _full_capacity __full_capacity)) _full_capacity)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; float64 _state_of_charge
    8
    ;; float64 _voltage
    8
    ;; float64 _average_current
    8
    ;; float64 _remaining_capacity
    8
    ;; float64 _full_capacity
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; float64 _state_of_charge
       (sys::poke _state_of_charge (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _voltage
       (sys::poke _voltage (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _average_current
       (sys::poke _average_current (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _remaining_capacity
       (sys::poke _remaining_capacity (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _full_capacity
       (sys::poke _full_capacity (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; float64 _state_of_charge
     (setq _state_of_charge (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _voltage
     (setq _voltage (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _average_current
     (setq _average_current (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _remaining_capacity
     (setq _remaining_capacity (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _full_capacity
     (setq _full_capacity (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(setf (get tauv_msgs::Battery :md5sum-) "6a49cc2bae2938fa186ed67011f717c3")
(setf (get tauv_msgs::Battery :datatype-) "tauv_msgs/Battery")
(setf (get tauv_msgs::Battery :definition-)
      "Header header

float64 state_of_charge
float64 voltage
float64 average_current
float64 remaining_capacity
float64 full_capacity
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



(provide :tauv_msgs/Battery "6a49cc2bae2938fa186ed67011f717c3")


