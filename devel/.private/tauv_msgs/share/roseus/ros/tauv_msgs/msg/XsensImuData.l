;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::XsensImuData)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'XsensImuData (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::XSENSIMUDATA")
  (make-package "TAUV_MSGS::XSENSIMUDATA"))

(in-package "ROS")
;;//! \htmlinclude XsensImuData.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass tauv_msgs::XsensImuData
  :super ros::object
  :slots (_header _ros_time _imu_time _triggered_dvl _orientation _rate_of_turn _linear_acceleration _free_acceleration ))

(defmethod tauv_msgs::XsensImuData
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:ros_time __ros_time) (instance ros::time :init))
    ((:imu_time __imu_time) (instance ros::time :init))
    ((:triggered_dvl __triggered_dvl) nil)
    ((:orientation __orientation) (instance geometry_msgs::Vector3 :init))
    ((:rate_of_turn __rate_of_turn) (instance geometry_msgs::Vector3 :init))
    ((:linear_acceleration __linear_acceleration) (instance geometry_msgs::Vector3 :init))
    ((:free_acceleration __free_acceleration) (instance geometry_msgs::Vector3 :init))
    )
   (send-super :init)
   (setq _header __header)
   (setq _ros_time __ros_time)
   (setq _imu_time __imu_time)
   (setq _triggered_dvl __triggered_dvl)
   (setq _orientation __orientation)
   (setq _rate_of_turn __rate_of_turn)
   (setq _linear_acceleration __linear_acceleration)
   (setq _free_acceleration __free_acceleration)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:ros_time
   (&optional __ros_time)
   (if __ros_time (setq _ros_time __ros_time)) _ros_time)
  (:imu_time
   (&optional __imu_time)
   (if __imu_time (setq _imu_time __imu_time)) _imu_time)
  (:triggered_dvl
   (&optional (__triggered_dvl :null))
   (if (not (eq __triggered_dvl :null)) (setq _triggered_dvl __triggered_dvl)) _triggered_dvl)
  (:orientation
   (&rest __orientation)
   (if (keywordp (car __orientation))
       (send* _orientation __orientation)
     (progn
       (if __orientation (setq _orientation (car __orientation)))
       _orientation)))
  (:rate_of_turn
   (&rest __rate_of_turn)
   (if (keywordp (car __rate_of_turn))
       (send* _rate_of_turn __rate_of_turn)
     (progn
       (if __rate_of_turn (setq _rate_of_turn (car __rate_of_turn)))
       _rate_of_turn)))
  (:linear_acceleration
   (&rest __linear_acceleration)
   (if (keywordp (car __linear_acceleration))
       (send* _linear_acceleration __linear_acceleration)
     (progn
       (if __linear_acceleration (setq _linear_acceleration (car __linear_acceleration)))
       _linear_acceleration)))
  (:free_acceleration
   (&rest __free_acceleration)
   (if (keywordp (car __free_acceleration))
       (send* _free_acceleration __free_acceleration)
     (progn
       (if __free_acceleration (setq _free_acceleration (car __free_acceleration)))
       _free_acceleration)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; time _ros_time
    8
    ;; time _imu_time
    8
    ;; bool _triggered_dvl
    1
    ;; geometry_msgs/Vector3 _orientation
    (send _orientation :serialization-length)
    ;; geometry_msgs/Vector3 _rate_of_turn
    (send _rate_of_turn :serialization-length)
    ;; geometry_msgs/Vector3 _linear_acceleration
    (send _linear_acceleration :serialization-length)
    ;; geometry_msgs/Vector3 _free_acceleration
    (send _free_acceleration :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; time _ros_time
       (write-long (send _ros_time :sec) s) (write-long (send _ros_time :nsec) s)
     ;; time _imu_time
       (write-long (send _imu_time :sec) s) (write-long (send _imu_time :nsec) s)
     ;; bool _triggered_dvl
       (if _triggered_dvl (write-byte -1 s) (write-byte 0 s))
     ;; geometry_msgs/Vector3 _orientation
       (send _orientation :serialize s)
     ;; geometry_msgs/Vector3 _rate_of_turn
       (send _rate_of_turn :serialize s)
     ;; geometry_msgs/Vector3 _linear_acceleration
       (send _linear_acceleration :serialize s)
     ;; geometry_msgs/Vector3 _free_acceleration
       (send _free_acceleration :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; time _ros_time
     (send _ros_time :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _ros_time :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; time _imu_time
     (send _imu_time :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _imu_time :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; bool _triggered_dvl
     (setq _triggered_dvl (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; geometry_msgs/Vector3 _orientation
     (send _orientation :deserialize buf ptr-) (incf ptr- (send _orientation :serialization-length))
   ;; geometry_msgs/Vector3 _rate_of_turn
     (send _rate_of_turn :deserialize buf ptr-) (incf ptr- (send _rate_of_turn :serialization-length))
   ;; geometry_msgs/Vector3 _linear_acceleration
     (send _linear_acceleration :deserialize buf ptr-) (incf ptr- (send _linear_acceleration :serialization-length))
   ;; geometry_msgs/Vector3 _free_acceleration
     (send _free_acceleration :deserialize buf ptr-) (incf ptr- (send _free_acceleration :serialization-length))
   ;;
   self)
  )

(setf (get tauv_msgs::XsensImuData :md5sum-) "a5e7b14c591863b869ed2281b0f6b1ed")
(setf (get tauv_msgs::XsensImuData :datatype-) "tauv_msgs/XsensImuData")
(setf (get tauv_msgs::XsensImuData :definition-)
      "Header header

time ros_time

time imu_time

bool triggered_dvl

geometry_msgs/Vector3 orientation

geometry_msgs/Vector3 rate_of_turn

geometry_msgs/Vector3 linear_acceleration

geometry_msgs/Vector3 free_acceleration


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



(provide :tauv_msgs/XsensImuData "a5e7b14c591863b869ed2281b0f6b1ed")


