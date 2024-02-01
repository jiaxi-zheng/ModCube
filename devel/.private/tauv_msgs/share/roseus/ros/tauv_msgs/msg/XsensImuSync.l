;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::XsensImuSync)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'XsensImuSync (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::XSENSIMUSYNC")
  (make-package "TAUV_MSGS::XSENSIMUSYNC"))

(in-package "ROS")
;;//! \htmlinclude XsensImuSync.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass tauv_msgs::XsensImuSync
  :super ros::object
  :slots (_header _ros_time _imu_time _triggered_dvl _d_corrected _d_ros _d_imu ))

(defmethod tauv_msgs::XsensImuSync
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:ros_time __ros_time) (instance ros::time :init))
    ((:imu_time __imu_time) (instance ros::time :init))
    ((:triggered_dvl __triggered_dvl) nil)
    ((:d_corrected __d_corrected) 0.0)
    ((:d_ros __d_ros) 0.0)
    ((:d_imu __d_imu) 0.0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _ros_time __ros_time)
   (setq _imu_time __imu_time)
   (setq _triggered_dvl __triggered_dvl)
   (setq _d_corrected (float __d_corrected))
   (setq _d_ros (float __d_ros))
   (setq _d_imu (float __d_imu))
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
  (:d_corrected
   (&optional __d_corrected)
   (if __d_corrected (setq _d_corrected __d_corrected)) _d_corrected)
  (:d_ros
   (&optional __d_ros)
   (if __d_ros (setq _d_ros __d_ros)) _d_ros)
  (:d_imu
   (&optional __d_imu)
   (if __d_imu (setq _d_imu __d_imu)) _d_imu)
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
    ;; float64 _d_corrected
    8
    ;; float64 _d_ros
    8
    ;; float64 _d_imu
    8
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
     ;; float64 _d_corrected
       (sys::poke _d_corrected (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _d_ros
       (sys::poke _d_ros (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _d_imu
       (sys::poke _d_imu (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
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
   ;; float64 _d_corrected
     (setq _d_corrected (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _d_ros
     (setq _d_ros (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _d_imu
     (setq _d_imu (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(setf (get tauv_msgs::XsensImuSync :md5sum-) "8d95951a55cfd457da142f2cc6b05ae6")
(setf (get tauv_msgs::XsensImuSync :datatype-) "tauv_msgs/XsensImuSync")
(setf (get tauv_msgs::XsensImuSync :definition-)
      "Header header

time ros_time

time imu_time

bool triggered_dvl

float64 d_corrected # optional

float64 d_ros # optional

float64 d_imu # optional

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



(provide :tauv_msgs/XsensImuSync "8d95951a55cfd457da142f2cc6b05ae6")


