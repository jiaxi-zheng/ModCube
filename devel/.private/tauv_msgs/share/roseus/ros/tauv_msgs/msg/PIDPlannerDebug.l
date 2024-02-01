;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::PIDPlannerDebug)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'PIDPlannerDebug (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::PIDPLANNERDEBUG")
  (make-package "TAUV_MSGS::PIDPLANNERDEBUG"))

(in-package "ROS")
;;//! \htmlinclude PIDPlannerDebug.msg.html


(defclass tauv_msgs::PIDPlannerDebug
  :super ros::object
  :slots (_stamp _x _y _z _roll _pitch _yaw ))

(defmethod tauv_msgs::PIDPlannerDebug
  (:init
   (&key
    ((:stamp __stamp) (instance ros::time :init))
    ((:x __x) (instance tauv_msgs::PIDDebug :init))
    ((:y __y) (instance tauv_msgs::PIDDebug :init))
    ((:z __z) (instance tauv_msgs::PIDDebug :init))
    ((:roll __roll) (instance tauv_msgs::PIDDebug :init))
    ((:pitch __pitch) (instance tauv_msgs::PIDDebug :init))
    ((:yaw __yaw) (instance tauv_msgs::PIDDebug :init))
    )
   (send-super :init)
   (setq _stamp __stamp)
   (setq _x __x)
   (setq _y __y)
   (setq _z __z)
   (setq _roll __roll)
   (setq _pitch __pitch)
   (setq _yaw __yaw)
   self)
  (:stamp
   (&optional __stamp)
   (if __stamp (setq _stamp __stamp)) _stamp)
  (:x
   (&rest __x)
   (if (keywordp (car __x))
       (send* _x __x)
     (progn
       (if __x (setq _x (car __x)))
       _x)))
  (:y
   (&rest __y)
   (if (keywordp (car __y))
       (send* _y __y)
     (progn
       (if __y (setq _y (car __y)))
       _y)))
  (:z
   (&rest __z)
   (if (keywordp (car __z))
       (send* _z __z)
     (progn
       (if __z (setq _z (car __z)))
       _z)))
  (:roll
   (&rest __roll)
   (if (keywordp (car __roll))
       (send* _roll __roll)
     (progn
       (if __roll (setq _roll (car __roll)))
       _roll)))
  (:pitch
   (&rest __pitch)
   (if (keywordp (car __pitch))
       (send* _pitch __pitch)
     (progn
       (if __pitch (setq _pitch (car __pitch)))
       _pitch)))
  (:yaw
   (&rest __yaw)
   (if (keywordp (car __yaw))
       (send* _yaw __yaw)
     (progn
       (if __yaw (setq _yaw (car __yaw)))
       _yaw)))
  (:serialization-length
   ()
   (+
    ;; time _stamp
    8
    ;; tauv_msgs/PIDDebug _x
    (send _x :serialization-length)
    ;; tauv_msgs/PIDDebug _y
    (send _y :serialization-length)
    ;; tauv_msgs/PIDDebug _z
    (send _z :serialization-length)
    ;; tauv_msgs/PIDDebug _roll
    (send _roll :serialization-length)
    ;; tauv_msgs/PIDDebug _pitch
    (send _pitch :serialization-length)
    ;; tauv_msgs/PIDDebug _yaw
    (send _yaw :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; time _stamp
       (write-long (send _stamp :sec) s) (write-long (send _stamp :nsec) s)
     ;; tauv_msgs/PIDDebug _x
       (send _x :serialize s)
     ;; tauv_msgs/PIDDebug _y
       (send _y :serialize s)
     ;; tauv_msgs/PIDDebug _z
       (send _z :serialize s)
     ;; tauv_msgs/PIDDebug _roll
       (send _roll :serialize s)
     ;; tauv_msgs/PIDDebug _pitch
       (send _pitch :serialize s)
     ;; tauv_msgs/PIDDebug _yaw
       (send _yaw :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; time _stamp
     (send _stamp :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _stamp :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; tauv_msgs/PIDDebug _x
     (send _x :deserialize buf ptr-) (incf ptr- (send _x :serialization-length))
   ;; tauv_msgs/PIDDebug _y
     (send _y :deserialize buf ptr-) (incf ptr- (send _y :serialization-length))
   ;; tauv_msgs/PIDDebug _z
     (send _z :deserialize buf ptr-) (incf ptr- (send _z :serialization-length))
   ;; tauv_msgs/PIDDebug _roll
     (send _roll :deserialize buf ptr-) (incf ptr- (send _roll :serialization-length))
   ;; tauv_msgs/PIDDebug _pitch
     (send _pitch :deserialize buf ptr-) (incf ptr- (send _pitch :serialization-length))
   ;; tauv_msgs/PIDDebug _yaw
     (send _yaw :deserialize buf ptr-) (incf ptr- (send _yaw :serialization-length))
   ;;
   self)
  )

(setf (get tauv_msgs::PIDPlannerDebug :md5sum-) "0f555c58b06b2c2d39d97456d9e34ef3")
(setf (get tauv_msgs::PIDPlannerDebug :datatype-) "tauv_msgs/PIDPlannerDebug")
(setf (get tauv_msgs::PIDPlannerDebug :definition-)
      "time stamp

PIDDebug x
PIDDebug y
PIDDebug z
PIDDebug roll
PIDDebug pitch
PIDDebug yaw
================================================================================
MSG: tauv_msgs/PIDDebug
PIDTuning tuning

float64 value
float64 setpoint
float64 error
float64 proportional
float64 integral
float64 derivative
float64 effort
================================================================================
MSG: tauv_msgs/PIDTuning
string axis
float64 kp
float64 ki
float64 kd
float64 tau
float64[2] limits
")



(provide :tauv_msgs/PIDPlannerDebug "0f555c58b06b2c2d39d97456d9e34ef3")


