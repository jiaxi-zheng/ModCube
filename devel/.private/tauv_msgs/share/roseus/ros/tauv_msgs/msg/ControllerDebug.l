;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::ControllerDebug)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'ControllerDebug (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::CONTROLLERDEBUG")
  (make-package "TAUV_MSGS::CONTROLLERDEBUG"))

(in-package "ROS")
;;//! \htmlinclude ControllerDebug.msg.html


(defclass tauv_msgs::ControllerDebug
  :super ros::object
  :slots (_stamp _z _roll _pitch ))

(defmethod tauv_msgs::ControllerDebug
  (:init
   (&key
    ((:stamp __stamp) (instance ros::time :init))
    ((:z __z) (instance tauv_msgs::PIDDebug :init))
    ((:roll __roll) (instance tauv_msgs::PIDDebug :init))
    ((:pitch __pitch) (instance tauv_msgs::PIDDebug :init))
    )
   (send-super :init)
   (setq _stamp __stamp)
   (setq _z __z)
   (setq _roll __roll)
   (setq _pitch __pitch)
   self)
  (:stamp
   (&optional __stamp)
   (if __stamp (setq _stamp __stamp)) _stamp)
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
  (:serialization-length
   ()
   (+
    ;; time _stamp
    8
    ;; tauv_msgs/PIDDebug _z
    (send _z :serialization-length)
    ;; tauv_msgs/PIDDebug _roll
    (send _roll :serialization-length)
    ;; tauv_msgs/PIDDebug _pitch
    (send _pitch :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; time _stamp
       (write-long (send _stamp :sec) s) (write-long (send _stamp :nsec) s)
     ;; tauv_msgs/PIDDebug _z
       (send _z :serialize s)
     ;; tauv_msgs/PIDDebug _roll
       (send _roll :serialize s)
     ;; tauv_msgs/PIDDebug _pitch
       (send _pitch :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; time _stamp
     (send _stamp :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _stamp :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; tauv_msgs/PIDDebug _z
     (send _z :deserialize buf ptr-) (incf ptr- (send _z :serialization-length))
   ;; tauv_msgs/PIDDebug _roll
     (send _roll :deserialize buf ptr-) (incf ptr- (send _roll :serialization-length))
   ;; tauv_msgs/PIDDebug _pitch
     (send _pitch :deserialize buf ptr-) (incf ptr- (send _pitch :serialization-length))
   ;;
   self)
  )

(setf (get tauv_msgs::ControllerDebug :md5sum-) "bb322eef3e5c11d51e1cb54450d584b1")
(setf (get tauv_msgs::ControllerDebug :datatype-) "tauv_msgs/ControllerDebug")
(setf (get tauv_msgs::ControllerDebug :definition-)
      "time stamp

PIDDebug z
PIDDebug roll
PIDDebug pitch
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



(provide :tauv_msgs/ControllerDebug "bb322eef3e5c11d51e1cb54450d584b1")


