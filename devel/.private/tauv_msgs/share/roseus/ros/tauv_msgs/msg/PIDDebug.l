;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::PIDDebug)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'PIDDebug (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::PIDDEBUG")
  (make-package "TAUV_MSGS::PIDDEBUG"))

(in-package "ROS")
;;//! \htmlinclude PIDDebug.msg.html


(defclass tauv_msgs::PIDDebug
  :super ros::object
  :slots (_tuning _value _setpoint _error _proportional _integral _derivative _effort ))

(defmethod tauv_msgs::PIDDebug
  (:init
   (&key
    ((:tuning __tuning) (instance tauv_msgs::PIDTuning :init))
    ((:value __value) 0.0)
    ((:setpoint __setpoint) 0.0)
    ((:error __error) 0.0)
    ((:proportional __proportional) 0.0)
    ((:integral __integral) 0.0)
    ((:derivative __derivative) 0.0)
    ((:effort __effort) 0.0)
    )
   (send-super :init)
   (setq _tuning __tuning)
   (setq _value (float __value))
   (setq _setpoint (float __setpoint))
   (setq _error (float __error))
   (setq _proportional (float __proportional))
   (setq _integral (float __integral))
   (setq _derivative (float __derivative))
   (setq _effort (float __effort))
   self)
  (:tuning
   (&rest __tuning)
   (if (keywordp (car __tuning))
       (send* _tuning __tuning)
     (progn
       (if __tuning (setq _tuning (car __tuning)))
       _tuning)))
  (:value
   (&optional __value)
   (if __value (setq _value __value)) _value)
  (:setpoint
   (&optional __setpoint)
   (if __setpoint (setq _setpoint __setpoint)) _setpoint)
  (:error
   (&optional __error)
   (if __error (setq _error __error)) _error)
  (:proportional
   (&optional __proportional)
   (if __proportional (setq _proportional __proportional)) _proportional)
  (:integral
   (&optional __integral)
   (if __integral (setq _integral __integral)) _integral)
  (:derivative
   (&optional __derivative)
   (if __derivative (setq _derivative __derivative)) _derivative)
  (:effort
   (&optional __effort)
   (if __effort (setq _effort __effort)) _effort)
  (:serialization-length
   ()
   (+
    ;; tauv_msgs/PIDTuning _tuning
    (send _tuning :serialization-length)
    ;; float64 _value
    8
    ;; float64 _setpoint
    8
    ;; float64 _error
    8
    ;; float64 _proportional
    8
    ;; float64 _integral
    8
    ;; float64 _derivative
    8
    ;; float64 _effort
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; tauv_msgs/PIDTuning _tuning
       (send _tuning :serialize s)
     ;; float64 _value
       (sys::poke _value (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _setpoint
       (sys::poke _setpoint (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _error
       (sys::poke _error (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _proportional
       (sys::poke _proportional (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _integral
       (sys::poke _integral (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _derivative
       (sys::poke _derivative (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _effort
       (sys::poke _effort (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; tauv_msgs/PIDTuning _tuning
     (send _tuning :deserialize buf ptr-) (incf ptr- (send _tuning :serialization-length))
   ;; float64 _value
     (setq _value (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _setpoint
     (setq _setpoint (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _error
     (setq _error (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _proportional
     (setq _proportional (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _integral
     (setq _integral (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _derivative
     (setq _derivative (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _effort
     (setq _effort (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(setf (get tauv_msgs::PIDDebug :md5sum-) "16ed0a79b3cb076d76fdb91c1c560d6a")
(setf (get tauv_msgs::PIDDebug :datatype-) "tauv_msgs/PIDDebug")
(setf (get tauv_msgs::PIDDebug :definition-)
      "PIDTuning tuning

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



(provide :tauv_msgs/PIDDebug "16ed0a79b3cb076d76fdb91c1c560d6a")


