;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::PIDTuning)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'PIDTuning (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::PIDTUNING")
  (make-package "TAUV_MSGS::PIDTUNING"))

(in-package "ROS")
;;//! \htmlinclude PIDTuning.msg.html


(defclass tauv_msgs::PIDTuning
  :super ros::object
  :slots (_axis _kp _ki _kd _tau _limits ))

(defmethod tauv_msgs::PIDTuning
  (:init
   (&key
    ((:axis __axis) "")
    ((:kp __kp) 0.0)
    ((:ki __ki) 0.0)
    ((:kd __kd) 0.0)
    ((:tau __tau) 0.0)
    ((:limits __limits) (make-array 2 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _axis (string __axis))
   (setq _kp (float __kp))
   (setq _ki (float __ki))
   (setq _kd (float __kd))
   (setq _tau (float __tau))
   (setq _limits __limits)
   self)
  (:axis
   (&optional __axis)
   (if __axis (setq _axis __axis)) _axis)
  (:kp
   (&optional __kp)
   (if __kp (setq _kp __kp)) _kp)
  (:ki
   (&optional __ki)
   (if __ki (setq _ki __ki)) _ki)
  (:kd
   (&optional __kd)
   (if __kd (setq _kd __kd)) _kd)
  (:tau
   (&optional __tau)
   (if __tau (setq _tau __tau)) _tau)
  (:limits
   (&optional __limits)
   (if __limits (setq _limits __limits)) _limits)
  (:serialization-length
   ()
   (+
    ;; string _axis
    4 (length _axis)
    ;; float64 _kp
    8
    ;; float64 _ki
    8
    ;; float64 _kd
    8
    ;; float64 _tau
    8
    ;; float64[2] _limits
    (* 8    2)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _axis
       (write-long (length _axis) s) (princ _axis s)
     ;; float64 _kp
       (sys::poke _kp (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _ki
       (sys::poke _ki (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _kd
       (sys::poke _kd (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _tau
       (sys::poke _tau (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64[2] _limits
     (dotimes (i 2)
       (sys::poke (elt _limits i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _axis
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _axis (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; float64 _kp
     (setq _kp (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _ki
     (setq _ki (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _kd
     (setq _kd (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _tau
     (setq _tau (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64[2] _limits
   (dotimes (i (length _limits))
     (setf (elt _limits i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;;
   self)
  )

(setf (get tauv_msgs::PIDTuning :md5sum-) "0c319ae505ebfe0fb1604e4f13fce3d5")
(setf (get tauv_msgs::PIDTuning :datatype-) "tauv_msgs/PIDTuning")
(setf (get tauv_msgs::PIDTuning :definition-)
      "string axis
float64 kp
float64 ki
float64 kd
float64 tau
float64[2] limits
")



(provide :tauv_msgs/PIDTuning "0c319ae505ebfe0fb1604e4f13fce3d5")


