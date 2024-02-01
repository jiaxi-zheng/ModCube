;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::TuneController)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'TuneController (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::TUNECONTROLLER")
  (make-package "TAUV_MSGS::TUNECONTROLLER"))
(unless (find-package "TAUV_MSGS::TUNECONTROLLERREQUEST")
  (make-package "TAUV_MSGS::TUNECONTROLLERREQUEST"))
(unless (find-package "TAUV_MSGS::TUNECONTROLLERRESPONSE")
  (make-package "TAUV_MSGS::TUNECONTROLLERRESPONSE"))

(in-package "ROS")





(defclass tauv_msgs::TuneControllerRequest
  :super ros::object
  :slots (_tunings ))

(defmethod tauv_msgs::TuneControllerRequest
  (:init
   (&key
    ((:tunings __tunings) ())
    )
   (send-super :init)
   (setq _tunings __tunings)
   self)
  (:tunings
   (&rest __tunings)
   (if (keywordp (car __tunings))
       (send* _tunings __tunings)
     (progn
       (if __tunings (setq _tunings (car __tunings)))
       _tunings)))
  (:serialization-length
   ()
   (+
    ;; tauv_msgs/PIDTuning[] _tunings
    (apply #'+ (send-all _tunings :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; tauv_msgs/PIDTuning[] _tunings
     (write-long (length _tunings) s)
     (dolist (elem _tunings)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; tauv_msgs/PIDTuning[] _tunings
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _tunings (let (r) (dotimes (i n) (push (instance tauv_msgs::PIDTuning :init) r)) r))
     (dolist (elem- _tunings)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(defclass tauv_msgs::TuneControllerResponse
  :super ros::object
  :slots (_success ))

(defmethod tauv_msgs::TuneControllerResponse
  (:init
   (&key
    ((:success __success) nil)
    )
   (send-super :init)
   (setq _success __success)
   self)
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:serialization-length
   ()
   (+
    ;; bool _success
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass tauv_msgs::TuneController
  :super ros::object
  :slots ())

(setf (get tauv_msgs::TuneController :md5sum-) "c6a95158ee66091a3c801f9968586b2d")
(setf (get tauv_msgs::TuneController :datatype-) "tauv_msgs/TuneController")
(setf (get tauv_msgs::TuneController :request) tauv_msgs::TuneControllerRequest)
(setf (get tauv_msgs::TuneController :response) tauv_msgs::TuneControllerResponse)

(defmethod tauv_msgs::TuneControllerRequest
  (:response () (instance tauv_msgs::TuneControllerResponse :init)))

(setf (get tauv_msgs::TuneControllerRequest :md5sum-) "c6a95158ee66091a3c801f9968586b2d")
(setf (get tauv_msgs::TuneControllerRequest :datatype-) "tauv_msgs/TuneControllerRequest")
(setf (get tauv_msgs::TuneControllerRequest :definition-)
      "tauv_msgs/PIDTuning[] tunings

================================================================================
MSG: tauv_msgs/PIDTuning
string axis
float64 kp
float64 ki
float64 kd
float64 tau
---
bool success
")

(setf (get tauv_msgs::TuneControllerResponse :md5sum-) "c6a95158ee66091a3c801f9968586b2d")
(setf (get tauv_msgs::TuneControllerResponse :datatype-) "tauv_msgs/TuneControllerResponse")
(setf (get tauv_msgs::TuneControllerResponse :definition-)
      "tauv_msgs/PIDTuning[] tunings

================================================================================
MSG: tauv_msgs/PIDTuning
string axis
float64 kp
float64 ki
float64 kd
float64 tau
---
bool success
")



(provide :tauv_msgs/TuneController "c6a95158ee66091a3c801f9968586b2d")


