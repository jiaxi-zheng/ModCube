;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::SonarControl)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'SonarControl (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::SONARCONTROL")
  (make-package "TAUV_MSGS::SONARCONTROL"))
(unless (find-package "TAUV_MSGS::SONARCONTROLREQUEST")
  (make-package "TAUV_MSGS::SONARCONTROLREQUEST"))
(unless (find-package "TAUV_MSGS::SONARCONTROLRESPONSE")
  (make-package "TAUV_MSGS::SONARCONTROLRESPONSE"))

(in-package "ROS")





(defclass tauv_msgs::SonarControlRequest
  :super ros::object
  :slots (_op ))

(defmethod tauv_msgs::SonarControlRequest
  (:init
   (&key
    ((:op __op) 0)
    )
   (send-super :init)
   (setq _op (round __op))
   self)
  (:op
   (&optional __op)
   (if __op (setq _op __op)) _op)
  (:serialization-length
   ()
   (+
    ;; int8 _op
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int8 _op
       (write-byte _op s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int8 _op
     (setq _op (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _op 127) (setq _op (- _op 256)))
   ;;
   self)
  )

(defclass tauv_msgs::SonarControlResponse
  :super ros::object
  :slots (_success ))

(defmethod tauv_msgs::SonarControlResponse
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

(defclass tauv_msgs::SonarControl
  :super ros::object
  :slots ())

(setf (get tauv_msgs::SonarControl :md5sum-) "cdbd07758f57cd8fbf9c7b8e225a19cf")
(setf (get tauv_msgs::SonarControl :datatype-) "tauv_msgs/SonarControl")
(setf (get tauv_msgs::SonarControl :request) tauv_msgs::SonarControlRequest)
(setf (get tauv_msgs::SonarControl :response) tauv_msgs::SonarControlResponse)

(defmethod tauv_msgs::SonarControlRequest
  (:response () (instance tauv_msgs::SonarControlResponse :init)))

(setf (get tauv_msgs::SonarControlRequest :md5sum-) "cdbd07758f57cd8fbf9c7b8e225a19cf")
(setf (get tauv_msgs::SonarControlRequest :datatype-) "tauv_msgs/SonarControlRequest")
(setf (get tauv_msgs::SonarControlRequest :definition-)
      "int8 op
---
bool success
")

(setf (get tauv_msgs::SonarControlResponse :md5sum-) "cdbd07758f57cd8fbf9c7b8e225a19cf")
(setf (get tauv_msgs::SonarControlResponse :datatype-) "tauv_msgs/SonarControlResponse")
(setf (get tauv_msgs::SonarControlResponse :definition-)
      "int8 op
---
bool success
")



(provide :tauv_msgs/SonarControl "cdbd07758f57cd8fbf9c7b8e225a19cf")


