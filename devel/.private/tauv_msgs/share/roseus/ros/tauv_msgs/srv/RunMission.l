;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::RunMission)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'RunMission (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::RUNMISSION")
  (make-package "TAUV_MSGS::RUNMISSION"))
(unless (find-package "TAUV_MSGS::RUNMISSIONREQUEST")
  (make-package "TAUV_MSGS::RUNMISSIONREQUEST"))
(unless (find-package "TAUV_MSGS::RUNMISSIONRESPONSE")
  (make-package "TAUV_MSGS::RUNMISSIONRESPONSE"))

(in-package "ROS")





(defclass tauv_msgs::RunMissionRequest
  :super ros::object
  :slots (_mission_name ))

(defmethod tauv_msgs::RunMissionRequest
  (:init
   (&key
    ((:mission_name __mission_name) "")
    )
   (send-super :init)
   (setq _mission_name (string __mission_name))
   self)
  (:mission_name
   (&optional __mission_name)
   (if __mission_name (setq _mission_name __mission_name)) _mission_name)
  (:serialization-length
   ()
   (+
    ;; string _mission_name
    4 (length _mission_name)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _mission_name
       (write-long (length _mission_name) s) (princ _mission_name s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _mission_name
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _mission_name (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass tauv_msgs::RunMissionResponse
  :super ros::object
  :slots (_success _message ))

(defmethod tauv_msgs::RunMissionResponse
  (:init
   (&key
    ((:success __success) nil)
    ((:message __message) "")
    )
   (send-super :init)
   (setq _success __success)
   (setq _message (string __message))
   self)
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:message
   (&optional __message)
   (if __message (setq _message __message)) _message)
  (:serialization-length
   ()
   (+
    ;; bool _success
    1
    ;; string _message
    4 (length _message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;; string _message
       (write-long (length _message) s) (princ _message s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; string _message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass tauv_msgs::RunMission
  :super ros::object
  :slots ())

(setf (get tauv_msgs::RunMission :md5sum-) "9f981363eccc116598d821fe063b0990")
(setf (get tauv_msgs::RunMission :datatype-) "tauv_msgs/RunMission")
(setf (get tauv_msgs::RunMission :request) tauv_msgs::RunMissionRequest)
(setf (get tauv_msgs::RunMission :response) tauv_msgs::RunMissionResponse)

(defmethod tauv_msgs::RunMissionRequest
  (:response () (instance tauv_msgs::RunMissionResponse :init)))

(setf (get tauv_msgs::RunMissionRequest :md5sum-) "9f981363eccc116598d821fe063b0990")
(setf (get tauv_msgs::RunMissionRequest :datatype-) "tauv_msgs/RunMissionRequest")
(setf (get tauv_msgs::RunMissionRequest :definition-)
      "string mission_name
---
bool success
string message
")

(setf (get tauv_msgs::RunMissionResponse :md5sum-) "9f981363eccc116598d821fe063b0990")
(setf (get tauv_msgs::RunMissionResponse :datatype-) "tauv_msgs/RunMissionResponse")
(setf (get tauv_msgs::RunMissionResponse :definition-)
      "string mission_name
---
bool success
string message
")



(provide :tauv_msgs/RunMission "9f981363eccc116598d821fe063b0990")


