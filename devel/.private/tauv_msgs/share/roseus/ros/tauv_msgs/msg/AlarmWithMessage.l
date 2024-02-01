;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::AlarmWithMessage)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'AlarmWithMessage (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::ALARMWITHMESSAGE")
  (make-package "TAUV_MSGS::ALARMWITHMESSAGE"))

(in-package "ROS")
;;//! \htmlinclude AlarmWithMessage.msg.html


(defclass tauv_msgs::AlarmWithMessage
  :super ros::object
  :slots (_id _set _message ))

(defmethod tauv_msgs::AlarmWithMessage
  (:init
   (&key
    ((:id __id) 0)
    ((:set __set) nil)
    ((:message __message) "")
    )
   (send-super :init)
   (setq _id (round __id))
   (setq _set __set)
   (setq _message (string __message))
   self)
  (:id
   (&optional __id)
   (if __id (setq _id __id)) _id)
  (:set
   (&optional (__set :null))
   (if (not (eq __set :null)) (setq _set __set)) _set)
  (:message
   (&optional __message)
   (if __message (setq _message __message)) _message)
  (:serialization-length
   ()
   (+
    ;; int32 _id
    4
    ;; bool _set
    1
    ;; string _message
    4 (length _message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int32 _id
       (write-long _id s)
     ;; bool _set
       (if _set (write-byte -1 s) (write-byte 0 s))
     ;; string _message
       (write-long (length _message) s) (princ _message s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int32 _id
     (setq _id (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; bool _set
     (setq _set (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; string _message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(setf (get tauv_msgs::AlarmWithMessage :md5sum-) "2b0df1cdd443c9ac99920553ac2eb8d8")
(setf (get tauv_msgs::AlarmWithMessage :datatype-) "tauv_msgs/AlarmWithMessage")
(setf (get tauv_msgs::AlarmWithMessage :definition-)
      "int32 id            # ID of the alarm
bool set            # True = set, False = Cleared
string message      # Readable message
")



(provide :tauv_msgs/AlarmWithMessage "2b0df1cdd443c9ac99920553ac2eb8d8")


