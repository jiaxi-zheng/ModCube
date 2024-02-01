;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::AlarmReport)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'AlarmReport (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::ALARMREPORT")
  (make-package "TAUV_MSGS::ALARMREPORT"))

(in-package "ROS")
;;//! \htmlinclude AlarmReport.msg.html


(defclass tauv_msgs::AlarmReport
  :super ros::object
  :slots (_stamp _active_alarms ))

(defmethod tauv_msgs::AlarmReport
  (:init
   (&key
    ((:stamp __stamp) (instance ros::time :init))
    ((:active_alarms __active_alarms) (make-array 0 :initial-element 0 :element-type :integer))
    )
   (send-super :init)
   (setq _stamp __stamp)
   (setq _active_alarms __active_alarms)
   self)
  (:stamp
   (&optional __stamp)
   (if __stamp (setq _stamp __stamp)) _stamp)
  (:active_alarms
   (&optional __active_alarms)
   (if __active_alarms (setq _active_alarms __active_alarms)) _active_alarms)
  (:serialization-length
   ()
   (+
    ;; time _stamp
    8
    ;; int32[] _active_alarms
    (* 4    (length _active_alarms)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; time _stamp
       (write-long (send _stamp :sec) s) (write-long (send _stamp :nsec) s)
     ;; int32[] _active_alarms
     (write-long (length _active_alarms) s)
     (dotimes (i (length _active_alarms))
       (write-long (elt _active_alarms i) s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; time _stamp
     (send _stamp :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _stamp :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32[] _active_alarms
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _active_alarms (instantiate integer-vector n))
     (dotimes (i n)
     (setf (elt _active_alarms i) (sys::peek buf ptr- :integer)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(setf (get tauv_msgs::AlarmReport :md5sum-) "6041271f37a12a54ca5b8c77ba39eab9")
(setf (get tauv_msgs::AlarmReport :datatype-) "tauv_msgs/AlarmReport")
(setf (get tauv_msgs::AlarmReport :definition-)
      "time stamp
int32[] active_alarms
")



(provide :tauv_msgs/AlarmReport "6041271f37a12a54ca5b8c77ba39eab9")


