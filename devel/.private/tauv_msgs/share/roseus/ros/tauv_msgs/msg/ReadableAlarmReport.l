;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::ReadableAlarmReport)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'ReadableAlarmReport (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::READABLEALARMREPORT")
  (make-package "TAUV_MSGS::READABLEALARMREPORT"))

(in-package "ROS")
;;//! \htmlinclude ReadableAlarmReport.msg.html


(defclass tauv_msgs::ReadableAlarmReport
  :super ros::object
  :slots (_failure_level _alarms_no_failure _alarms_predive_failure _alarms_mission_failure _alarms_critical_failure ))

(defmethod tauv_msgs::ReadableAlarmReport
  (:init
   (&key
    ((:failure_level __failure_level) "")
    ((:alarms_no_failure __alarms_no_failure) (let (r) (dotimes (i 0) (push "" r)) r))
    ((:alarms_predive_failure __alarms_predive_failure) (let (r) (dotimes (i 0) (push "" r)) r))
    ((:alarms_mission_failure __alarms_mission_failure) (let (r) (dotimes (i 0) (push "" r)) r))
    ((:alarms_critical_failure __alarms_critical_failure) (let (r) (dotimes (i 0) (push "" r)) r))
    )
   (send-super :init)
   (setq _failure_level (string __failure_level))
   (setq _alarms_no_failure __alarms_no_failure)
   (setq _alarms_predive_failure __alarms_predive_failure)
   (setq _alarms_mission_failure __alarms_mission_failure)
   (setq _alarms_critical_failure __alarms_critical_failure)
   self)
  (:failure_level
   (&optional __failure_level)
   (if __failure_level (setq _failure_level __failure_level)) _failure_level)
  (:alarms_no_failure
   (&optional __alarms_no_failure)
   (if __alarms_no_failure (setq _alarms_no_failure __alarms_no_failure)) _alarms_no_failure)
  (:alarms_predive_failure
   (&optional __alarms_predive_failure)
   (if __alarms_predive_failure (setq _alarms_predive_failure __alarms_predive_failure)) _alarms_predive_failure)
  (:alarms_mission_failure
   (&optional __alarms_mission_failure)
   (if __alarms_mission_failure (setq _alarms_mission_failure __alarms_mission_failure)) _alarms_mission_failure)
  (:alarms_critical_failure
   (&optional __alarms_critical_failure)
   (if __alarms_critical_failure (setq _alarms_critical_failure __alarms_critical_failure)) _alarms_critical_failure)
  (:serialization-length
   ()
   (+
    ;; string _failure_level
    4 (length _failure_level)
    ;; string[] _alarms_no_failure
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _alarms_no_failure)) 4
    ;; string[] _alarms_predive_failure
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _alarms_predive_failure)) 4
    ;; string[] _alarms_mission_failure
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _alarms_mission_failure)) 4
    ;; string[] _alarms_critical_failure
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _alarms_critical_failure)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _failure_level
       (write-long (length _failure_level) s) (princ _failure_level s)
     ;; string[] _alarms_no_failure
     (write-long (length _alarms_no_failure) s)
     (dolist (elem _alarms_no_failure)
       (write-long (length elem) s) (princ elem s)
       )
     ;; string[] _alarms_predive_failure
     (write-long (length _alarms_predive_failure) s)
     (dolist (elem _alarms_predive_failure)
       (write-long (length elem) s) (princ elem s)
       )
     ;; string[] _alarms_mission_failure
     (write-long (length _alarms_mission_failure) s)
     (dolist (elem _alarms_mission_failure)
       (write-long (length elem) s) (princ elem s)
       )
     ;; string[] _alarms_critical_failure
     (write-long (length _alarms_critical_failure) s)
     (dolist (elem _alarms_critical_failure)
       (write-long (length elem) s) (princ elem s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _failure_level
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _failure_level (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string[] _alarms_no_failure
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _alarms_no_failure (make-list n))
     (dotimes (i n)
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setf (elt _alarms_no_failure i) (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
     ))
   ;; string[] _alarms_predive_failure
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _alarms_predive_failure (make-list n))
     (dotimes (i n)
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setf (elt _alarms_predive_failure i) (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
     ))
   ;; string[] _alarms_mission_failure
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _alarms_mission_failure (make-list n))
     (dotimes (i n)
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setf (elt _alarms_mission_failure i) (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
     ))
   ;; string[] _alarms_critical_failure
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _alarms_critical_failure (make-list n))
     (dotimes (i n)
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setf (elt _alarms_critical_failure i) (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
     ))
   ;;
   self)
  )

(setf (get tauv_msgs::ReadableAlarmReport :md5sum-) "5420ac12365c08ba46b84a0a6b7db17d")
(setf (get tauv_msgs::ReadableAlarmReport :datatype-) "tauv_msgs/ReadableAlarmReport")
(setf (get tauv_msgs::ReadableAlarmReport :definition-)
      "string failure_level
string[] alarms_no_failure
string[] alarms_predive_failure
string[] alarms_mission_failure
string[] alarms_critical_failure
")



(provide :tauv_msgs/ReadableAlarmReport "5420ac12365c08ba46b84a0a6b7db17d")


