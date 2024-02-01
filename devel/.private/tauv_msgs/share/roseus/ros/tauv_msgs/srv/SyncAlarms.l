;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::SyncAlarms)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'SyncAlarms (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::SYNCALARMS")
  (make-package "TAUV_MSGS::SYNCALARMS"))
(unless (find-package "TAUV_MSGS::SYNCALARMSREQUEST")
  (make-package "TAUV_MSGS::SYNCALARMSREQUEST"))
(unless (find-package "TAUV_MSGS::SYNCALARMSRESPONSE")
  (make-package "TAUV_MSGS::SYNCALARMSRESPONSE"))

(in-package "ROS")





(defclass tauv_msgs::SyncAlarmsRequest
  :super ros::object
  :slots (_diff ))

(defmethod tauv_msgs::SyncAlarmsRequest
  (:init
   (&key
    ((:diff __diff) ())
    )
   (send-super :init)
   (setq _diff __diff)
   self)
  (:diff
   (&rest __diff)
   (if (keywordp (car __diff))
       (send* _diff __diff)
     (progn
       (if __diff (setq _diff (car __diff)))
       _diff)))
  (:serialization-length
   ()
   (+
    ;; tauv_msgs/AlarmWithMessage[] _diff
    (apply #'+ (send-all _diff :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; tauv_msgs/AlarmWithMessage[] _diff
     (write-long (length _diff) s)
     (dolist (elem _diff)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; tauv_msgs/AlarmWithMessage[] _diff
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _diff (let (r) (dotimes (i n) (push (instance tauv_msgs::AlarmWithMessage :init) r)) r))
     (dolist (elem- _diff)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(defclass tauv_msgs::SyncAlarmsResponse
  :super ros::object
  :slots (_stamp _active_alarms _success ))

(defmethod tauv_msgs::SyncAlarmsResponse
  (:init
   (&key
    ((:stamp __stamp) (instance ros::time :init))
    ((:active_alarms __active_alarms) (make-array 0 :initial-element 0 :element-type :integer))
    ((:success __success) nil)
    )
   (send-super :init)
   (setq _stamp __stamp)
   (setq _active_alarms __active_alarms)
   (setq _success __success)
   self)
  (:stamp
   (&optional __stamp)
   (if __stamp (setq _stamp __stamp)) _stamp)
  (:active_alarms
   (&optional __active_alarms)
   (if __active_alarms (setq _active_alarms __active_alarms)) _active_alarms)
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:serialization-length
   ()
   (+
    ;; time _stamp
    8
    ;; int32[] _active_alarms
    (* 4    (length _active_alarms)) 4
    ;; bool _success
    1
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
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
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
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass tauv_msgs::SyncAlarms
  :super ros::object
  :slots ())

(setf (get tauv_msgs::SyncAlarms :md5sum-) "54b1739021e723bf57d59b6622adc3ef")
(setf (get tauv_msgs::SyncAlarms :datatype-) "tauv_msgs/SyncAlarms")
(setf (get tauv_msgs::SyncAlarms :request) tauv_msgs::SyncAlarmsRequest)
(setf (get tauv_msgs::SyncAlarms :response) tauv_msgs::SyncAlarmsResponse)

(defmethod tauv_msgs::SyncAlarmsRequest
  (:response () (instance tauv_msgs::SyncAlarmsResponse :init)))

(setf (get tauv_msgs::SyncAlarmsRequest :md5sum-) "54b1739021e723bf57d59b6622adc3ef")
(setf (get tauv_msgs::SyncAlarmsRequest :datatype-) "tauv_msgs/SyncAlarmsRequest")
(setf (get tauv_msgs::SyncAlarmsRequest :definition-)
      "# Note: Angular velocities outside of yaw (z axis) are currently unused.

tauv_msgs/AlarmWithMessage[] diff


================================================================================
MSG: tauv_msgs/AlarmWithMessage
int32 id            # ID of the alarm
bool set            # True = set, False = Cleared
---

time stamp
int32[] active_alarms
bool success  # false indicates some sort of failure
")

(setf (get tauv_msgs::SyncAlarmsResponse :md5sum-) "54b1739021e723bf57d59b6622adc3ef")
(setf (get tauv_msgs::SyncAlarmsResponse :datatype-) "tauv_msgs/SyncAlarmsResponse")
(setf (get tauv_msgs::SyncAlarmsResponse :definition-)
      "# Note: Angular velocities outside of yaw (z axis) are currently unused.

tauv_msgs/AlarmWithMessage[] diff


================================================================================
MSG: tauv_msgs/AlarmWithMessage
int32 id            # ID of the alarm
bool set            # True = set, False = Cleared
---

time stamp
int32[] active_alarms
bool success  # false indicates some sort of failure
")



(provide :tauv_msgs/SyncAlarms "54b1739021e723bf57d59b6622adc3ef")


