;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::MapFind)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'MapFind (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::MAPFIND")
  (make-package "TAUV_MSGS::MAPFIND"))
(unless (find-package "TAUV_MSGS::MAPFINDREQUEST")
  (make-package "TAUV_MSGS::MAPFINDREQUEST"))
(unless (find-package "TAUV_MSGS::MAPFINDRESPONSE")
  (make-package "TAUV_MSGS::MAPFINDRESPONSE"))

(in-package "ROS")





(defclass tauv_msgs::MapFindRequest
  :super ros::object
  :slots (_tag ))

(defmethod tauv_msgs::MapFindRequest
  (:init
   (&key
    ((:tag __tag) "")
    )
   (send-super :init)
   (setq _tag (string __tag))
   self)
  (:tag
   (&optional __tag)
   (if __tag (setq _tag __tag)) _tag)
  (:serialization-length
   ()
   (+
    ;; string _tag
    4 (length _tag)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _tag
       (write-long (length _tag) s) (princ _tag s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _tag
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _tag (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass tauv_msgs::MapFindResponse
  :super ros::object
  :slots (_detections _success ))

(defmethod tauv_msgs::MapFindResponse
  (:init
   (&key
    ((:detections __detections) ())
    ((:success __success) nil)
    )
   (send-super :init)
   (setq _detections __detections)
   (setq _success __success)
   self)
  (:detections
   (&rest __detections)
   (if (keywordp (car __detections))
       (send* _detections __detections)
     (progn
       (if __detections (setq _detections (car __detections)))
       _detections)))
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:serialization-length
   ()
   (+
    ;; tauv_msgs/FeatureDetection[] _detections
    (apply #'+ (send-all _detections :serialization-length)) 4
    ;; bool _success
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; tauv_msgs/FeatureDetection[] _detections
     (write-long (length _detections) s)
     (dolist (elem _detections)
       (send elem :serialize s)
       )
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; tauv_msgs/FeatureDetection[] _detections
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _detections (let (r) (dotimes (i n) (push (instance tauv_msgs::FeatureDetection :init) r)) r))
     (dolist (elem- _detections)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass tauv_msgs::MapFind
  :super ros::object
  :slots ())

(setf (get tauv_msgs::MapFind :md5sum-) "c4b23e80f3f361fabb0b381682792779")
(setf (get tauv_msgs::MapFind :datatype-) "tauv_msgs/MapFind")
(setf (get tauv_msgs::MapFind :request) tauv_msgs::MapFindRequest)
(setf (get tauv_msgs::MapFind :response) tauv_msgs::MapFindResponse)

(defmethod tauv_msgs::MapFindRequest
  (:response () (instance tauv_msgs::MapFindResponse :init)))

(setf (get tauv_msgs::MapFindRequest :md5sum-) "c4b23e80f3f361fabb0b381682792779")
(setf (get tauv_msgs::MapFindRequest :datatype-) "tauv_msgs/MapFindRequest")
(setf (get tauv_msgs::MapFindRequest :definition-)
      "string tag
---
tauv_msgs/FeatureDetection[] detections
bool success

================================================================================
MSG: tauv_msgs/FeatureDetection
Header header
geometry_msgs/Point position #SE2 msgs will only use x y
geometry_msgs/Point orientation #SE2 msgs will only use x (theta)
string tag
float64 confidence
bool SE2
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z
")

(setf (get tauv_msgs::MapFindResponse :md5sum-) "c4b23e80f3f361fabb0b381682792779")
(setf (get tauv_msgs::MapFindResponse :datatype-) "tauv_msgs/MapFindResponse")
(setf (get tauv_msgs::MapFindResponse :definition-)
      "string tag
---
tauv_msgs/FeatureDetection[] detections
bool success

================================================================================
MSG: tauv_msgs/FeatureDetection
Header header
geometry_msgs/Point position #SE2 msgs will only use x y
geometry_msgs/Point orientation #SE2 msgs will only use x (theta)
string tag
float64 confidence
bool SE2
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z
")



(provide :tauv_msgs/MapFind "c4b23e80f3f361fabb0b381682792779")


