;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::MapFindClosest)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'MapFindClosest (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::MAPFINDCLOSEST")
  (make-package "TAUV_MSGS::MAPFINDCLOSEST"))
(unless (find-package "TAUV_MSGS::MAPFINDCLOSESTREQUEST")
  (make-package "TAUV_MSGS::MAPFINDCLOSESTREQUEST"))
(unless (find-package "TAUV_MSGS::MAPFINDCLOSESTRESPONSE")
  (make-package "TAUV_MSGS::MAPFINDCLOSESTRESPONSE"))

(in-package "ROS")

(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))




(defclass tauv_msgs::MapFindClosestRequest
  :super ros::object
  :slots (_tag _point ))

(defmethod tauv_msgs::MapFindClosestRequest
  (:init
   (&key
    ((:tag __tag) "")
    ((:point __point) (instance geometry_msgs::Point :init))
    )
   (send-super :init)
   (setq _tag (string __tag))
   (setq _point __point)
   self)
  (:tag
   (&optional __tag)
   (if __tag (setq _tag __tag)) _tag)
  (:point
   (&rest __point)
   (if (keywordp (car __point))
       (send* _point __point)
     (progn
       (if __point (setq _point (car __point)))
       _point)))
  (:serialization-length
   ()
   (+
    ;; string _tag
    4 (length _tag)
    ;; geometry_msgs/Point _point
    (send _point :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _tag
       (write-long (length _tag) s) (princ _tag s)
     ;; geometry_msgs/Point _point
       (send _point :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _tag
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _tag (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; geometry_msgs/Point _point
     (send _point :deserialize buf ptr-) (incf ptr- (send _point :serialization-length))
   ;;
   self)
  )

(defclass tauv_msgs::MapFindClosestResponse
  :super ros::object
  :slots (_detection _success ))

(defmethod tauv_msgs::MapFindClosestResponse
  (:init
   (&key
    ((:detection __detection) (instance tauv_msgs::FeatureDetection :init))
    ((:success __success) nil)
    )
   (send-super :init)
   (setq _detection __detection)
   (setq _success __success)
   self)
  (:detection
   (&rest __detection)
   (if (keywordp (car __detection))
       (send* _detection __detection)
     (progn
       (if __detection (setq _detection (car __detection)))
       _detection)))
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:serialization-length
   ()
   (+
    ;; tauv_msgs/FeatureDetection _detection
    (send _detection :serialization-length)
    ;; bool _success
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; tauv_msgs/FeatureDetection _detection
       (send _detection :serialize s)
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; tauv_msgs/FeatureDetection _detection
     (send _detection :deserialize buf ptr-) (incf ptr- (send _detection :serialization-length))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass tauv_msgs::MapFindClosest
  :super ros::object
  :slots ())

(setf (get tauv_msgs::MapFindClosest :md5sum-) "8fcd7cf3f0341186fb07c7acdf8e1704")
(setf (get tauv_msgs::MapFindClosest :datatype-) "tauv_msgs/MapFindClosest")
(setf (get tauv_msgs::MapFindClosest :request) tauv_msgs::MapFindClosestRequest)
(setf (get tauv_msgs::MapFindClosest :response) tauv_msgs::MapFindClosestResponse)

(defmethod tauv_msgs::MapFindClosestRequest
  (:response () (instance tauv_msgs::MapFindClosestResponse :init)))

(setf (get tauv_msgs::MapFindClosestRequest :md5sum-) "8fcd7cf3f0341186fb07c7acdf8e1704")
(setf (get tauv_msgs::MapFindClosestRequest :datatype-) "tauv_msgs/MapFindClosestRequest")
(setf (get tauv_msgs::MapFindClosestRequest :definition-)
      "string tag
geometry_msgs/Point point

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z
---
tauv_msgs/FeatureDetection detection
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

(setf (get tauv_msgs::MapFindClosestResponse :md5sum-) "8fcd7cf3f0341186fb07c7acdf8e1704")
(setf (get tauv_msgs::MapFindClosestResponse :datatype-) "tauv_msgs/MapFindClosestResponse")
(setf (get tauv_msgs::MapFindClosestResponse :definition-)
      "string tag
geometry_msgs/Point point

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z
---
tauv_msgs/FeatureDetection detection
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



(provide :tauv_msgs/MapFindClosest "8fcd7cf3f0341186fb07c7acdf8e1704")


