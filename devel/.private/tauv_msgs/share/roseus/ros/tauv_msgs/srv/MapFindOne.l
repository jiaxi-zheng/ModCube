;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::MapFindOne)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'MapFindOne (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::MAPFINDONE")
  (make-package "TAUV_MSGS::MAPFINDONE"))
(unless (find-package "TAUV_MSGS::MAPFINDONEREQUEST")
  (make-package "TAUV_MSGS::MAPFINDONEREQUEST"))
(unless (find-package "TAUV_MSGS::MAPFINDONERESPONSE")
  (make-package "TAUV_MSGS::MAPFINDONERESPONSE"))

(in-package "ROS")





(defclass tauv_msgs::MapFindOneRequest
  :super ros::object
  :slots (_tag ))

(defmethod tauv_msgs::MapFindOneRequest
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

(defclass tauv_msgs::MapFindOneResponse
  :super ros::object
  :slots (_detection _success ))

(defmethod tauv_msgs::MapFindOneResponse
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

(defclass tauv_msgs::MapFindOne
  :super ros::object
  :slots ())

(setf (get tauv_msgs::MapFindOne :md5sum-) "0b9e59f5c45dc6444e155df33a798059")
(setf (get tauv_msgs::MapFindOne :datatype-) "tauv_msgs/MapFindOne")
(setf (get tauv_msgs::MapFindOne :request) tauv_msgs::MapFindOneRequest)
(setf (get tauv_msgs::MapFindOne :response) tauv_msgs::MapFindOneResponse)

(defmethod tauv_msgs::MapFindOneRequest
  (:response () (instance tauv_msgs::MapFindOneResponse :init)))

(setf (get tauv_msgs::MapFindOneRequest :md5sum-) "0b9e59f5c45dc6444e155df33a798059")
(setf (get tauv_msgs::MapFindOneRequest :datatype-) "tauv_msgs/MapFindOneRequest")
(setf (get tauv_msgs::MapFindOneRequest :definition-)
      "string tag
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

(setf (get tauv_msgs::MapFindOneResponse :md5sum-) "0b9e59f5c45dc6444e155df33a798059")
(setf (get tauv_msgs::MapFindOneResponse :datatype-) "tauv_msgs/MapFindOneResponse")
(setf (get tauv_msgs::MapFindOneResponse :definition-)
      "string tag
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



(provide :tauv_msgs/MapFindOne "0b9e59f5c45dc6444e155df33a798059")


