;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::FeatureDetectionsSync)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'FeatureDetectionsSync (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::FEATUREDETECTIONSSYNC")
  (make-package "TAUV_MSGS::FEATUREDETECTIONSSYNC"))
(unless (find-package "TAUV_MSGS::FEATUREDETECTIONSSYNCREQUEST")
  (make-package "TAUV_MSGS::FEATUREDETECTIONSSYNCREQUEST"))
(unless (find-package "TAUV_MSGS::FEATUREDETECTIONSSYNCRESPONSE")
  (make-package "TAUV_MSGS::FEATUREDETECTIONSSYNCRESPONSE"))

(in-package "ROS")





(defclass tauv_msgs::FeatureDetectionsSyncRequest
  :super ros::object
  :slots (_detections ))

(defmethod tauv_msgs::FeatureDetectionsSyncRequest
  (:init
   (&key
    ((:detections __detections) (instance tauv_msgs::FeatureDetections :init))
    )
   (send-super :init)
   (setq _detections __detections)
   self)
  (:detections
   (&rest __detections)
   (if (keywordp (car __detections))
       (send* _detections __detections)
     (progn
       (if __detections (setq _detections (car __detections)))
       _detections)))
  (:serialization-length
   ()
   (+
    ;; tauv_msgs/FeatureDetections _detections
    (send _detections :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; tauv_msgs/FeatureDetections _detections
       (send _detections :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; tauv_msgs/FeatureDetections _detections
     (send _detections :deserialize buf ptr-) (incf ptr- (send _detections :serialization-length))
   ;;
   self)
  )

(defclass tauv_msgs::FeatureDetectionsSyncResponse
  :super ros::object
  :slots (_success ))

(defmethod tauv_msgs::FeatureDetectionsSyncResponse
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

(defclass tauv_msgs::FeatureDetectionsSync
  :super ros::object
  :slots ())

(setf (get tauv_msgs::FeatureDetectionsSync :md5sum-) "2da451629f8ca3f1c8b6e832ba66c6c5")
(setf (get tauv_msgs::FeatureDetectionsSync :datatype-) "tauv_msgs/FeatureDetectionsSync")
(setf (get tauv_msgs::FeatureDetectionsSync :request) tauv_msgs::FeatureDetectionsSyncRequest)
(setf (get tauv_msgs::FeatureDetectionsSync :response) tauv_msgs::FeatureDetectionsSyncResponse)

(defmethod tauv_msgs::FeatureDetectionsSyncRequest
  (:response () (instance tauv_msgs::FeatureDetectionsSyncResponse :init)))

(setf (get tauv_msgs::FeatureDetectionsSyncRequest :md5sum-) "2da451629f8ca3f1c8b6e832ba66c6c5")
(setf (get tauv_msgs::FeatureDetectionsSyncRequest :datatype-) "tauv_msgs/FeatureDetectionsSyncRequest")
(setf (get tauv_msgs::FeatureDetectionsSyncRequest :definition-)
      "tauv_msgs/FeatureDetections detections

================================================================================
MSG: tauv_msgs/FeatureDetections
FeatureDetection[] detections
string detector_tag
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
---
bool success
")

(setf (get tauv_msgs::FeatureDetectionsSyncResponse :md5sum-) "2da451629f8ca3f1c8b6e832ba66c6c5")
(setf (get tauv_msgs::FeatureDetectionsSyncResponse :datatype-) "tauv_msgs/FeatureDetectionsSyncResponse")
(setf (get tauv_msgs::FeatureDetectionsSyncResponse :definition-)
      "tauv_msgs/FeatureDetections detections

================================================================================
MSG: tauv_msgs/FeatureDetections
FeatureDetection[] detections
string detector_tag
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
---
bool success
")



(provide :tauv_msgs/FeatureDetectionsSync "2da451629f8ca3f1c8b6e832ba66c6c5")


