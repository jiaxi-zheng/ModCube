;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::FeatureDetections)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'FeatureDetections (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::FEATUREDETECTIONS")
  (make-package "TAUV_MSGS::FEATUREDETECTIONS"))

(in-package "ROS")
;;//! \htmlinclude FeatureDetections.msg.html


(defclass tauv_msgs::FeatureDetections
  :super ros::object
  :slots (_detections _detector_tag ))

(defmethod tauv_msgs::FeatureDetections
  (:init
   (&key
    ((:detections __detections) ())
    ((:detector_tag __detector_tag) "")
    )
   (send-super :init)
   (setq _detections __detections)
   (setq _detector_tag (string __detector_tag))
   self)
  (:detections
   (&rest __detections)
   (if (keywordp (car __detections))
       (send* _detections __detections)
     (progn
       (if __detections (setq _detections (car __detections)))
       _detections)))
  (:detector_tag
   (&optional __detector_tag)
   (if __detector_tag (setq _detector_tag __detector_tag)) _detector_tag)
  (:serialization-length
   ()
   (+
    ;; tauv_msgs/FeatureDetection[] _detections
    (apply #'+ (send-all _detections :serialization-length)) 4
    ;; string _detector_tag
    4 (length _detector_tag)
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
     ;; string _detector_tag
       (write-long (length _detector_tag) s) (princ _detector_tag s)
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
   ;; string _detector_tag
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _detector_tag (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(setf (get tauv_msgs::FeatureDetections :md5sum-) "b198f96e11b160e3e0b3e1f890a3f57d")
(setf (get tauv_msgs::FeatureDetections :datatype-) "tauv_msgs/FeatureDetections")
(setf (get tauv_msgs::FeatureDetections :definition-)
      "FeatureDetection[] detections
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

")



(provide :tauv_msgs/FeatureDetections "b198f96e11b160e3e0b3e1f890a3f57d")


