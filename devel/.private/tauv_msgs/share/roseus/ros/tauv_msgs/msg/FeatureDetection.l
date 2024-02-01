;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::FeatureDetection)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'FeatureDetection (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::FEATUREDETECTION")
  (make-package "TAUV_MSGS::FEATUREDETECTION"))

(in-package "ROS")
;;//! \htmlinclude FeatureDetection.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass tauv_msgs::FeatureDetection
  :super ros::object
  :slots (_header _position _orientation _tag _confidence _SE2 ))

(defmethod tauv_msgs::FeatureDetection
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:position __position) (instance geometry_msgs::Point :init))
    ((:orientation __orientation) (instance geometry_msgs::Point :init))
    ((:tag __tag) "")
    ((:confidence __confidence) 0.0)
    ((:SE2 __SE2) nil)
    )
   (send-super :init)
   (setq _header __header)
   (setq _position __position)
   (setq _orientation __orientation)
   (setq _tag (string __tag))
   (setq _confidence (float __confidence))
   (setq _SE2 __SE2)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:position
   (&rest __position)
   (if (keywordp (car __position))
       (send* _position __position)
     (progn
       (if __position (setq _position (car __position)))
       _position)))
  (:orientation
   (&rest __orientation)
   (if (keywordp (car __orientation))
       (send* _orientation __orientation)
     (progn
       (if __orientation (setq _orientation (car __orientation)))
       _orientation)))
  (:tag
   (&optional __tag)
   (if __tag (setq _tag __tag)) _tag)
  (:confidence
   (&optional __confidence)
   (if __confidence (setq _confidence __confidence)) _confidence)
  (:SE2
   (&optional (__SE2 :null))
   (if (not (eq __SE2 :null)) (setq _SE2 __SE2)) _SE2)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; geometry_msgs/Point _position
    (send _position :serialization-length)
    ;; geometry_msgs/Point _orientation
    (send _orientation :serialization-length)
    ;; string _tag
    4 (length _tag)
    ;; float64 _confidence
    8
    ;; bool _SE2
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; geometry_msgs/Point _position
       (send _position :serialize s)
     ;; geometry_msgs/Point _orientation
       (send _orientation :serialize s)
     ;; string _tag
       (write-long (length _tag) s) (princ _tag s)
     ;; float64 _confidence
       (sys::poke _confidence (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; bool _SE2
       (if _SE2 (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; geometry_msgs/Point _position
     (send _position :deserialize buf ptr-) (incf ptr- (send _position :serialization-length))
   ;; geometry_msgs/Point _orientation
     (send _orientation :deserialize buf ptr-) (incf ptr- (send _orientation :serialization-length))
   ;; string _tag
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _tag (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; float64 _confidence
     (setq _confidence (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; bool _SE2
     (setq _SE2 (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(setf (get tauv_msgs::FeatureDetection :md5sum-) "a256c9fb7a965addac0015cd369172cd")
(setf (get tauv_msgs::FeatureDetection :datatype-) "tauv_msgs/FeatureDetection")
(setf (get tauv_msgs::FeatureDetection :definition-)
      "Header header
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



(provide :tauv_msgs/FeatureDetection "a256c9fb7a965addac0015cd369172cd")


