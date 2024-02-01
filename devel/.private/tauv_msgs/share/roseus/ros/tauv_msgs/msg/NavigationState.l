;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::NavigationState)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'NavigationState (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::NAVIGATIONSTATE")
  (make-package "TAUV_MSGS::NAVIGATIONSTATE"))

(in-package "ROS")
;;//! \htmlinclude NavigationState.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass tauv_msgs::NavigationState
  :super ros::object
  :slots (_header _position _linear_velocity _linear_acceleration _orientation _euler_velocity _euler_acceleration ))

(defmethod tauv_msgs::NavigationState
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:position __position) (instance geometry_msgs::Vector3 :init))
    ((:linear_velocity __linear_velocity) (instance geometry_msgs::Vector3 :init))
    ((:linear_acceleration __linear_acceleration) (instance geometry_msgs::Vector3 :init))
    ((:orientation __orientation) (instance geometry_msgs::Vector3 :init))
    ((:euler_velocity __euler_velocity) (instance geometry_msgs::Vector3 :init))
    ((:euler_acceleration __euler_acceleration) (instance geometry_msgs::Vector3 :init))
    )
   (send-super :init)
   (setq _header __header)
   (setq _position __position)
   (setq _linear_velocity __linear_velocity)
   (setq _linear_acceleration __linear_acceleration)
   (setq _orientation __orientation)
   (setq _euler_velocity __euler_velocity)
   (setq _euler_acceleration __euler_acceleration)
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
  (:linear_velocity
   (&rest __linear_velocity)
   (if (keywordp (car __linear_velocity))
       (send* _linear_velocity __linear_velocity)
     (progn
       (if __linear_velocity (setq _linear_velocity (car __linear_velocity)))
       _linear_velocity)))
  (:linear_acceleration
   (&rest __linear_acceleration)
   (if (keywordp (car __linear_acceleration))
       (send* _linear_acceleration __linear_acceleration)
     (progn
       (if __linear_acceleration (setq _linear_acceleration (car __linear_acceleration)))
       _linear_acceleration)))
  (:orientation
   (&rest __orientation)
   (if (keywordp (car __orientation))
       (send* _orientation __orientation)
     (progn
       (if __orientation (setq _orientation (car __orientation)))
       _orientation)))
  (:euler_velocity
   (&rest __euler_velocity)
   (if (keywordp (car __euler_velocity))
       (send* _euler_velocity __euler_velocity)
     (progn
       (if __euler_velocity (setq _euler_velocity (car __euler_velocity)))
       _euler_velocity)))
  (:euler_acceleration
   (&rest __euler_acceleration)
   (if (keywordp (car __euler_acceleration))
       (send* _euler_acceleration __euler_acceleration)
     (progn
       (if __euler_acceleration (setq _euler_acceleration (car __euler_acceleration)))
       _euler_acceleration)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; geometry_msgs/Vector3 _position
    (send _position :serialization-length)
    ;; geometry_msgs/Vector3 _linear_velocity
    (send _linear_velocity :serialization-length)
    ;; geometry_msgs/Vector3 _linear_acceleration
    (send _linear_acceleration :serialization-length)
    ;; geometry_msgs/Vector3 _orientation
    (send _orientation :serialization-length)
    ;; geometry_msgs/Vector3 _euler_velocity
    (send _euler_velocity :serialization-length)
    ;; geometry_msgs/Vector3 _euler_acceleration
    (send _euler_acceleration :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; geometry_msgs/Vector3 _position
       (send _position :serialize s)
     ;; geometry_msgs/Vector3 _linear_velocity
       (send _linear_velocity :serialize s)
     ;; geometry_msgs/Vector3 _linear_acceleration
       (send _linear_acceleration :serialize s)
     ;; geometry_msgs/Vector3 _orientation
       (send _orientation :serialize s)
     ;; geometry_msgs/Vector3 _euler_velocity
       (send _euler_velocity :serialize s)
     ;; geometry_msgs/Vector3 _euler_acceleration
       (send _euler_acceleration :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; geometry_msgs/Vector3 _position
     (send _position :deserialize buf ptr-) (incf ptr- (send _position :serialization-length))
   ;; geometry_msgs/Vector3 _linear_velocity
     (send _linear_velocity :deserialize buf ptr-) (incf ptr- (send _linear_velocity :serialization-length))
   ;; geometry_msgs/Vector3 _linear_acceleration
     (send _linear_acceleration :deserialize buf ptr-) (incf ptr- (send _linear_acceleration :serialization-length))
   ;; geometry_msgs/Vector3 _orientation
     (send _orientation :deserialize buf ptr-) (incf ptr- (send _orientation :serialization-length))
   ;; geometry_msgs/Vector3 _euler_velocity
     (send _euler_velocity :deserialize buf ptr-) (incf ptr- (send _euler_velocity :serialization-length))
   ;; geometry_msgs/Vector3 _euler_acceleration
     (send _euler_acceleration :deserialize buf ptr-) (incf ptr- (send _euler_acceleration :serialization-length))
   ;;
   self)
  )

(setf (get tauv_msgs::NavigationState :md5sum-) "3507e091ce71f01fa9dea0061304746a")
(setf (get tauv_msgs::NavigationState :datatype-) "tauv_msgs/NavigationState")
(setf (get tauv_msgs::NavigationState :definition-)
      "Header header

geometry_msgs/Vector3 position
geometry_msgs/Vector3 linear_velocity      # body
geometry_msgs/Vector3 linear_acceleration  # body
geometry_msgs/Vector3 orientation          # (r, p, y)
geometry_msgs/Vector3 euler_velocity       # (dr/dt, dp/dt, dy/dt)
geometry_msgs/Vector3 euler_acceleration   # (d2r/dt2, d2p/dt2, d2y/dt2)

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
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
float64 z
")



(provide :tauv_msgs/NavigationState "3507e091ce71f01fa9dea0061304746a")


