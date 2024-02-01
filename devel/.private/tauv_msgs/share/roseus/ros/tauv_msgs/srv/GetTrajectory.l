;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::GetTrajectory)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'GetTrajectory (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::GETTRAJECTORY")
  (make-package "TAUV_MSGS::GETTRAJECTORY"))
(unless (find-package "TAUV_MSGS::GETTRAJECTORYREQUEST")
  (make-package "TAUV_MSGS::GETTRAJECTORYREQUEST"))
(unless (find-package "TAUV_MSGS::GETTRAJECTORYRESPONSE")
  (make-package "TAUV_MSGS::GETTRAJECTORYRESPONSE"))

(in-package "ROS")

(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass tauv_msgs::GetTrajectoryRequest
  :super ros::object
  :slots (_header _curr_pose _curr_twist _curr_time _len _dt ))

(defmethod tauv_msgs::GetTrajectoryRequest
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:curr_pose __curr_pose) (instance geometry_msgs::Pose :init))
    ((:curr_twist __curr_twist) (instance geometry_msgs::Twist :init))
    ((:curr_time __curr_time) (instance ros::time :init))
    ((:len __len) 0)
    ((:dt __dt) 0.0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _curr_pose __curr_pose)
   (setq _curr_twist __curr_twist)
   (setq _curr_time __curr_time)
   (setq _len (round __len))
   (setq _dt (float __dt))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:curr_pose
   (&rest __curr_pose)
   (if (keywordp (car __curr_pose))
       (send* _curr_pose __curr_pose)
     (progn
       (if __curr_pose (setq _curr_pose (car __curr_pose)))
       _curr_pose)))
  (:curr_twist
   (&rest __curr_twist)
   (if (keywordp (car __curr_twist))
       (send* _curr_twist __curr_twist)
     (progn
       (if __curr_twist (setq _curr_twist (car __curr_twist)))
       _curr_twist)))
  (:curr_time
   (&optional __curr_time)
   (if __curr_time (setq _curr_time __curr_time)) _curr_time)
  (:len
   (&optional __len)
   (if __len (setq _len __len)) _len)
  (:dt
   (&optional __dt)
   (if __dt (setq _dt __dt)) _dt)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; geometry_msgs/Pose _curr_pose
    (send _curr_pose :serialization-length)
    ;; geometry_msgs/Twist _curr_twist
    (send _curr_twist :serialization-length)
    ;; time _curr_time
    8
    ;; int32 _len
    4
    ;; float32 _dt
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; geometry_msgs/Pose _curr_pose
       (send _curr_pose :serialize s)
     ;; geometry_msgs/Twist _curr_twist
       (send _curr_twist :serialize s)
     ;; time _curr_time
       (write-long (send _curr_time :sec) s) (write-long (send _curr_time :nsec) s)
     ;; int32 _len
       (write-long _len s)
     ;; float32 _dt
       (sys::poke _dt (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; geometry_msgs/Pose _curr_pose
     (send _curr_pose :deserialize buf ptr-) (incf ptr- (send _curr_pose :serialization-length))
   ;; geometry_msgs/Twist _curr_twist
     (send _curr_twist :deserialize buf ptr-) (incf ptr- (send _curr_twist :serialization-length))
   ;; time _curr_time
     (send _curr_time :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _curr_time :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _len
     (setq _len (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; float32 _dt
     (setq _dt (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(defclass tauv_msgs::GetTrajectoryResponse
  :super ros::object
  :slots (_poses _twists _success _message ))

(defmethod tauv_msgs::GetTrajectoryResponse
  (:init
   (&key
    ((:poses __poses) ())
    ((:twists __twists) ())
    ((:success __success) nil)
    ((:message __message) "")
    )
   (send-super :init)
   (setq _poses __poses)
   (setq _twists __twists)
   (setq _success __success)
   (setq _message (string __message))
   self)
  (:poses
   (&rest __poses)
   (if (keywordp (car __poses))
       (send* _poses __poses)
     (progn
       (if __poses (setq _poses (car __poses)))
       _poses)))
  (:twists
   (&rest __twists)
   (if (keywordp (car __twists))
       (send* _twists __twists)
     (progn
       (if __twists (setq _twists (car __twists)))
       _twists)))
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:message
   (&optional __message)
   (if __message (setq _message __message)) _message)
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Pose[] _poses
    (apply #'+ (send-all _poses :serialization-length)) 4
    ;; geometry_msgs/Twist[] _twists
    (apply #'+ (send-all _twists :serialization-length)) 4
    ;; bool _success
    1
    ;; string _message
    4 (length _message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Pose[] _poses
     (write-long (length _poses) s)
     (dolist (elem _poses)
       (send elem :serialize s)
       )
     ;; geometry_msgs/Twist[] _twists
     (write-long (length _twists) s)
     (dolist (elem _twists)
       (send elem :serialize s)
       )
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;; string _message
       (write-long (length _message) s) (princ _message s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Pose[] _poses
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _poses (let (r) (dotimes (i n) (push (instance geometry_msgs::Pose :init) r)) r))
     (dolist (elem- _poses)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; geometry_msgs/Twist[] _twists
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _twists (let (r) (dotimes (i n) (push (instance geometry_msgs::Twist :init) r)) r))
     (dolist (elem- _twists)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; string _message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass tauv_msgs::GetTrajectory
  :super ros::object
  :slots ())

(setf (get tauv_msgs::GetTrajectory :md5sum-) "a821c3e69b37025c156850881154fe33")
(setf (get tauv_msgs::GetTrajectory :datatype-) "tauv_msgs/GetTrajectory")
(setf (get tauv_msgs::GetTrajectory :request) tauv_msgs::GetTrajectoryRequest)
(setf (get tauv_msgs::GetTrajectory :response) tauv_msgs::GetTrajectoryResponse)

(defmethod tauv_msgs::GetTrajectoryRequest
  (:response () (instance tauv_msgs::GetTrajectoryResponse :init)))

(setf (get tauv_msgs::GetTrajectoryRequest :md5sum-) "a821c3e69b37025c156850881154fe33")
(setf (get tauv_msgs::GetTrajectoryRequest :datatype-) "tauv_msgs/GetTrajectoryRequest")
(setf (get tauv_msgs::GetTrajectoryRequest :definition-)
      "# Note: Angular velocities outside of yaw (z axis) are currently unused.

std_msgs/Header header
geometry_msgs/Pose curr_pose  # Current positions
geometry_msgs/Twist curr_twist  # Current velocities (in world frame! Not body velocities!)
time curr_time
int32 len  # Number of samples to look ahead on the trajectory. (First sample corresponds to current time, second is time + dt, etc)
float32 dt  # time difference between samples


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
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

================================================================================
MSG: geometry_msgs/Twist
# This expresses velocity in free space broken into its linear and angular parts.
Vector3  linear
Vector3  angular

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
---

geometry_msgs/Pose[] poses  # list of poses on trajectory
geometry_msgs/Twist[] twists  # list of twists on trajectory (in world frame! Not body velocities!)
bool success  # false indicates some sort of failure
string message

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

================================================================================
MSG: geometry_msgs/Twist
# This expresses velocity in free space broken into its linear and angular parts.
Vector3  linear
Vector3  angular

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
")

(setf (get tauv_msgs::GetTrajectoryResponse :md5sum-) "a821c3e69b37025c156850881154fe33")
(setf (get tauv_msgs::GetTrajectoryResponse :datatype-) "tauv_msgs/GetTrajectoryResponse")
(setf (get tauv_msgs::GetTrajectoryResponse :definition-)
      "# Note: Angular velocities outside of yaw (z axis) are currently unused.

std_msgs/Header header
geometry_msgs/Pose curr_pose  # Current positions
geometry_msgs/Twist curr_twist  # Current velocities (in world frame! Not body velocities!)
time curr_time
int32 len  # Number of samples to look ahead on the trajectory. (First sample corresponds to current time, second is time + dt, etc)
float32 dt  # time difference between samples


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
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

================================================================================
MSG: geometry_msgs/Twist
# This expresses velocity in free space broken into its linear and angular parts.
Vector3  linear
Vector3  angular

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
---

geometry_msgs/Pose[] poses  # list of poses on trajectory
geometry_msgs/Twist[] twists  # list of twists on trajectory (in world frame! Not body velocities!)
bool success  # false indicates some sort of failure
string message

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

================================================================================
MSG: geometry_msgs/Twist
# This expresses velocity in free space broken into its linear and angular parts.
Vector3  linear
Vector3  angular

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
")



(provide :tauv_msgs/GetTrajectory "a821c3e69b37025c156850881154fe33")


