;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::ControllerCommand)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'ControllerCommand (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::CONTROLLERCOMMAND")
  (make-package "TAUV_MSGS::CONTROLLERCOMMAND"))

(in-package "ROS")
;;//! \htmlinclude ControllerCommand.msg.html


(defclass tauv_msgs::ControllerCommand
  :super ros::object
  :slots (_a_x _a_y _a_z _a_roll _a_pitch _a_yaw _f_x _f_y _f_z _f_roll _f_pitch _f_yaw _use_f_x _use_f_y _use_f_z _use_f_roll _use_f_pitch _use_f_yaw _setpoint_z _setpoint_roll _setpoint_pitch _use_setpoint_z _use_setpoint_roll _use_setpoint_pitch ))

(defmethod tauv_msgs::ControllerCommand
  (:init
   (&key
    ((:a_x __a_x) 0.0)
    ((:a_y __a_y) 0.0)
    ((:a_z __a_z) 0.0)
    ((:a_roll __a_roll) 0.0)
    ((:a_pitch __a_pitch) 0.0)
    ((:a_yaw __a_yaw) 0.0)
    ((:f_x __f_x) 0.0)
    ((:f_y __f_y) 0.0)
    ((:f_z __f_z) 0.0)
    ((:f_roll __f_roll) 0.0)
    ((:f_pitch __f_pitch) 0.0)
    ((:f_yaw __f_yaw) 0.0)
    ((:use_f_x __use_f_x) nil)
    ((:use_f_y __use_f_y) nil)
    ((:use_f_z __use_f_z) nil)
    ((:use_f_roll __use_f_roll) nil)
    ((:use_f_pitch __use_f_pitch) nil)
    ((:use_f_yaw __use_f_yaw) nil)
    ((:setpoint_z __setpoint_z) 0.0)
    ((:setpoint_roll __setpoint_roll) 0.0)
    ((:setpoint_pitch __setpoint_pitch) 0.0)
    ((:use_setpoint_z __use_setpoint_z) nil)
    ((:use_setpoint_roll __use_setpoint_roll) nil)
    ((:use_setpoint_pitch __use_setpoint_pitch) nil)
    )
   (send-super :init)
   (setq _a_x (float __a_x))
   (setq _a_y (float __a_y))
   (setq _a_z (float __a_z))
   (setq _a_roll (float __a_roll))
   (setq _a_pitch (float __a_pitch))
   (setq _a_yaw (float __a_yaw))
   (setq _f_x (float __f_x))
   (setq _f_y (float __f_y))
   (setq _f_z (float __f_z))
   (setq _f_roll (float __f_roll))
   (setq _f_pitch (float __f_pitch))
   (setq _f_yaw (float __f_yaw))
   (setq _use_f_x __use_f_x)
   (setq _use_f_y __use_f_y)
   (setq _use_f_z __use_f_z)
   (setq _use_f_roll __use_f_roll)
   (setq _use_f_pitch __use_f_pitch)
   (setq _use_f_yaw __use_f_yaw)
   (setq _setpoint_z (float __setpoint_z))
   (setq _setpoint_roll (float __setpoint_roll))
   (setq _setpoint_pitch (float __setpoint_pitch))
   (setq _use_setpoint_z __use_setpoint_z)
   (setq _use_setpoint_roll __use_setpoint_roll)
   (setq _use_setpoint_pitch __use_setpoint_pitch)
   self)
  (:a_x
   (&optional __a_x)
   (if __a_x (setq _a_x __a_x)) _a_x)
  (:a_y
   (&optional __a_y)
   (if __a_y (setq _a_y __a_y)) _a_y)
  (:a_z
   (&optional __a_z)
   (if __a_z (setq _a_z __a_z)) _a_z)
  (:a_roll
   (&optional __a_roll)
   (if __a_roll (setq _a_roll __a_roll)) _a_roll)
  (:a_pitch
   (&optional __a_pitch)
   (if __a_pitch (setq _a_pitch __a_pitch)) _a_pitch)
  (:a_yaw
   (&optional __a_yaw)
   (if __a_yaw (setq _a_yaw __a_yaw)) _a_yaw)
  (:f_x
   (&optional __f_x)
   (if __f_x (setq _f_x __f_x)) _f_x)
  (:f_y
   (&optional __f_y)
   (if __f_y (setq _f_y __f_y)) _f_y)
  (:f_z
   (&optional __f_z)
   (if __f_z (setq _f_z __f_z)) _f_z)
  (:f_roll
   (&optional __f_roll)
   (if __f_roll (setq _f_roll __f_roll)) _f_roll)
  (:f_pitch
   (&optional __f_pitch)
   (if __f_pitch (setq _f_pitch __f_pitch)) _f_pitch)
  (:f_yaw
   (&optional __f_yaw)
   (if __f_yaw (setq _f_yaw __f_yaw)) _f_yaw)
  (:use_f_x
   (&optional (__use_f_x :null))
   (if (not (eq __use_f_x :null)) (setq _use_f_x __use_f_x)) _use_f_x)
  (:use_f_y
   (&optional (__use_f_y :null))
   (if (not (eq __use_f_y :null)) (setq _use_f_y __use_f_y)) _use_f_y)
  (:use_f_z
   (&optional (__use_f_z :null))
   (if (not (eq __use_f_z :null)) (setq _use_f_z __use_f_z)) _use_f_z)
  (:use_f_roll
   (&optional (__use_f_roll :null))
   (if (not (eq __use_f_roll :null)) (setq _use_f_roll __use_f_roll)) _use_f_roll)
  (:use_f_pitch
   (&optional (__use_f_pitch :null))
   (if (not (eq __use_f_pitch :null)) (setq _use_f_pitch __use_f_pitch)) _use_f_pitch)
  (:use_f_yaw
   (&optional (__use_f_yaw :null))
   (if (not (eq __use_f_yaw :null)) (setq _use_f_yaw __use_f_yaw)) _use_f_yaw)
  (:setpoint_z
   (&optional __setpoint_z)
   (if __setpoint_z (setq _setpoint_z __setpoint_z)) _setpoint_z)
  (:setpoint_roll
   (&optional __setpoint_roll)
   (if __setpoint_roll (setq _setpoint_roll __setpoint_roll)) _setpoint_roll)
  (:setpoint_pitch
   (&optional __setpoint_pitch)
   (if __setpoint_pitch (setq _setpoint_pitch __setpoint_pitch)) _setpoint_pitch)
  (:use_setpoint_z
   (&optional (__use_setpoint_z :null))
   (if (not (eq __use_setpoint_z :null)) (setq _use_setpoint_z __use_setpoint_z)) _use_setpoint_z)
  (:use_setpoint_roll
   (&optional (__use_setpoint_roll :null))
   (if (not (eq __use_setpoint_roll :null)) (setq _use_setpoint_roll __use_setpoint_roll)) _use_setpoint_roll)
  (:use_setpoint_pitch
   (&optional (__use_setpoint_pitch :null))
   (if (not (eq __use_setpoint_pitch :null)) (setq _use_setpoint_pitch __use_setpoint_pitch)) _use_setpoint_pitch)
  (:serialization-length
   ()
   (+
    ;; float32 _a_x
    4
    ;; float32 _a_y
    4
    ;; float32 _a_z
    4
    ;; float32 _a_roll
    4
    ;; float32 _a_pitch
    4
    ;; float32 _a_yaw
    4
    ;; float32 _f_x
    4
    ;; float32 _f_y
    4
    ;; float32 _f_z
    4
    ;; float32 _f_roll
    4
    ;; float32 _f_pitch
    4
    ;; float32 _f_yaw
    4
    ;; bool _use_f_x
    1
    ;; bool _use_f_y
    1
    ;; bool _use_f_z
    1
    ;; bool _use_f_roll
    1
    ;; bool _use_f_pitch
    1
    ;; bool _use_f_yaw
    1
    ;; float32 _setpoint_z
    4
    ;; float32 _setpoint_roll
    4
    ;; float32 _setpoint_pitch
    4
    ;; bool _use_setpoint_z
    1
    ;; bool _use_setpoint_roll
    1
    ;; bool _use_setpoint_pitch
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _a_x
       (sys::poke _a_x (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _a_y
       (sys::poke _a_y (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _a_z
       (sys::poke _a_z (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _a_roll
       (sys::poke _a_roll (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _a_pitch
       (sys::poke _a_pitch (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _a_yaw
       (sys::poke _a_yaw (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _f_x
       (sys::poke _f_x (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _f_y
       (sys::poke _f_y (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _f_z
       (sys::poke _f_z (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _f_roll
       (sys::poke _f_roll (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _f_pitch
       (sys::poke _f_pitch (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _f_yaw
       (sys::poke _f_yaw (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; bool _use_f_x
       (if _use_f_x (write-byte -1 s) (write-byte 0 s))
     ;; bool _use_f_y
       (if _use_f_y (write-byte -1 s) (write-byte 0 s))
     ;; bool _use_f_z
       (if _use_f_z (write-byte -1 s) (write-byte 0 s))
     ;; bool _use_f_roll
       (if _use_f_roll (write-byte -1 s) (write-byte 0 s))
     ;; bool _use_f_pitch
       (if _use_f_pitch (write-byte -1 s) (write-byte 0 s))
     ;; bool _use_f_yaw
       (if _use_f_yaw (write-byte -1 s) (write-byte 0 s))
     ;; float32 _setpoint_z
       (sys::poke _setpoint_z (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _setpoint_roll
       (sys::poke _setpoint_roll (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _setpoint_pitch
       (sys::poke _setpoint_pitch (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; bool _use_setpoint_z
       (if _use_setpoint_z (write-byte -1 s) (write-byte 0 s))
     ;; bool _use_setpoint_roll
       (if _use_setpoint_roll (write-byte -1 s) (write-byte 0 s))
     ;; bool _use_setpoint_pitch
       (if _use_setpoint_pitch (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _a_x
     (setq _a_x (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _a_y
     (setq _a_y (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _a_z
     (setq _a_z (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _a_roll
     (setq _a_roll (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _a_pitch
     (setq _a_pitch (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _a_yaw
     (setq _a_yaw (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _f_x
     (setq _f_x (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _f_y
     (setq _f_y (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _f_z
     (setq _f_z (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _f_roll
     (setq _f_roll (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _f_pitch
     (setq _f_pitch (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _f_yaw
     (setq _f_yaw (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; bool _use_f_x
     (setq _use_f_x (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _use_f_y
     (setq _use_f_y (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _use_f_z
     (setq _use_f_z (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _use_f_roll
     (setq _use_f_roll (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _use_f_pitch
     (setq _use_f_pitch (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _use_f_yaw
     (setq _use_f_yaw (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float32 _setpoint_z
     (setq _setpoint_z (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _setpoint_roll
     (setq _setpoint_roll (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _setpoint_pitch
     (setq _setpoint_pitch (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; bool _use_setpoint_z
     (setq _use_setpoint_z (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _use_setpoint_roll
     (setq _use_setpoint_roll (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _use_setpoint_pitch
     (setq _use_setpoint_pitch (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(setf (get tauv_msgs::ControllerCommand :md5sum-) "ad1b57ce703dafd167a4f28711d03e4e")
(setf (get tauv_msgs::ControllerCommand :datatype-) "tauv_msgs/ControllerCommand")
(setf (get tauv_msgs::ControllerCommand :definition-)
      "# Accelerations
float32 a_x
float32 a_y
float32 a_z
float32 a_roll
float32 a_pitch
float32 a_yaw

# Forces
float32 f_x
float32 f_y
float32 f_z
float32 f_roll
float32 f_pitch
float32 f_yaw

# If set, override accelerations and use forces
bool use_f_x
bool use_f_y
bool use_f_z
bool use_f_roll
bool use_f_pitch
bool use_f_yaw

# Setpoints
float32 setpoint_z
float32 setpoint_roll
float32 setpoint_pitch

# If set, override accelerations and forces and use setpoints
bool use_setpoint_z
bool use_setpoint_roll
bool use_setpoint_pitch
")



(provide :tauv_msgs/ControllerCommand "ad1b57ce703dafd167a4f28711d03e4e")


