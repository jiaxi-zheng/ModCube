;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::Can)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'Can (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::CAN")
  (make-package "TAUV_MSGS::CAN"))

(in-package "ROS")
;;//! \htmlinclude Can.msg.html


(defclass tauv_msgs::Can
  :super ros::object
  :slots (_voltage_current _temperature_cabin_current _rm_speed_rpm _rm_given_current _rm_total_angle _FB_auv_pit _FB_auv_rol _FB_auv_yaw _FB_auv_deep _FB_auv_deep_vel _FB_auv_ang_vel_pit _FB_auv_ang_vel_rol _FB_auv_ang_vel_yaw _PWM_set ))

(defmethod tauv_msgs::Can
  (:init
   (&key
    ((:voltage_current __voltage_current) 0)
    ((:temperature_cabin_current __temperature_cabin_current) 0)
    ((:rm_speed_rpm __rm_speed_rpm) (make-array 4 :initial-element 0 :element-type :integer))
    ((:rm_given_current __rm_given_current) (make-array 4 :initial-element 0 :element-type :integer))
    ((:rm_total_angle __rm_total_angle) (make-array 4 :initial-element 0 :element-type :integer))
    ((:FB_auv_pit __FB_auv_pit) (make-array 4 :initial-element 0 :element-type :integer))
    ((:FB_auv_rol __FB_auv_rol) (make-array 4 :initial-element 0 :element-type :integer))
    ((:FB_auv_yaw __FB_auv_yaw) (make-array 4 :initial-element 0 :element-type :integer))
    ((:FB_auv_deep __FB_auv_deep) (make-array 4 :initial-element 0 :element-type :integer))
    ((:FB_auv_deep_vel __FB_auv_deep_vel) (make-array 4 :initial-element 0 :element-type :integer))
    ((:FB_auv_ang_vel_pit __FB_auv_ang_vel_pit) (make-array 4 :initial-element 0 :element-type :integer))
    ((:FB_auv_ang_vel_rol __FB_auv_ang_vel_rol) (make-array 4 :initial-element 0 :element-type :integer))
    ((:FB_auv_ang_vel_yaw __FB_auv_ang_vel_yaw) (make-array 4 :initial-element 0 :element-type :integer))
    ((:PWM_set __PWM_set) (make-array 8 :initial-element 0 :element-type :integer))
    )
   (send-super :init)
   (setq _voltage_current (round __voltage_current))
   (setq _temperature_cabin_current (round __temperature_cabin_current))
   (setq _rm_speed_rpm __rm_speed_rpm)
   (setq _rm_given_current __rm_given_current)
   (setq _rm_total_angle __rm_total_angle)
   (setq _FB_auv_pit __FB_auv_pit)
   (setq _FB_auv_rol __FB_auv_rol)
   (setq _FB_auv_yaw __FB_auv_yaw)
   (setq _FB_auv_deep __FB_auv_deep)
   (setq _FB_auv_deep_vel __FB_auv_deep_vel)
   (setq _FB_auv_ang_vel_pit __FB_auv_ang_vel_pit)
   (setq _FB_auv_ang_vel_rol __FB_auv_ang_vel_rol)
   (setq _FB_auv_ang_vel_yaw __FB_auv_ang_vel_yaw)
   (setq _PWM_set __PWM_set)
   self)
  (:voltage_current
   (&optional __voltage_current)
   (if __voltage_current (setq _voltage_current __voltage_current)) _voltage_current)
  (:temperature_cabin_current
   (&optional __temperature_cabin_current)
   (if __temperature_cabin_current (setq _temperature_cabin_current __temperature_cabin_current)) _temperature_cabin_current)
  (:rm_speed_rpm
   (&optional __rm_speed_rpm)
   (if __rm_speed_rpm (setq _rm_speed_rpm __rm_speed_rpm)) _rm_speed_rpm)
  (:rm_given_current
   (&optional __rm_given_current)
   (if __rm_given_current (setq _rm_given_current __rm_given_current)) _rm_given_current)
  (:rm_total_angle
   (&optional __rm_total_angle)
   (if __rm_total_angle (setq _rm_total_angle __rm_total_angle)) _rm_total_angle)
  (:FB_auv_pit
   (&optional __FB_auv_pit)
   (if __FB_auv_pit (setq _FB_auv_pit __FB_auv_pit)) _FB_auv_pit)
  (:FB_auv_rol
   (&optional __FB_auv_rol)
   (if __FB_auv_rol (setq _FB_auv_rol __FB_auv_rol)) _FB_auv_rol)
  (:FB_auv_yaw
   (&optional __FB_auv_yaw)
   (if __FB_auv_yaw (setq _FB_auv_yaw __FB_auv_yaw)) _FB_auv_yaw)
  (:FB_auv_deep
   (&optional __FB_auv_deep)
   (if __FB_auv_deep (setq _FB_auv_deep __FB_auv_deep)) _FB_auv_deep)
  (:FB_auv_deep_vel
   (&optional __FB_auv_deep_vel)
   (if __FB_auv_deep_vel (setq _FB_auv_deep_vel __FB_auv_deep_vel)) _FB_auv_deep_vel)
  (:FB_auv_ang_vel_pit
   (&optional __FB_auv_ang_vel_pit)
   (if __FB_auv_ang_vel_pit (setq _FB_auv_ang_vel_pit __FB_auv_ang_vel_pit)) _FB_auv_ang_vel_pit)
  (:FB_auv_ang_vel_rol
   (&optional __FB_auv_ang_vel_rol)
   (if __FB_auv_ang_vel_rol (setq _FB_auv_ang_vel_rol __FB_auv_ang_vel_rol)) _FB_auv_ang_vel_rol)
  (:FB_auv_ang_vel_yaw
   (&optional __FB_auv_ang_vel_yaw)
   (if __FB_auv_ang_vel_yaw (setq _FB_auv_ang_vel_yaw __FB_auv_ang_vel_yaw)) _FB_auv_ang_vel_yaw)
  (:PWM_set
   (&optional __PWM_set)
   (if __PWM_set (setq _PWM_set __PWM_set)) _PWM_set)
  (:serialization-length
   ()
   (+
    ;; uint8 _voltage_current
    1
    ;; uint8 _temperature_cabin_current
    1
    ;; int16[4] _rm_speed_rpm
    (* 2    4)
    ;; int16[4] _rm_given_current
    (* 2    4)
    ;; int32[4] _rm_total_angle
    (* 4    4)
    ;; int16[4] _FB_auv_pit
    (* 2    4)
    ;; int16[4] _FB_auv_rol
    (* 2    4)
    ;; int16[4] _FB_auv_yaw
    (* 2    4)
    ;; int16[4] _FB_auv_deep
    (* 2    4)
    ;; int16[4] _FB_auv_deep_vel
    (* 2    4)
    ;; int16[4] _FB_auv_ang_vel_pit
    (* 2    4)
    ;; int16[4] _FB_auv_ang_vel_rol
    (* 2    4)
    ;; int16[4] _FB_auv_ang_vel_yaw
    (* 2    4)
    ;; int16[8] _PWM_set
    (* 2    8)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _voltage_current
       (write-byte _voltage_current s)
     ;; uint8 _temperature_cabin_current
       (write-byte _temperature_cabin_current s)
     ;; int16[4] _rm_speed_rpm
     (dotimes (i 4)
       (write-word (elt _rm_speed_rpm i) s)
       )
     ;; int16[4] _rm_given_current
     (dotimes (i 4)
       (write-word (elt _rm_given_current i) s)
       )
     ;; int32[4] _rm_total_angle
     (dotimes (i 4)
       (write-long (elt _rm_total_angle i) s)
       )
     ;; int16[4] _FB_auv_pit
     (dotimes (i 4)
       (write-word (elt _FB_auv_pit i) s)
       )
     ;; int16[4] _FB_auv_rol
     (dotimes (i 4)
       (write-word (elt _FB_auv_rol i) s)
       )
     ;; int16[4] _FB_auv_yaw
     (dotimes (i 4)
       (write-word (elt _FB_auv_yaw i) s)
       )
     ;; int16[4] _FB_auv_deep
     (dotimes (i 4)
       (write-word (elt _FB_auv_deep i) s)
       )
     ;; int16[4] _FB_auv_deep_vel
     (dotimes (i 4)
       (write-word (elt _FB_auv_deep_vel i) s)
       )
     ;; int16[4] _FB_auv_ang_vel_pit
     (dotimes (i 4)
       (write-word (elt _FB_auv_ang_vel_pit i) s)
       )
     ;; int16[4] _FB_auv_ang_vel_rol
     (dotimes (i 4)
       (write-word (elt _FB_auv_ang_vel_rol i) s)
       )
     ;; int16[4] _FB_auv_ang_vel_yaw
     (dotimes (i 4)
       (write-word (elt _FB_auv_ang_vel_yaw i) s)
       )
     ;; int16[8] _PWM_set
     (dotimes (i 8)
       (write-word (elt _PWM_set i) s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _voltage_current
     (setq _voltage_current (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _temperature_cabin_current
     (setq _temperature_cabin_current (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; int16[4] _rm_speed_rpm
   (dotimes (i (length _rm_speed_rpm))
     (setf (elt _rm_speed_rpm i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; int16[4] _rm_given_current
   (dotimes (i (length _rm_given_current))
     (setf (elt _rm_given_current i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; int32[4] _rm_total_angle
   (dotimes (i (length _rm_total_angle))
     (setf (elt _rm_total_angle i) (sys::peek buf ptr- :integer)) (incf ptr- 4)
     )
   ;; int16[4] _FB_auv_pit
   (dotimes (i (length _FB_auv_pit))
     (setf (elt _FB_auv_pit i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; int16[4] _FB_auv_rol
   (dotimes (i (length _FB_auv_rol))
     (setf (elt _FB_auv_rol i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; int16[4] _FB_auv_yaw
   (dotimes (i (length _FB_auv_yaw))
     (setf (elt _FB_auv_yaw i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; int16[4] _FB_auv_deep
   (dotimes (i (length _FB_auv_deep))
     (setf (elt _FB_auv_deep i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; int16[4] _FB_auv_deep_vel
   (dotimes (i (length _FB_auv_deep_vel))
     (setf (elt _FB_auv_deep_vel i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; int16[4] _FB_auv_ang_vel_pit
   (dotimes (i (length _FB_auv_ang_vel_pit))
     (setf (elt _FB_auv_ang_vel_pit i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; int16[4] _FB_auv_ang_vel_rol
   (dotimes (i (length _FB_auv_ang_vel_rol))
     (setf (elt _FB_auv_ang_vel_rol i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; int16[4] _FB_auv_ang_vel_yaw
   (dotimes (i (length _FB_auv_ang_vel_yaw))
     (setf (elt _FB_auv_ang_vel_yaw i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; int16[8] _PWM_set
   (dotimes (i (length _PWM_set))
     (setf (elt _PWM_set i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;;
   self)
  )

(setf (get tauv_msgs::Can :md5sum-) "38928799346a1e1fb02c8f1ad6011cc1")
(setf (get tauv_msgs::Can :datatype-) "tauv_msgs/Can")
(setf (get tauv_msgs::Can :definition-)
      "uint8 voltage_current
uint8 temperature_cabin_current

int16[4] rm_speed_rpm
int16[4] rm_given_current
int32[4] rm_total_angle

int16[4] FB_auv_pit
int16[4] FB_auv_rol
int16[4] FB_auv_yaw
int16[4] FB_auv_deep
int16[4] FB_auv_deep_vel
int16[4] FB_auv_ang_vel_pit
int16[4] FB_auv_ang_vel_rol
int16[4] FB_auv_ang_vel_yaw

int16[8] PWM_set

")



(provide :tauv_msgs/Can "38928799346a1e1fb02c8f1ad6011cc1")


