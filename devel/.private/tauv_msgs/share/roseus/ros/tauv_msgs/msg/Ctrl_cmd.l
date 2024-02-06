;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::Ctrl_cmd)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'Ctrl_cmd (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::CTRL_CMD")
  (make-package "TAUV_MSGS::CTRL_CMD"))

(in-package "ROS")
;;//! \htmlinclude Ctrl_cmd.msg.html


(defclass tauv_msgs::Ctrl_cmd
  :super ros::object
  :slots (_Ctrl_vel_X _Ctrl_vel_Y _Ctrl_vel_Z _Ctrl_fixed_Z _Ctrl_vel_Rol _Ctrl_vel_Pit _Ctrl_vel_Yaw _Ctrl_fixed_Yaw _Ctrl_pivot_1 _Ctrl_pivot_2 _Ctrl_pivot_3 _Ctrl_pivot_4 _Ctrl_emagnet_1 _Ctrl_emagnet_2 _Ctrl_emagnet_3 _Ctrl_emagnet_4 _Ctrl_arm_joint_1 _Ctrl_arm_joint_2 _Joy_Button_Y _Joy_Button_X _Joy_Button_A _Joy_Button_B _Joy_Button_LB _Joy_Button_RB _Joy_Button_STICK_LEFT _Joy_Button_STICK_RIGHT _Reset_pwm ))

(defmethod tauv_msgs::Ctrl_cmd
  (:init
   (&key
    ((:Ctrl_vel_X __Ctrl_vel_X) (make-array 4 :initial-element 0 :element-type :char))
    ((:Ctrl_vel_Y __Ctrl_vel_Y) (make-array 4 :initial-element 0 :element-type :char))
    ((:Ctrl_vel_Z __Ctrl_vel_Z) (make-array 4 :initial-element 0 :element-type :char))
    ((:Ctrl_fixed_Z __Ctrl_fixed_Z) (make-array 4 :initial-element 0 :element-type :char))
    ((:Ctrl_vel_Rol __Ctrl_vel_Rol) (make-array 4 :initial-element 0 :element-type :char))
    ((:Ctrl_vel_Pit __Ctrl_vel_Pit) (make-array 4 :initial-element 0 :element-type :char))
    ((:Ctrl_vel_Yaw __Ctrl_vel_Yaw) (make-array 4 :initial-element 0 :element-type :char))
    ((:Ctrl_fixed_Yaw __Ctrl_fixed_Yaw) (make-array 4 :initial-element 0 :element-type :char))
    ((:Ctrl_pivot_1 __Ctrl_pivot_1) (make-array 4 :initial-element 0 :element-type :char))
    ((:Ctrl_pivot_2 __Ctrl_pivot_2) (make-array 4 :initial-element 0 :element-type :char))
    ((:Ctrl_pivot_3 __Ctrl_pivot_3) (make-array 4 :initial-element 0 :element-type :char))
    ((:Ctrl_pivot_4 __Ctrl_pivot_4) (make-array 4 :initial-element 0 :element-type :char))
    ((:Ctrl_emagnet_1 __Ctrl_emagnet_1) (make-array 4 :initial-element 0 :element-type :char))
    ((:Ctrl_emagnet_2 __Ctrl_emagnet_2) (make-array 4 :initial-element 0 :element-type :char))
    ((:Ctrl_emagnet_3 __Ctrl_emagnet_3) (make-array 4 :initial-element 0 :element-type :char))
    ((:Ctrl_emagnet_4 __Ctrl_emagnet_4) (make-array 4 :initial-element 0 :element-type :char))
    ((:Ctrl_arm_joint_1 __Ctrl_arm_joint_1) (make-array 4 :initial-element 0 :element-type :char))
    ((:Ctrl_arm_joint_2 __Ctrl_arm_joint_2) (make-array 4 :initial-element 0 :element-type :char))
    ((:Joy_Button_Y __Joy_Button_Y) (make-array 4 :initial-element 0 :element-type :integer))
    ((:Joy_Button_X __Joy_Button_X) (make-array 4 :initial-element 0 :element-type :integer))
    ((:Joy_Button_A __Joy_Button_A) (make-array 4 :initial-element 0 :element-type :integer))
    ((:Joy_Button_B __Joy_Button_B) (make-array 4 :initial-element 0 :element-type :integer))
    ((:Joy_Button_LB __Joy_Button_LB) (make-array 4 :initial-element 0 :element-type :integer))
    ((:Joy_Button_RB __Joy_Button_RB) (make-array 4 :initial-element 0 :element-type :integer))
    ((:Joy_Button_STICK_LEFT __Joy_Button_STICK_LEFT) (make-array 4 :initial-element 0 :element-type :integer))
    ((:Joy_Button_STICK_RIGHT __Joy_Button_STICK_RIGHT) (make-array 4 :initial-element 0 :element-type :integer))
    ((:Reset_pwm __Reset_pwm) (make-array 4 :initial-element 0 :element-type :integer))
    )
   (send-super :init)
   (setq _Ctrl_vel_X __Ctrl_vel_X)
   (setq _Ctrl_vel_Y __Ctrl_vel_Y)
   (setq _Ctrl_vel_Z __Ctrl_vel_Z)
   (setq _Ctrl_fixed_Z __Ctrl_fixed_Z)
   (setq _Ctrl_vel_Rol __Ctrl_vel_Rol)
   (setq _Ctrl_vel_Pit __Ctrl_vel_Pit)
   (setq _Ctrl_vel_Yaw __Ctrl_vel_Yaw)
   (setq _Ctrl_fixed_Yaw __Ctrl_fixed_Yaw)
   (setq _Ctrl_pivot_1 __Ctrl_pivot_1)
   (setq _Ctrl_pivot_2 __Ctrl_pivot_2)
   (setq _Ctrl_pivot_3 __Ctrl_pivot_3)
   (setq _Ctrl_pivot_4 __Ctrl_pivot_4)
   (setq _Ctrl_emagnet_1 __Ctrl_emagnet_1)
   (setq _Ctrl_emagnet_2 __Ctrl_emagnet_2)
   (setq _Ctrl_emagnet_3 __Ctrl_emagnet_3)
   (setq _Ctrl_emagnet_4 __Ctrl_emagnet_4)
   (setq _Ctrl_arm_joint_1 __Ctrl_arm_joint_1)
   (setq _Ctrl_arm_joint_2 __Ctrl_arm_joint_2)
   (setq _Joy_Button_Y __Joy_Button_Y)
   (setq _Joy_Button_X __Joy_Button_X)
   (setq _Joy_Button_A __Joy_Button_A)
   (setq _Joy_Button_B __Joy_Button_B)
   (setq _Joy_Button_LB __Joy_Button_LB)
   (setq _Joy_Button_RB __Joy_Button_RB)
   (setq _Joy_Button_STICK_LEFT __Joy_Button_STICK_LEFT)
   (setq _Joy_Button_STICK_RIGHT __Joy_Button_STICK_RIGHT)
   (setq _Reset_pwm __Reset_pwm)
   self)
  (:Ctrl_vel_X
   (&optional __Ctrl_vel_X)
   (if __Ctrl_vel_X (setq _Ctrl_vel_X __Ctrl_vel_X)) _Ctrl_vel_X)
  (:Ctrl_vel_Y
   (&optional __Ctrl_vel_Y)
   (if __Ctrl_vel_Y (setq _Ctrl_vel_Y __Ctrl_vel_Y)) _Ctrl_vel_Y)
  (:Ctrl_vel_Z
   (&optional __Ctrl_vel_Z)
   (if __Ctrl_vel_Z (setq _Ctrl_vel_Z __Ctrl_vel_Z)) _Ctrl_vel_Z)
  (:Ctrl_fixed_Z
   (&optional __Ctrl_fixed_Z)
   (if __Ctrl_fixed_Z (setq _Ctrl_fixed_Z __Ctrl_fixed_Z)) _Ctrl_fixed_Z)
  (:Ctrl_vel_Rol
   (&optional __Ctrl_vel_Rol)
   (if __Ctrl_vel_Rol (setq _Ctrl_vel_Rol __Ctrl_vel_Rol)) _Ctrl_vel_Rol)
  (:Ctrl_vel_Pit
   (&optional __Ctrl_vel_Pit)
   (if __Ctrl_vel_Pit (setq _Ctrl_vel_Pit __Ctrl_vel_Pit)) _Ctrl_vel_Pit)
  (:Ctrl_vel_Yaw
   (&optional __Ctrl_vel_Yaw)
   (if __Ctrl_vel_Yaw (setq _Ctrl_vel_Yaw __Ctrl_vel_Yaw)) _Ctrl_vel_Yaw)
  (:Ctrl_fixed_Yaw
   (&optional __Ctrl_fixed_Yaw)
   (if __Ctrl_fixed_Yaw (setq _Ctrl_fixed_Yaw __Ctrl_fixed_Yaw)) _Ctrl_fixed_Yaw)
  (:Ctrl_pivot_1
   (&optional __Ctrl_pivot_1)
   (if __Ctrl_pivot_1 (setq _Ctrl_pivot_1 __Ctrl_pivot_1)) _Ctrl_pivot_1)
  (:Ctrl_pivot_2
   (&optional __Ctrl_pivot_2)
   (if __Ctrl_pivot_2 (setq _Ctrl_pivot_2 __Ctrl_pivot_2)) _Ctrl_pivot_2)
  (:Ctrl_pivot_3
   (&optional __Ctrl_pivot_3)
   (if __Ctrl_pivot_3 (setq _Ctrl_pivot_3 __Ctrl_pivot_3)) _Ctrl_pivot_3)
  (:Ctrl_pivot_4
   (&optional __Ctrl_pivot_4)
   (if __Ctrl_pivot_4 (setq _Ctrl_pivot_4 __Ctrl_pivot_4)) _Ctrl_pivot_4)
  (:Ctrl_emagnet_1
   (&optional __Ctrl_emagnet_1)
   (if __Ctrl_emagnet_1 (setq _Ctrl_emagnet_1 __Ctrl_emagnet_1)) _Ctrl_emagnet_1)
  (:Ctrl_emagnet_2
   (&optional __Ctrl_emagnet_2)
   (if __Ctrl_emagnet_2 (setq _Ctrl_emagnet_2 __Ctrl_emagnet_2)) _Ctrl_emagnet_2)
  (:Ctrl_emagnet_3
   (&optional __Ctrl_emagnet_3)
   (if __Ctrl_emagnet_3 (setq _Ctrl_emagnet_3 __Ctrl_emagnet_3)) _Ctrl_emagnet_3)
  (:Ctrl_emagnet_4
   (&optional __Ctrl_emagnet_4)
   (if __Ctrl_emagnet_4 (setq _Ctrl_emagnet_4 __Ctrl_emagnet_4)) _Ctrl_emagnet_4)
  (:Ctrl_arm_joint_1
   (&optional __Ctrl_arm_joint_1)
   (if __Ctrl_arm_joint_1 (setq _Ctrl_arm_joint_1 __Ctrl_arm_joint_1)) _Ctrl_arm_joint_1)
  (:Ctrl_arm_joint_2
   (&optional __Ctrl_arm_joint_2)
   (if __Ctrl_arm_joint_2 (setq _Ctrl_arm_joint_2 __Ctrl_arm_joint_2)) _Ctrl_arm_joint_2)
  (:Joy_Button_Y
   (&optional __Joy_Button_Y)
   (if __Joy_Button_Y (setq _Joy_Button_Y __Joy_Button_Y)) _Joy_Button_Y)
  (:Joy_Button_X
   (&optional __Joy_Button_X)
   (if __Joy_Button_X (setq _Joy_Button_X __Joy_Button_X)) _Joy_Button_X)
  (:Joy_Button_A
   (&optional __Joy_Button_A)
   (if __Joy_Button_A (setq _Joy_Button_A __Joy_Button_A)) _Joy_Button_A)
  (:Joy_Button_B
   (&optional __Joy_Button_B)
   (if __Joy_Button_B (setq _Joy_Button_B __Joy_Button_B)) _Joy_Button_B)
  (:Joy_Button_LB
   (&optional __Joy_Button_LB)
   (if __Joy_Button_LB (setq _Joy_Button_LB __Joy_Button_LB)) _Joy_Button_LB)
  (:Joy_Button_RB
   (&optional __Joy_Button_RB)
   (if __Joy_Button_RB (setq _Joy_Button_RB __Joy_Button_RB)) _Joy_Button_RB)
  (:Joy_Button_STICK_LEFT
   (&optional __Joy_Button_STICK_LEFT)
   (if __Joy_Button_STICK_LEFT (setq _Joy_Button_STICK_LEFT __Joy_Button_STICK_LEFT)) _Joy_Button_STICK_LEFT)
  (:Joy_Button_STICK_RIGHT
   (&optional __Joy_Button_STICK_RIGHT)
   (if __Joy_Button_STICK_RIGHT (setq _Joy_Button_STICK_RIGHT __Joy_Button_STICK_RIGHT)) _Joy_Button_STICK_RIGHT)
  (:Reset_pwm
   (&optional __Reset_pwm)
   (if __Reset_pwm (setq _Reset_pwm __Reset_pwm)) _Reset_pwm)
  (:serialization-length
   ()
   (+
    ;; uint8[4] _Ctrl_vel_X
    (* 1    4)
    ;; uint8[4] _Ctrl_vel_Y
    (* 1    4)
    ;; uint8[4] _Ctrl_vel_Z
    (* 1    4)
    ;; uint8[4] _Ctrl_fixed_Z
    (* 1    4)
    ;; uint8[4] _Ctrl_vel_Rol
    (* 1    4)
    ;; uint8[4] _Ctrl_vel_Pit
    (* 1    4)
    ;; uint8[4] _Ctrl_vel_Yaw
    (* 1    4)
    ;; uint8[4] _Ctrl_fixed_Yaw
    (* 1    4)
    ;; uint8[4] _Ctrl_pivot_1
    (* 1    4)
    ;; uint8[4] _Ctrl_pivot_2
    (* 1    4)
    ;; uint8[4] _Ctrl_pivot_3
    (* 1    4)
    ;; uint8[4] _Ctrl_pivot_4
    (* 1    4)
    ;; uint8[4] _Ctrl_emagnet_1
    (* 1    4)
    ;; uint8[4] _Ctrl_emagnet_2
    (* 1    4)
    ;; uint8[4] _Ctrl_emagnet_3
    (* 1    4)
    ;; uint8[4] _Ctrl_emagnet_4
    (* 1    4)
    ;; uint8[4] _Ctrl_arm_joint_1
    (* 1    4)
    ;; uint8[4] _Ctrl_arm_joint_2
    (* 1    4)
    ;; int16[4] _Joy_Button_Y
    (* 2    4)
    ;; int16[4] _Joy_Button_X
    (* 2    4)
    ;; int16[4] _Joy_Button_A
    (* 2    4)
    ;; int16[4] _Joy_Button_B
    (* 2    4)
    ;; int16[4] _Joy_Button_LB
    (* 2    4)
    ;; int16[4] _Joy_Button_RB
    (* 2    4)
    ;; int16[4] _Joy_Button_STICK_LEFT
    (* 2    4)
    ;; int16[4] _Joy_Button_STICK_RIGHT
    (* 2    4)
    ;; int16[4] _Reset_pwm
    (* 2    4)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8[4] _Ctrl_vel_X
     (princ _Ctrl_vel_X s)
     ;; uint8[4] _Ctrl_vel_Y
     (princ _Ctrl_vel_Y s)
     ;; uint8[4] _Ctrl_vel_Z
     (princ _Ctrl_vel_Z s)
     ;; uint8[4] _Ctrl_fixed_Z
     (princ _Ctrl_fixed_Z s)
     ;; uint8[4] _Ctrl_vel_Rol
     (princ _Ctrl_vel_Rol s)
     ;; uint8[4] _Ctrl_vel_Pit
     (princ _Ctrl_vel_Pit s)
     ;; uint8[4] _Ctrl_vel_Yaw
     (princ _Ctrl_vel_Yaw s)
     ;; uint8[4] _Ctrl_fixed_Yaw
     (princ _Ctrl_fixed_Yaw s)
     ;; uint8[4] _Ctrl_pivot_1
     (princ _Ctrl_pivot_1 s)
     ;; uint8[4] _Ctrl_pivot_2
     (princ _Ctrl_pivot_2 s)
     ;; uint8[4] _Ctrl_pivot_3
     (princ _Ctrl_pivot_3 s)
     ;; uint8[4] _Ctrl_pivot_4
     (princ _Ctrl_pivot_4 s)
     ;; uint8[4] _Ctrl_emagnet_1
     (princ _Ctrl_emagnet_1 s)
     ;; uint8[4] _Ctrl_emagnet_2
     (princ _Ctrl_emagnet_2 s)
     ;; uint8[4] _Ctrl_emagnet_3
     (princ _Ctrl_emagnet_3 s)
     ;; uint8[4] _Ctrl_emagnet_4
     (princ _Ctrl_emagnet_4 s)
     ;; uint8[4] _Ctrl_arm_joint_1
     (princ _Ctrl_arm_joint_1 s)
     ;; uint8[4] _Ctrl_arm_joint_2
     (princ _Ctrl_arm_joint_2 s)
     ;; int16[4] _Joy_Button_Y
     (dotimes (i 4)
       (write-word (elt _Joy_Button_Y i) s)
       )
     ;; int16[4] _Joy_Button_X
     (dotimes (i 4)
       (write-word (elt _Joy_Button_X i) s)
       )
     ;; int16[4] _Joy_Button_A
     (dotimes (i 4)
       (write-word (elt _Joy_Button_A i) s)
       )
     ;; int16[4] _Joy_Button_B
     (dotimes (i 4)
       (write-word (elt _Joy_Button_B i) s)
       )
     ;; int16[4] _Joy_Button_LB
     (dotimes (i 4)
       (write-word (elt _Joy_Button_LB i) s)
       )
     ;; int16[4] _Joy_Button_RB
     (dotimes (i 4)
       (write-word (elt _Joy_Button_RB i) s)
       )
     ;; int16[4] _Joy_Button_STICK_LEFT
     (dotimes (i 4)
       (write-word (elt _Joy_Button_STICK_LEFT i) s)
       )
     ;; int16[4] _Joy_Button_STICK_RIGHT
     (dotimes (i 4)
       (write-word (elt _Joy_Button_STICK_RIGHT i) s)
       )
     ;; int16[4] _Reset_pwm
     (dotimes (i 4)
       (write-word (elt _Reset_pwm i) s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8[4] _Ctrl_vel_X
   (setq _Ctrl_vel_X (make-array 4 :element-type :char))
   (replace _Ctrl_vel_X buf :start2 ptr-) (incf ptr- 4)
   ;; uint8[4] _Ctrl_vel_Y
   (setq _Ctrl_vel_Y (make-array 4 :element-type :char))
   (replace _Ctrl_vel_Y buf :start2 ptr-) (incf ptr- 4)
   ;; uint8[4] _Ctrl_vel_Z
   (setq _Ctrl_vel_Z (make-array 4 :element-type :char))
   (replace _Ctrl_vel_Z buf :start2 ptr-) (incf ptr- 4)
   ;; uint8[4] _Ctrl_fixed_Z
   (setq _Ctrl_fixed_Z (make-array 4 :element-type :char))
   (replace _Ctrl_fixed_Z buf :start2 ptr-) (incf ptr- 4)
   ;; uint8[4] _Ctrl_vel_Rol
   (setq _Ctrl_vel_Rol (make-array 4 :element-type :char))
   (replace _Ctrl_vel_Rol buf :start2 ptr-) (incf ptr- 4)
   ;; uint8[4] _Ctrl_vel_Pit
   (setq _Ctrl_vel_Pit (make-array 4 :element-type :char))
   (replace _Ctrl_vel_Pit buf :start2 ptr-) (incf ptr- 4)
   ;; uint8[4] _Ctrl_vel_Yaw
   (setq _Ctrl_vel_Yaw (make-array 4 :element-type :char))
   (replace _Ctrl_vel_Yaw buf :start2 ptr-) (incf ptr- 4)
   ;; uint8[4] _Ctrl_fixed_Yaw
   (setq _Ctrl_fixed_Yaw (make-array 4 :element-type :char))
   (replace _Ctrl_fixed_Yaw buf :start2 ptr-) (incf ptr- 4)
   ;; uint8[4] _Ctrl_pivot_1
   (setq _Ctrl_pivot_1 (make-array 4 :element-type :char))
   (replace _Ctrl_pivot_1 buf :start2 ptr-) (incf ptr- 4)
   ;; uint8[4] _Ctrl_pivot_2
   (setq _Ctrl_pivot_2 (make-array 4 :element-type :char))
   (replace _Ctrl_pivot_2 buf :start2 ptr-) (incf ptr- 4)
   ;; uint8[4] _Ctrl_pivot_3
   (setq _Ctrl_pivot_3 (make-array 4 :element-type :char))
   (replace _Ctrl_pivot_3 buf :start2 ptr-) (incf ptr- 4)
   ;; uint8[4] _Ctrl_pivot_4
   (setq _Ctrl_pivot_4 (make-array 4 :element-type :char))
   (replace _Ctrl_pivot_4 buf :start2 ptr-) (incf ptr- 4)
   ;; uint8[4] _Ctrl_emagnet_1
   (setq _Ctrl_emagnet_1 (make-array 4 :element-type :char))
   (replace _Ctrl_emagnet_1 buf :start2 ptr-) (incf ptr- 4)
   ;; uint8[4] _Ctrl_emagnet_2
   (setq _Ctrl_emagnet_2 (make-array 4 :element-type :char))
   (replace _Ctrl_emagnet_2 buf :start2 ptr-) (incf ptr- 4)
   ;; uint8[4] _Ctrl_emagnet_3
   (setq _Ctrl_emagnet_3 (make-array 4 :element-type :char))
   (replace _Ctrl_emagnet_3 buf :start2 ptr-) (incf ptr- 4)
   ;; uint8[4] _Ctrl_emagnet_4
   (setq _Ctrl_emagnet_4 (make-array 4 :element-type :char))
   (replace _Ctrl_emagnet_4 buf :start2 ptr-) (incf ptr- 4)
   ;; uint8[4] _Ctrl_arm_joint_1
   (setq _Ctrl_arm_joint_1 (make-array 4 :element-type :char))
   (replace _Ctrl_arm_joint_1 buf :start2 ptr-) (incf ptr- 4)
   ;; uint8[4] _Ctrl_arm_joint_2
   (setq _Ctrl_arm_joint_2 (make-array 4 :element-type :char))
   (replace _Ctrl_arm_joint_2 buf :start2 ptr-) (incf ptr- 4)
   ;; int16[4] _Joy_Button_Y
   (dotimes (i (length _Joy_Button_Y))
     (setf (elt _Joy_Button_Y i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; int16[4] _Joy_Button_X
   (dotimes (i (length _Joy_Button_X))
     (setf (elt _Joy_Button_X i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; int16[4] _Joy_Button_A
   (dotimes (i (length _Joy_Button_A))
     (setf (elt _Joy_Button_A i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; int16[4] _Joy_Button_B
   (dotimes (i (length _Joy_Button_B))
     (setf (elt _Joy_Button_B i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; int16[4] _Joy_Button_LB
   (dotimes (i (length _Joy_Button_LB))
     (setf (elt _Joy_Button_LB i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; int16[4] _Joy_Button_RB
   (dotimes (i (length _Joy_Button_RB))
     (setf (elt _Joy_Button_RB i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; int16[4] _Joy_Button_STICK_LEFT
   (dotimes (i (length _Joy_Button_STICK_LEFT))
     (setf (elt _Joy_Button_STICK_LEFT i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; int16[4] _Joy_Button_STICK_RIGHT
   (dotimes (i (length _Joy_Button_STICK_RIGHT))
     (setf (elt _Joy_Button_STICK_RIGHT i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; int16[4] _Reset_pwm
   (dotimes (i (length _Reset_pwm))
     (setf (elt _Reset_pwm i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;;
   self)
  )

(setf (get tauv_msgs::Ctrl_cmd :md5sum-) "7a746ffc64e1d5a26fb5d6cba04e63c2")
(setf (get tauv_msgs::Ctrl_cmd :datatype-) "tauv_msgs/Ctrl_cmd")
(setf (get tauv_msgs::Ctrl_cmd :definition-)
      "
uint8[4] Ctrl_vel_X
uint8[4] Ctrl_vel_Y
uint8[4] Ctrl_vel_Z
uint8[4] Ctrl_fixed_Z
uint8[4] Ctrl_vel_Rol
uint8[4] Ctrl_vel_Pit
uint8[4] Ctrl_vel_Yaw
uint8[4] Ctrl_fixed_Yaw

uint8[4] Ctrl_pivot_1
uint8[4] Ctrl_pivot_2
uint8[4] Ctrl_pivot_3
uint8[4] Ctrl_pivot_4

uint8[4] Ctrl_emagnet_1
uint8[4] Ctrl_emagnet_2
uint8[4] Ctrl_emagnet_3
uint8[4] Ctrl_emagnet_4

uint8[4] Ctrl_arm_joint_1
uint8[4] Ctrl_arm_joint_2

int16[4] Joy_Button_Y
int16[4] Joy_Button_X
int16[4] Joy_Button_A
int16[4] Joy_Button_B
int16[4] Joy_Button_LB
int16[4] Joy_Button_RB
int16[4] Joy_Button_STICK_LEFT
int16[4] Joy_Button_STICK_RIGHT
int16[4] Reset_pwm

")



(provide :tauv_msgs/Ctrl_cmd "7a746ffc64e1d5a26fb5d6cba04e63c2")


