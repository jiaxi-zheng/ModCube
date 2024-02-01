;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::GateDetection)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'GateDetection (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::GATEDETECTION")
  (make-package "TAUV_MSGS::GATEDETECTION"))

(in-package "ROS")
;;//! \htmlinclude GateDetection.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass tauv_msgs::GateDetection
  :super ros::object
  :slots (_leftCorner _rightCorner _centerPoint ))

(defmethod tauv_msgs::GateDetection
  (:init
   (&key
    ((:leftCorner __leftCorner) (instance geometry_msgs::Point :init))
    ((:rightCorner __rightCorner) (instance geometry_msgs::Point :init))
    ((:centerPoint __centerPoint) (instance geometry_msgs::Point :init))
    )
   (send-super :init)
   (setq _leftCorner __leftCorner)
   (setq _rightCorner __rightCorner)
   (setq _centerPoint __centerPoint)
   self)
  (:leftCorner
   (&rest __leftCorner)
   (if (keywordp (car __leftCorner))
       (send* _leftCorner __leftCorner)
     (progn
       (if __leftCorner (setq _leftCorner (car __leftCorner)))
       _leftCorner)))
  (:rightCorner
   (&rest __rightCorner)
   (if (keywordp (car __rightCorner))
       (send* _rightCorner __rightCorner)
     (progn
       (if __rightCorner (setq _rightCorner (car __rightCorner)))
       _rightCorner)))
  (:centerPoint
   (&rest __centerPoint)
   (if (keywordp (car __centerPoint))
       (send* _centerPoint __centerPoint)
     (progn
       (if __centerPoint (setq _centerPoint (car __centerPoint)))
       _centerPoint)))
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Point _leftCorner
    (send _leftCorner :serialization-length)
    ;; geometry_msgs/Point _rightCorner
    (send _rightCorner :serialization-length)
    ;; geometry_msgs/Point _centerPoint
    (send _centerPoint :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Point _leftCorner
       (send _leftCorner :serialize s)
     ;; geometry_msgs/Point _rightCorner
       (send _rightCorner :serialize s)
     ;; geometry_msgs/Point _centerPoint
       (send _centerPoint :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Point _leftCorner
     (send _leftCorner :deserialize buf ptr-) (incf ptr- (send _leftCorner :serialization-length))
   ;; geometry_msgs/Point _rightCorner
     (send _rightCorner :deserialize buf ptr-) (incf ptr- (send _rightCorner :serialization-length))
   ;; geometry_msgs/Point _centerPoint
     (send _centerPoint :deserialize buf ptr-) (incf ptr- (send _centerPoint :serialization-length))
   ;;
   self)
  )

(setf (get tauv_msgs::GateDetection :md5sum-) "8a67a29e701c68c820bd45950230c212")
(setf (get tauv_msgs::GateDetection :datatype-) "tauv_msgs/GateDetection")
(setf (get tauv_msgs::GateDetection :definition-)
      "geometry_msgs/Point leftCorner
geometry_msgs/Point rightCorner
geometry_msgs/Point centerPoint
================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

")



(provide :tauv_msgs/GateDetection "8a67a29e701c68c820bd45950230c212")


