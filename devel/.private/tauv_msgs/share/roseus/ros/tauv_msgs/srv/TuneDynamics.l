;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::TuneDynamics)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'TuneDynamics (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::TUNEDYNAMICS")
  (make-package "TAUV_MSGS::TUNEDYNAMICS"))
(unless (find-package "TAUV_MSGS::TUNEDYNAMICSREQUEST")
  (make-package "TAUV_MSGS::TUNEDYNAMICSREQUEST"))
(unless (find-package "TAUV_MSGS::TUNEDYNAMICSRESPONSE")
  (make-package "TAUV_MSGS::TUNEDYNAMICSRESPONSE"))

(in-package "ROS")





(defclass tauv_msgs::TuneDynamicsRequest
  :super ros::object
  :slots (_tuning ))

(defmethod tauv_msgs::TuneDynamicsRequest
  (:init
   (&key
    ((:tuning __tuning) (instance tauv_msgs::DynamicsTuning :init))
    )
   (send-super :init)
   (setq _tuning __tuning)
   self)
  (:tuning
   (&rest __tuning)
   (if (keywordp (car __tuning))
       (send* _tuning __tuning)
     (progn
       (if __tuning (setq _tuning (car __tuning)))
       _tuning)))
  (:serialization-length
   ()
   (+
    ;; tauv_msgs/DynamicsTuning _tuning
    (send _tuning :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; tauv_msgs/DynamicsTuning _tuning
       (send _tuning :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; tauv_msgs/DynamicsTuning _tuning
     (send _tuning :deserialize buf ptr-) (incf ptr- (send _tuning :serialization-length))
   ;;
   self)
  )

(defclass tauv_msgs::TuneDynamicsResponse
  :super ros::object
  :slots (_success ))

(defmethod tauv_msgs::TuneDynamicsResponse
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

(defclass tauv_msgs::TuneDynamics
  :super ros::object
  :slots ())

(setf (get tauv_msgs::TuneDynamics :md5sum-) "7d0bd5eafc7372a83c5502516a094dbc")
(setf (get tauv_msgs::TuneDynamics :datatype-) "tauv_msgs/TuneDynamics")
(setf (get tauv_msgs::TuneDynamics :request) tauv_msgs::TuneDynamicsRequest)
(setf (get tauv_msgs::TuneDynamics :response) tauv_msgs::TuneDynamicsResponse)

(defmethod tauv_msgs::TuneDynamicsRequest
  (:response () (instance tauv_msgs::TuneDynamicsResponse :init)))

(setf (get tauv_msgs::TuneDynamicsRequest :md5sum-) "7d0bd5eafc7372a83c5502516a094dbc")
(setf (get tauv_msgs::TuneDynamicsRequest :datatype-) "tauv_msgs/TuneDynamicsRequest")
(setf (get tauv_msgs::TuneDynamicsRequest :definition-)
      "tauv_msgs/DynamicsTuning tuning

================================================================================
MSG: tauv_msgs/DynamicsTuning
bool update_mass
float64 mass
bool update_volume
float64 volume
bool update_water_density
float64 water_density
bool update_center_of_gravity
float64[3] center_of_gravity
bool update_center_of_buoyancy
float64[3] center_of_buoyancy
bool update_moments
float64[6] moments
bool update_linear_damping
float64[6] linear_damping
bool update_quadratic_damping
float64[6] quadratic_damping
bool update_added_mass
---
bool success
")

(setf (get tauv_msgs::TuneDynamicsResponse :md5sum-) "7d0bd5eafc7372a83c5502516a094dbc")
(setf (get tauv_msgs::TuneDynamicsResponse :datatype-) "tauv_msgs/TuneDynamicsResponse")
(setf (get tauv_msgs::TuneDynamicsResponse :definition-)
      "tauv_msgs/DynamicsTuning tuning

================================================================================
MSG: tauv_msgs/DynamicsTuning
bool update_mass
float64 mass
bool update_volume
float64 volume
bool update_water_density
float64 water_density
bool update_center_of_gravity
float64[3] center_of_gravity
bool update_center_of_buoyancy
float64[3] center_of_buoyancy
bool update_moments
float64[6] moments
bool update_linear_damping
float64[6] linear_damping
bool update_quadratic_damping
float64[6] quadratic_damping
bool update_added_mass
---
bool success
")



(provide :tauv_msgs/TuneDynamics "7d0bd5eafc7372a83c5502516a094dbc")


