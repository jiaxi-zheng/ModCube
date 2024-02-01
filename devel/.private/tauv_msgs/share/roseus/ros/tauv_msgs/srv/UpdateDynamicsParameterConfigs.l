;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::UpdateDynamicsParameterConfigs)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'UpdateDynamicsParameterConfigs (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::UPDATEDYNAMICSPARAMETERCONFIGS")
  (make-package "TAUV_MSGS::UPDATEDYNAMICSPARAMETERCONFIGS"))
(unless (find-package "TAUV_MSGS::UPDATEDYNAMICSPARAMETERCONFIGSREQUEST")
  (make-package "TAUV_MSGS::UPDATEDYNAMICSPARAMETERCONFIGSREQUEST"))
(unless (find-package "TAUV_MSGS::UPDATEDYNAMICSPARAMETERCONFIGSRESPONSE")
  (make-package "TAUV_MSGS::UPDATEDYNAMICSPARAMETERCONFIGSRESPONSE"))

(in-package "ROS")





(defclass tauv_msgs::UpdateDynamicsParameterConfigsRequest
  :super ros::object
  :slots (_updates ))

(defmethod tauv_msgs::UpdateDynamicsParameterConfigsRequest
  (:init
   (&key
    ((:updates __updates) ())
    )
   (send-super :init)
   (setq _updates __updates)
   self)
  (:updates
   (&rest __updates)
   (if (keywordp (car __updates))
       (send* _updates __updates)
     (progn
       (if __updates (setq _updates (car __updates)))
       _updates)))
  (:serialization-length
   ()
   (+
    ;; tauv_msgs/DynamicsParameterConfigUpdate[] _updates
    (apply #'+ (send-all _updates :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; tauv_msgs/DynamicsParameterConfigUpdate[] _updates
     (write-long (length _updates) s)
     (dolist (elem _updates)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; tauv_msgs/DynamicsParameterConfigUpdate[] _updates
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _updates (let (r) (dotimes (i n) (push (instance tauv_msgs::DynamicsParameterConfigUpdate :init) r)) r))
     (dolist (elem- _updates)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(defclass tauv_msgs::UpdateDynamicsParameterConfigsResponse
  :super ros::object
  :slots (_success ))

(defmethod tauv_msgs::UpdateDynamicsParameterConfigsResponse
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

(defclass tauv_msgs::UpdateDynamicsParameterConfigs
  :super ros::object
  :slots ())

(setf (get tauv_msgs::UpdateDynamicsParameterConfigs :md5sum-) "613592a768bbac2700fe9ff9c5b0cffe")
(setf (get tauv_msgs::UpdateDynamicsParameterConfigs :datatype-) "tauv_msgs/UpdateDynamicsParameterConfigs")
(setf (get tauv_msgs::UpdateDynamicsParameterConfigs :request) tauv_msgs::UpdateDynamicsParameterConfigsRequest)
(setf (get tauv_msgs::UpdateDynamicsParameterConfigs :response) tauv_msgs::UpdateDynamicsParameterConfigsResponse)

(defmethod tauv_msgs::UpdateDynamicsParameterConfigsRequest
  (:response () (instance tauv_msgs::UpdateDynamicsParameterConfigsResponse :init)))

(setf (get tauv_msgs::UpdateDynamicsParameterConfigsRequest :md5sum-) "613592a768bbac2700fe9ff9c5b0cffe")
(setf (get tauv_msgs::UpdateDynamicsParameterConfigsRequest :datatype-) "tauv_msgs/UpdateDynamicsParameterConfigsRequest")
(setf (get tauv_msgs::UpdateDynamicsParameterConfigsRequest :definition-)
      "tauv_msgs/DynamicsParameterConfigUpdate[] updates

================================================================================
MSG: tauv_msgs/DynamicsParameterConfigUpdate
string name

bool update_initial_value
float64 initial_value

bool update_fixed
bool fixed

bool update_initial_covariance
float64 initial_covariance

bool update_process_covariance
float64 process_covariance

bool update_limits
float64[2] limits

bool reset


---
bool success
")

(setf (get tauv_msgs::UpdateDynamicsParameterConfigsResponse :md5sum-) "613592a768bbac2700fe9ff9c5b0cffe")
(setf (get tauv_msgs::UpdateDynamicsParameterConfigsResponse :datatype-) "tauv_msgs/UpdateDynamicsParameterConfigsResponse")
(setf (get tauv_msgs::UpdateDynamicsParameterConfigsResponse :definition-)
      "tauv_msgs/DynamicsParameterConfigUpdate[] updates

================================================================================
MSG: tauv_msgs/DynamicsParameterConfigUpdate
string name

bool update_initial_value
float64 initial_value

bool update_fixed
bool fixed

bool update_initial_covariance
float64 initial_covariance

bool update_process_covariance
float64 process_covariance

bool update_limits
float64[2] limits

bool reset


---
bool success
")



(provide :tauv_msgs/UpdateDynamicsParameterConfigs "613592a768bbac2700fe9ff9c5b0cffe")


