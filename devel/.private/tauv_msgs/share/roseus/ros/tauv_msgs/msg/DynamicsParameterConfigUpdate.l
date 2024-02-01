;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::DynamicsParameterConfigUpdate)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'DynamicsParameterConfigUpdate (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::DYNAMICSPARAMETERCONFIGUPDATE")
  (make-package "TAUV_MSGS::DYNAMICSPARAMETERCONFIGUPDATE"))

(in-package "ROS")
;;//! \htmlinclude DynamicsParameterConfigUpdate.msg.html


(defclass tauv_msgs::DynamicsParameterConfigUpdate
  :super ros::object
  :slots (_name _update_initial_value _initial_value _update_fixed _fixed _update_initial_covariance _initial_covariance _update_process_covariance _process_covariance _update_limits _limits _reset ))

(defmethod tauv_msgs::DynamicsParameterConfigUpdate
  (:init
   (&key
    ((:name __name) "")
    ((:update_initial_value __update_initial_value) nil)
    ((:initial_value __initial_value) 0.0)
    ((:update_fixed __update_fixed) nil)
    ((:fixed __fixed) nil)
    ((:update_initial_covariance __update_initial_covariance) nil)
    ((:initial_covariance __initial_covariance) 0.0)
    ((:update_process_covariance __update_process_covariance) nil)
    ((:process_covariance __process_covariance) 0.0)
    ((:update_limits __update_limits) nil)
    ((:limits __limits) (make-array 2 :initial-element 0.0 :element-type :float))
    ((:reset __reset) nil)
    )
   (send-super :init)
   (setq _name (string __name))
   (setq _update_initial_value __update_initial_value)
   (setq _initial_value (float __initial_value))
   (setq _update_fixed __update_fixed)
   (setq _fixed __fixed)
   (setq _update_initial_covariance __update_initial_covariance)
   (setq _initial_covariance (float __initial_covariance))
   (setq _update_process_covariance __update_process_covariance)
   (setq _process_covariance (float __process_covariance))
   (setq _update_limits __update_limits)
   (setq _limits __limits)
   (setq _reset __reset)
   self)
  (:name
   (&optional __name)
   (if __name (setq _name __name)) _name)
  (:update_initial_value
   (&optional (__update_initial_value :null))
   (if (not (eq __update_initial_value :null)) (setq _update_initial_value __update_initial_value)) _update_initial_value)
  (:initial_value
   (&optional __initial_value)
   (if __initial_value (setq _initial_value __initial_value)) _initial_value)
  (:update_fixed
   (&optional (__update_fixed :null))
   (if (not (eq __update_fixed :null)) (setq _update_fixed __update_fixed)) _update_fixed)
  (:fixed
   (&optional (__fixed :null))
   (if (not (eq __fixed :null)) (setq _fixed __fixed)) _fixed)
  (:update_initial_covariance
   (&optional (__update_initial_covariance :null))
   (if (not (eq __update_initial_covariance :null)) (setq _update_initial_covariance __update_initial_covariance)) _update_initial_covariance)
  (:initial_covariance
   (&optional __initial_covariance)
   (if __initial_covariance (setq _initial_covariance __initial_covariance)) _initial_covariance)
  (:update_process_covariance
   (&optional (__update_process_covariance :null))
   (if (not (eq __update_process_covariance :null)) (setq _update_process_covariance __update_process_covariance)) _update_process_covariance)
  (:process_covariance
   (&optional __process_covariance)
   (if __process_covariance (setq _process_covariance __process_covariance)) _process_covariance)
  (:update_limits
   (&optional (__update_limits :null))
   (if (not (eq __update_limits :null)) (setq _update_limits __update_limits)) _update_limits)
  (:limits
   (&optional __limits)
   (if __limits (setq _limits __limits)) _limits)
  (:reset
   (&optional (__reset :null))
   (if (not (eq __reset :null)) (setq _reset __reset)) _reset)
  (:serialization-length
   ()
   (+
    ;; string _name
    4 (length _name)
    ;; bool _update_initial_value
    1
    ;; float64 _initial_value
    8
    ;; bool _update_fixed
    1
    ;; bool _fixed
    1
    ;; bool _update_initial_covariance
    1
    ;; float64 _initial_covariance
    8
    ;; bool _update_process_covariance
    1
    ;; float64 _process_covariance
    8
    ;; bool _update_limits
    1
    ;; float64[2] _limits
    (* 8    2)
    ;; bool _reset
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _name
       (write-long (length _name) s) (princ _name s)
     ;; bool _update_initial_value
       (if _update_initial_value (write-byte -1 s) (write-byte 0 s))
     ;; float64 _initial_value
       (sys::poke _initial_value (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; bool _update_fixed
       (if _update_fixed (write-byte -1 s) (write-byte 0 s))
     ;; bool _fixed
       (if _fixed (write-byte -1 s) (write-byte 0 s))
     ;; bool _update_initial_covariance
       (if _update_initial_covariance (write-byte -1 s) (write-byte 0 s))
     ;; float64 _initial_covariance
       (sys::poke _initial_covariance (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; bool _update_process_covariance
       (if _update_process_covariance (write-byte -1 s) (write-byte 0 s))
     ;; float64 _process_covariance
       (sys::poke _process_covariance (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; bool _update_limits
       (if _update_limits (write-byte -1 s) (write-byte 0 s))
     ;; float64[2] _limits
     (dotimes (i 2)
       (sys::poke (elt _limits i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; bool _reset
       (if _reset (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _name
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _name (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; bool _update_initial_value
     (setq _update_initial_value (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float64 _initial_value
     (setq _initial_value (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; bool _update_fixed
     (setq _update_fixed (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _fixed
     (setq _fixed (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _update_initial_covariance
     (setq _update_initial_covariance (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float64 _initial_covariance
     (setq _initial_covariance (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; bool _update_process_covariance
     (setq _update_process_covariance (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float64 _process_covariance
     (setq _process_covariance (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; bool _update_limits
     (setq _update_limits (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float64[2] _limits
   (dotimes (i (length _limits))
     (setf (elt _limits i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; bool _reset
     (setq _reset (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(setf (get tauv_msgs::DynamicsParameterConfigUpdate :md5sum-) "31294fd8c67bf91ba516e5502711b385")
(setf (get tauv_msgs::DynamicsParameterConfigUpdate :datatype-) "tauv_msgs/DynamicsParameterConfigUpdate")
(setf (get tauv_msgs::DynamicsParameterConfigUpdate :definition-)
      "string name

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



")



(provide :tauv_msgs/DynamicsParameterConfigUpdate "31294fd8c67bf91ba516e5502711b385")


