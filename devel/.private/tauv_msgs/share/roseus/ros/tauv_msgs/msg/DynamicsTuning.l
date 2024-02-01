;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::DynamicsTuning)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'DynamicsTuning (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::DYNAMICSTUNING")
  (make-package "TAUV_MSGS::DYNAMICSTUNING"))

(in-package "ROS")
;;//! \htmlinclude DynamicsTuning.msg.html


(defclass tauv_msgs::DynamicsTuning
  :super ros::object
  :slots (_update_mass _mass _update_volume _volume _update_water_density _water_density _update_center_of_gravity _center_of_gravity _update_center_of_buoyancy _center_of_buoyancy _update_moments _moments _update_linear_damping _linear_damping _update_quadratic_damping _quadratic_damping _update_added_mass _added_mass ))

(defmethod tauv_msgs::DynamicsTuning
  (:init
   (&key
    ((:update_mass __update_mass) nil)
    ((:mass __mass) 0.0)
    ((:update_volume __update_volume) nil)
    ((:volume __volume) 0.0)
    ((:update_water_density __update_water_density) nil)
    ((:water_density __water_density) 0.0)
    ((:update_center_of_gravity __update_center_of_gravity) nil)
    ((:center_of_gravity __center_of_gravity) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:update_center_of_buoyancy __update_center_of_buoyancy) nil)
    ((:center_of_buoyancy __center_of_buoyancy) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:update_moments __update_moments) nil)
    ((:moments __moments) (make-array 6 :initial-element 0.0 :element-type :float))
    ((:update_linear_damping __update_linear_damping) nil)
    ((:linear_damping __linear_damping) (make-array 6 :initial-element 0.0 :element-type :float))
    ((:update_quadratic_damping __update_quadratic_damping) nil)
    ((:quadratic_damping __quadratic_damping) (make-array 6 :initial-element 0.0 :element-type :float))
    ((:update_added_mass __update_added_mass) nil)
    ((:added_mass __added_mass) (make-array 6 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _update_mass __update_mass)
   (setq _mass (float __mass))
   (setq _update_volume __update_volume)
   (setq _volume (float __volume))
   (setq _update_water_density __update_water_density)
   (setq _water_density (float __water_density))
   (setq _update_center_of_gravity __update_center_of_gravity)
   (setq _center_of_gravity __center_of_gravity)
   (setq _update_center_of_buoyancy __update_center_of_buoyancy)
   (setq _center_of_buoyancy __center_of_buoyancy)
   (setq _update_moments __update_moments)
   (setq _moments __moments)
   (setq _update_linear_damping __update_linear_damping)
   (setq _linear_damping __linear_damping)
   (setq _update_quadratic_damping __update_quadratic_damping)
   (setq _quadratic_damping __quadratic_damping)
   (setq _update_added_mass __update_added_mass)
   (setq _added_mass __added_mass)
   self)
  (:update_mass
   (&optional (__update_mass :null))
   (if (not (eq __update_mass :null)) (setq _update_mass __update_mass)) _update_mass)
  (:mass
   (&optional __mass)
   (if __mass (setq _mass __mass)) _mass)
  (:update_volume
   (&optional (__update_volume :null))
   (if (not (eq __update_volume :null)) (setq _update_volume __update_volume)) _update_volume)
  (:volume
   (&optional __volume)
   (if __volume (setq _volume __volume)) _volume)
  (:update_water_density
   (&optional (__update_water_density :null))
   (if (not (eq __update_water_density :null)) (setq _update_water_density __update_water_density)) _update_water_density)
  (:water_density
   (&optional __water_density)
   (if __water_density (setq _water_density __water_density)) _water_density)
  (:update_center_of_gravity
   (&optional (__update_center_of_gravity :null))
   (if (not (eq __update_center_of_gravity :null)) (setq _update_center_of_gravity __update_center_of_gravity)) _update_center_of_gravity)
  (:center_of_gravity
   (&optional __center_of_gravity)
   (if __center_of_gravity (setq _center_of_gravity __center_of_gravity)) _center_of_gravity)
  (:update_center_of_buoyancy
   (&optional (__update_center_of_buoyancy :null))
   (if (not (eq __update_center_of_buoyancy :null)) (setq _update_center_of_buoyancy __update_center_of_buoyancy)) _update_center_of_buoyancy)
  (:center_of_buoyancy
   (&optional __center_of_buoyancy)
   (if __center_of_buoyancy (setq _center_of_buoyancy __center_of_buoyancy)) _center_of_buoyancy)
  (:update_moments
   (&optional (__update_moments :null))
   (if (not (eq __update_moments :null)) (setq _update_moments __update_moments)) _update_moments)
  (:moments
   (&optional __moments)
   (if __moments (setq _moments __moments)) _moments)
  (:update_linear_damping
   (&optional (__update_linear_damping :null))
   (if (not (eq __update_linear_damping :null)) (setq _update_linear_damping __update_linear_damping)) _update_linear_damping)
  (:linear_damping
   (&optional __linear_damping)
   (if __linear_damping (setq _linear_damping __linear_damping)) _linear_damping)
  (:update_quadratic_damping
   (&optional (__update_quadratic_damping :null))
   (if (not (eq __update_quadratic_damping :null)) (setq _update_quadratic_damping __update_quadratic_damping)) _update_quadratic_damping)
  (:quadratic_damping
   (&optional __quadratic_damping)
   (if __quadratic_damping (setq _quadratic_damping __quadratic_damping)) _quadratic_damping)
  (:update_added_mass
   (&optional (__update_added_mass :null))
   (if (not (eq __update_added_mass :null)) (setq _update_added_mass __update_added_mass)) _update_added_mass)
  (:added_mass
   (&optional __added_mass)
   (if __added_mass (setq _added_mass __added_mass)) _added_mass)
  (:serialization-length
   ()
   (+
    ;; bool _update_mass
    1
    ;; float64 _mass
    8
    ;; bool _update_volume
    1
    ;; float64 _volume
    8
    ;; bool _update_water_density
    1
    ;; float64 _water_density
    8
    ;; bool _update_center_of_gravity
    1
    ;; float64[3] _center_of_gravity
    (* 8    3)
    ;; bool _update_center_of_buoyancy
    1
    ;; float64[3] _center_of_buoyancy
    (* 8    3)
    ;; bool _update_moments
    1
    ;; float64[6] _moments
    (* 8    6)
    ;; bool _update_linear_damping
    1
    ;; float64[6] _linear_damping
    (* 8    6)
    ;; bool _update_quadratic_damping
    1
    ;; float64[6] _quadratic_damping
    (* 8    6)
    ;; bool _update_added_mass
    1
    ;; float64[6] _added_mass
    (* 8    6)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _update_mass
       (if _update_mass (write-byte -1 s) (write-byte 0 s))
     ;; float64 _mass
       (sys::poke _mass (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; bool _update_volume
       (if _update_volume (write-byte -1 s) (write-byte 0 s))
     ;; float64 _volume
       (sys::poke _volume (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; bool _update_water_density
       (if _update_water_density (write-byte -1 s) (write-byte 0 s))
     ;; float64 _water_density
       (sys::poke _water_density (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; bool _update_center_of_gravity
       (if _update_center_of_gravity (write-byte -1 s) (write-byte 0 s))
     ;; float64[3] _center_of_gravity
     (dotimes (i 3)
       (sys::poke (elt _center_of_gravity i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; bool _update_center_of_buoyancy
       (if _update_center_of_buoyancy (write-byte -1 s) (write-byte 0 s))
     ;; float64[3] _center_of_buoyancy
     (dotimes (i 3)
       (sys::poke (elt _center_of_buoyancy i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; bool _update_moments
       (if _update_moments (write-byte -1 s) (write-byte 0 s))
     ;; float64[6] _moments
     (dotimes (i 6)
       (sys::poke (elt _moments i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; bool _update_linear_damping
       (if _update_linear_damping (write-byte -1 s) (write-byte 0 s))
     ;; float64[6] _linear_damping
     (dotimes (i 6)
       (sys::poke (elt _linear_damping i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; bool _update_quadratic_damping
       (if _update_quadratic_damping (write-byte -1 s) (write-byte 0 s))
     ;; float64[6] _quadratic_damping
     (dotimes (i 6)
       (sys::poke (elt _quadratic_damping i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; bool _update_added_mass
       (if _update_added_mass (write-byte -1 s) (write-byte 0 s))
     ;; float64[6] _added_mass
     (dotimes (i 6)
       (sys::poke (elt _added_mass i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _update_mass
     (setq _update_mass (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float64 _mass
     (setq _mass (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; bool _update_volume
     (setq _update_volume (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float64 _volume
     (setq _volume (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; bool _update_water_density
     (setq _update_water_density (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float64 _water_density
     (setq _water_density (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; bool _update_center_of_gravity
     (setq _update_center_of_gravity (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float64[3] _center_of_gravity
   (dotimes (i (length _center_of_gravity))
     (setf (elt _center_of_gravity i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; bool _update_center_of_buoyancy
     (setq _update_center_of_buoyancy (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float64[3] _center_of_buoyancy
   (dotimes (i (length _center_of_buoyancy))
     (setf (elt _center_of_buoyancy i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; bool _update_moments
     (setq _update_moments (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float64[6] _moments
   (dotimes (i (length _moments))
     (setf (elt _moments i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; bool _update_linear_damping
     (setq _update_linear_damping (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float64[6] _linear_damping
   (dotimes (i (length _linear_damping))
     (setf (elt _linear_damping i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; bool _update_quadratic_damping
     (setq _update_quadratic_damping (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float64[6] _quadratic_damping
   (dotimes (i (length _quadratic_damping))
     (setf (elt _quadratic_damping i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; bool _update_added_mass
     (setq _update_added_mass (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float64[6] _added_mass
   (dotimes (i (length _added_mass))
     (setf (elt _added_mass i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;;
   self)
  )

(setf (get tauv_msgs::DynamicsTuning :md5sum-) "1657e74b53352c5e93a01a5d1743eeaa")
(setf (get tauv_msgs::DynamicsTuning :datatype-) "tauv_msgs/DynamicsTuning")
(setf (get tauv_msgs::DynamicsTuning :definition-)
      "bool update_mass
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
float64[6] added_mass
")



(provide :tauv_msgs/DynamicsTuning "1657e74b53352c5e93a01a5d1743eeaa")


