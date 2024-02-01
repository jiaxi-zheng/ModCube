;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::DynamicsParametersEstimate)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'DynamicsParametersEstimate (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::DYNAMICSPARAMETERSESTIMATE")
  (make-package "TAUV_MSGS::DYNAMICSPARAMETERSESTIMATE"))

(in-package "ROS")
;;//! \htmlinclude DynamicsParametersEstimate.msg.html


(defclass tauv_msgs::DynamicsParametersEstimate
  :super ros::object
  :slots (_stamp _m _cov_m _v _cov_v _g _cov_g _b _cov_b _I _cov_I _dl _cov_dl _dq _cov_dq _am _cov_am ))

(defmethod tauv_msgs::DynamicsParametersEstimate
  (:init
   (&key
    ((:stamp __stamp) (instance ros::time :init))
    ((:m __m) 0.0)
    ((:cov_m __cov_m) 0.0)
    ((:v __v) 0.0)
    ((:cov_v __cov_v) 0.0)
    ((:g __g) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:cov_g __cov_g) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:b __b) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:cov_b __cov_b) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:I __I) (make-array 6 :initial-element 0.0 :element-type :float))
    ((:cov_I __cov_I) (make-array 6 :initial-element 0.0 :element-type :float))
    ((:dl __dl) (make-array 6 :initial-element 0.0 :element-type :float))
    ((:cov_dl __cov_dl) (make-array 6 :initial-element 0.0 :element-type :float))
    ((:dq __dq) (make-array 6 :initial-element 0.0 :element-type :float))
    ((:cov_dq __cov_dq) (make-array 6 :initial-element 0.0 :element-type :float))
    ((:am __am) (make-array 6 :initial-element 0.0 :element-type :float))
    ((:cov_am __cov_am) (make-array 6 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _stamp __stamp)
   (setq _m (float __m))
   (setq _cov_m (float __cov_m))
   (setq _v (float __v))
   (setq _cov_v (float __cov_v))
   (setq _g __g)
   (setq _cov_g __cov_g)
   (setq _b __b)
   (setq _cov_b __cov_b)
   (setq _I __I)
   (setq _cov_I __cov_I)
   (setq _dl __dl)
   (setq _cov_dl __cov_dl)
   (setq _dq __dq)
   (setq _cov_dq __cov_dq)
   (setq _am __am)
   (setq _cov_am __cov_am)
   self)
  (:stamp
   (&optional __stamp)
   (if __stamp (setq _stamp __stamp)) _stamp)
  (:m
   (&optional __m)
   (if __m (setq _m __m)) _m)
  (:cov_m
   (&optional __cov_m)
   (if __cov_m (setq _cov_m __cov_m)) _cov_m)
  (:v
   (&optional __v)
   (if __v (setq _v __v)) _v)
  (:cov_v
   (&optional __cov_v)
   (if __cov_v (setq _cov_v __cov_v)) _cov_v)
  (:g
   (&optional __g)
   (if __g (setq _g __g)) _g)
  (:cov_g
   (&optional __cov_g)
   (if __cov_g (setq _cov_g __cov_g)) _cov_g)
  (:b
   (&optional __b)
   (if __b (setq _b __b)) _b)
  (:cov_b
   (&optional __cov_b)
   (if __cov_b (setq _cov_b __cov_b)) _cov_b)
  (:I
   (&optional __I)
   (if __I (setq _I __I)) _I)
  (:cov_I
   (&optional __cov_I)
   (if __cov_I (setq _cov_I __cov_I)) _cov_I)
  (:dl
   (&optional __dl)
   (if __dl (setq _dl __dl)) _dl)
  (:cov_dl
   (&optional __cov_dl)
   (if __cov_dl (setq _cov_dl __cov_dl)) _cov_dl)
  (:dq
   (&optional __dq)
   (if __dq (setq _dq __dq)) _dq)
  (:cov_dq
   (&optional __cov_dq)
   (if __cov_dq (setq _cov_dq __cov_dq)) _cov_dq)
  (:am
   (&optional __am)
   (if __am (setq _am __am)) _am)
  (:cov_am
   (&optional __cov_am)
   (if __cov_am (setq _cov_am __cov_am)) _cov_am)
  (:serialization-length
   ()
   (+
    ;; time _stamp
    8
    ;; float64 _m
    8
    ;; float64 _cov_m
    8
    ;; float64 _v
    8
    ;; float64 _cov_v
    8
    ;; float64[3] _g
    (* 8    3)
    ;; float64[3] _cov_g
    (* 8    3)
    ;; float64[3] _b
    (* 8    3)
    ;; float64[3] _cov_b
    (* 8    3)
    ;; float64[6] _I
    (* 8    6)
    ;; float64[6] _cov_I
    (* 8    6)
    ;; float64[6] _dl
    (* 8    6)
    ;; float64[6] _cov_dl
    (* 8    6)
    ;; float64[6] _dq
    (* 8    6)
    ;; float64[6] _cov_dq
    (* 8    6)
    ;; float64[6] _am
    (* 8    6)
    ;; float64[6] _cov_am
    (* 8    6)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; time _stamp
       (write-long (send _stamp :sec) s) (write-long (send _stamp :nsec) s)
     ;; float64 _m
       (sys::poke _m (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _cov_m
       (sys::poke _cov_m (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _v
       (sys::poke _v (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _cov_v
       (sys::poke _cov_v (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64[3] _g
     (dotimes (i 3)
       (sys::poke (elt _g i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[3] _cov_g
     (dotimes (i 3)
       (sys::poke (elt _cov_g i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[3] _b
     (dotimes (i 3)
       (sys::poke (elt _b i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[3] _cov_b
     (dotimes (i 3)
       (sys::poke (elt _cov_b i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[6] _I
     (dotimes (i 6)
       (sys::poke (elt _I i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[6] _cov_I
     (dotimes (i 6)
       (sys::poke (elt _cov_I i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[6] _dl
     (dotimes (i 6)
       (sys::poke (elt _dl i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[6] _cov_dl
     (dotimes (i 6)
       (sys::poke (elt _cov_dl i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[6] _dq
     (dotimes (i 6)
       (sys::poke (elt _dq i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[6] _cov_dq
     (dotimes (i 6)
       (sys::poke (elt _cov_dq i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[6] _am
     (dotimes (i 6)
       (sys::poke (elt _am i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[6] _cov_am
     (dotimes (i 6)
       (sys::poke (elt _cov_am i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; time _stamp
     (send _stamp :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _stamp :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; float64 _m
     (setq _m (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _cov_m
     (setq _cov_m (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _v
     (setq _v (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _cov_v
     (setq _cov_v (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64[3] _g
   (dotimes (i (length _g))
     (setf (elt _g i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[3] _cov_g
   (dotimes (i (length _cov_g))
     (setf (elt _cov_g i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[3] _b
   (dotimes (i (length _b))
     (setf (elt _b i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[3] _cov_b
   (dotimes (i (length _cov_b))
     (setf (elt _cov_b i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[6] _I
   (dotimes (i (length _I))
     (setf (elt _I i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[6] _cov_I
   (dotimes (i (length _cov_I))
     (setf (elt _cov_I i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[6] _dl
   (dotimes (i (length _dl))
     (setf (elt _dl i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[6] _cov_dl
   (dotimes (i (length _cov_dl))
     (setf (elt _cov_dl i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[6] _dq
   (dotimes (i (length _dq))
     (setf (elt _dq i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[6] _cov_dq
   (dotimes (i (length _cov_dq))
     (setf (elt _cov_dq i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[6] _am
   (dotimes (i (length _am))
     (setf (elt _am i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[6] _cov_am
   (dotimes (i (length _cov_am))
     (setf (elt _cov_am i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;;
   self)
  )

(setf (get tauv_msgs::DynamicsParametersEstimate :md5sum-) "bd252a6b82405355d619dce17f3eaa07")
(setf (get tauv_msgs::DynamicsParametersEstimate :datatype-) "tauv_msgs/DynamicsParametersEstimate")
(setf (get tauv_msgs::DynamicsParametersEstimate :definition-)
      "time stamp

float64 m
float64 cov_m
float64 v
float64 cov_v
float64[3] g
float64[3] cov_g
float64[3] b
float64[3] cov_b
float64[6] I
float64[6] cov_I
float64[6] dl
float64[6] cov_dl
float64[6] dq
float64[6] cov_dq
float64[6] am
float64[6] cov_am
")



(provide :tauv_msgs/DynamicsParametersEstimate "bd252a6b82405355d619dce17f3eaa07")


