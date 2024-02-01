;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::Servos)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'Servos (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::SERVOS")
  (make-package "TAUV_MSGS::SERVOS"))

(in-package "ROS")
;;//! \htmlinclude Servos.msg.html


(defclass tauv_msgs::Servos
  :super ros::object
  :slots (_targets ))

(defmethod tauv_msgs::Servos
  (:init
   (&key
    ((:targets __targets) (make-array 4 :initial-element 0 :element-type :integer))
    )
   (send-super :init)
   (setq _targets __targets)
   self)
  (:targets
   (&optional __targets)
   (if __targets (setq _targets __targets)) _targets)
  (:serialization-length
   ()
   (+
    ;; int64[4] _targets
    (* 8    4)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int64[4] _targets
     (dotimes (i 4)
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke (elt _targets i) (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class (elt _targets i)) (= (length ((elt _targets i) . bv)) 2)) ;; bignum
              (write-long (ash (elt ((elt _targets i) . bv) 0) 0) s)
              (write-long (ash (elt ((elt _targets i) . bv) 1) -1) s))
             ((and (class (elt _targets i)) (= (length ((elt _targets i) . bv)) 1)) ;; big1
              (write-long (elt ((elt _targets i) . bv) 0) s)
              (write-long (if (>= (elt _targets i) 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long (elt _targets i) s)(write-long (if (>= (elt _targets i) 0) 0 #xffffffff) s)))
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int64[4] _targets
   (dotimes (i (length _targets))
#+(or :alpha :irix6 :x86_64)
      (setf (elt _targets i) (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf (elt _targets i) (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
                  (b1 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4))))
              (cond ((= b1 -1) b0)
                     ((and (= b1  0)
                           (<= lisp::most-negative-fixnum b0 lisp::most-positive-fixnum))
                      b0)
                    ((= b1  0) (make-instance bignum :size 1 :bv (integer-vector b0)))
                    (t (make-instance bignum :size 2 :bv (integer-vector b0 (ash b1 1)))))))
     )
   ;;
   self)
  )

(setf (get tauv_msgs::Servos :md5sum-) "e8f52576b380ad554e7b8eee4133713d")
(setf (get tauv_msgs::Servos :datatype-) "tauv_msgs/Servos")
(setf (get tauv_msgs::Servos :definition-)
      "int64[4] targets
")



(provide :tauv_msgs/Servos "e8f52576b380ad554e7b8eee4133713d")


