;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::Message)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'Message (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::MESSAGE")
  (make-package "TAUV_MSGS::MESSAGE"))

(in-package "ROS")
;;//! \htmlinclude Message.msg.html


(intern "*ERROR*" (find-package "TAUV_MSGS::MESSAGE"))
(shadow '*ERROR* (find-package "TAUV_MSGS::MESSAGE"))
(defconstant tauv_msgs::Message::*ERROR* 0)
(intern "*WARNING*" (find-package "TAUV_MSGS::MESSAGE"))
(shadow '*WARNING* (find-package "TAUV_MSGS::MESSAGE"))
(defconstant tauv_msgs::Message::*WARNING* 1)
(intern "*INFO*" (find-package "TAUV_MSGS::MESSAGE"))
(shadow '*INFO* (find-package "TAUV_MSGS::MESSAGE"))
(defconstant tauv_msgs::Message::*INFO* 2)
(intern "*DEBUG*" (find-package "TAUV_MSGS::MESSAGE"))
(shadow '*DEBUG* (find-package "TAUV_MSGS::MESSAGE"))
(defconstant tauv_msgs::Message::*DEBUG* 3)

(defun tauv_msgs::Message-to-symbol (const)
  (cond
        ((= const 0) 'tauv_msgs::Message::*ERROR*)
        ((= const 1) 'tauv_msgs::Message::*WARNING*)
        ((= const 2) 'tauv_msgs::Message::*INFO*)
        ((= const 3) 'tauv_msgs::Message::*DEBUG*)
        (t nil)))

(defclass tauv_msgs::Message
  :super ros::object
  :slots (_stamp _color_code_256 _severity _message ))

(defmethod tauv_msgs::Message
  (:init
   (&key
    ((:stamp __stamp) (instance ros::time :init))
    ((:color_code_256 __color_code_256) 0)
    ((:severity __severity) 0)
    ((:message __message) "")
    )
   (send-super :init)
   (setq _stamp __stamp)
   (setq _color_code_256 (round __color_code_256))
   (setq _severity (round __severity))
   (setq _message (string __message))
   self)
  (:stamp
   (&optional __stamp)
   (if __stamp (setq _stamp __stamp)) _stamp)
  (:color_code_256
   (&optional __color_code_256)
   (if __color_code_256 (setq _color_code_256 __color_code_256)) _color_code_256)
  (:severity
   (&optional __severity)
   (if __severity (setq _severity __severity)) _severity)
  (:message
   (&optional __message)
   (if __message (setq _message __message)) _message)
  (:serialization-length
   ()
   (+
    ;; time _stamp
    8
    ;; uint8 _color_code_256
    1
    ;; int8 _severity
    1
    ;; string _message
    4 (length _message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; time _stamp
       (write-long (send _stamp :sec) s) (write-long (send _stamp :nsec) s)
     ;; uint8 _color_code_256
       (write-byte _color_code_256 s)
     ;; int8 _severity
       (write-byte _severity s)
     ;; string _message
       (write-long (length _message) s) (princ _message s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; time _stamp
     (send _stamp :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _stamp :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; uint8 _color_code_256
     (setq _color_code_256 (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; int8 _severity
     (setq _severity (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _severity 127) (setq _severity (- _severity 256)))
   ;; string _message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(setf (get tauv_msgs::Message :md5sum-) "9553ea1e0a9cfe897226edaceb34218e")
(setf (get tauv_msgs::Message :datatype-) "tauv_msgs/Message")
(setf (get tauv_msgs::Message :definition-)
      "int8 ERROR=0
int8 WARNING=1
int8 INFO=2
int8 DEBUG=3

time stamp         # time stamp of the message
uint8 color_code_256   # color code of the message
int8 severity       # severity level
string message      # message

")



(provide :tauv_msgs/Message "9553ea1e0a9cfe897226edaceb34218e")


