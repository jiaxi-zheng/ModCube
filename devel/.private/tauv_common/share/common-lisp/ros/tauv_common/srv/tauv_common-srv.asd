
(cl:in-package :asdf)

(defsystem "tauv_common-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
)
  :components ((:file "_package")
    (:file "GetThrusterCurve" :depends-on ("_package_GetThrusterCurve"))
    (:file "_package_GetThrusterCurve" :depends-on ("_package"))
    (:file "GetThrusterManagerConfig" :depends-on ("_package_GetThrusterManagerConfig"))
    (:file "_package_GetThrusterManagerConfig" :depends-on ("_package"))
    (:file "RetearSubPosition" :depends-on ("_package_RetearSubPosition"))
    (:file "_package_RetearSubPosition" :depends-on ("_package"))
    (:file "SetThrusterManagerConfig" :depends-on ("_package_SetThrusterManagerConfig"))
    (:file "_package_SetThrusterManagerConfig" :depends-on ("_package"))
    (:file "ThrusterManagerInfo" :depends-on ("_package_ThrusterManagerInfo"))
    (:file "_package_ThrusterManagerInfo" :depends-on ("_package"))
  ))