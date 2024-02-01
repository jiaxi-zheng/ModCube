;; Auto-generated. Do not edit!


(when (boundp 'tauv_msgs::TeledyneDvlData)
  (if (not (find-package "TAUV_MSGS"))
    (make-package "TAUV_MSGS"))
  (shadow 'TeledyneDvlData (find-package "TAUV_MSGS")))
(unless (find-package "TAUV_MSGS::TELEDYNEDVLDATA")
  (make-package "TAUV_MSGS::TELEDYNEDVLDATA"))

(in-package "ROS")
;;//! \htmlinclude TeledyneDvlData.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass tauv_msgs::TeledyneDvlData
  :super ros::object
  :slots (_header _ensemble_number _test_status _health_status _shallow_mode _depth _pressure _pressure_variance _heading _pitch _roll _speed_of_sound _salinity _temperature _transmit_voltage _transmit_current _transmit_impedance _velocity _velocity_error _hr_velocity _hr_velocity_error _is_hr_velocity_valid _beam_ranges _beam_correlations _beam_amplitudes _beam_percent_goods _beam_time_to_bottoms _beam_standard_deviations _beam_time_of_validities _beam_rssis _slant_range _axis_delta_range _vertical_range ))

(defmethod tauv_msgs::TeledyneDvlData
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:ensemble_number __ensemble_number) 0)
    ((:test_status __test_status) (instance std_msgs::String :init))
    ((:health_status __health_status) (instance std_msgs::String :init))
    ((:shallow_mode __shallow_mode) nil)
    ((:depth __depth) 0.0)
    ((:pressure __pressure) 0.0)
    ((:pressure_variance __pressure_variance) 0.0)
    ((:heading __heading) 0.0)
    ((:pitch __pitch) 0.0)
    ((:roll __roll) 0.0)
    ((:speed_of_sound __speed_of_sound) 0.0)
    ((:salinity __salinity) 0.0)
    ((:temperature __temperature) 0.0)
    ((:transmit_voltage __transmit_voltage) 0.0)
    ((:transmit_current __transmit_current) 0.0)
    ((:transmit_impedance __transmit_impedance) 0.0)
    ((:velocity __velocity) (instance geometry_msgs::Vector3 :init))
    ((:velocity_error __velocity_error) 0.0)
    ((:hr_velocity __hr_velocity) (instance geometry_msgs::Vector3 :init))
    ((:hr_velocity_error __hr_velocity_error) 0.0)
    ((:is_hr_velocity_valid __is_hr_velocity_valid) nil)
    ((:beam_ranges __beam_ranges) (make-array 4 :initial-element 0.0 :element-type :float))
    ((:beam_correlations __beam_correlations) (make-array 4 :initial-element 0.0 :element-type :float))
    ((:beam_amplitudes __beam_amplitudes) (make-array 4 :initial-element 0.0 :element-type :float))
    ((:beam_percent_goods __beam_percent_goods) (make-array 4 :initial-element 0.0 :element-type :float))
    ((:beam_time_to_bottoms __beam_time_to_bottoms) (make-array 4 :initial-element 0.0 :element-type :float))
    ((:beam_standard_deviations __beam_standard_deviations) (make-array 4 :initial-element 0.0 :element-type :float))
    ((:beam_time_of_validities __beam_time_of_validities) (make-array 4 :initial-element 0.0 :element-type :float))
    ((:beam_rssis __beam_rssis) (make-array 4 :initial-element 0.0 :element-type :float))
    ((:slant_range __slant_range) 0.0)
    ((:axis_delta_range __axis_delta_range) 0.0)
    ((:vertical_range __vertical_range) 0.0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _ensemble_number (round __ensemble_number))
   (setq _test_status __test_status)
   (setq _health_status __health_status)
   (setq _shallow_mode __shallow_mode)
   (setq _depth (float __depth))
   (setq _pressure (float __pressure))
   (setq _pressure_variance (float __pressure_variance))
   (setq _heading (float __heading))
   (setq _pitch (float __pitch))
   (setq _roll (float __roll))
   (setq _speed_of_sound (float __speed_of_sound))
   (setq _salinity (float __salinity))
   (setq _temperature (float __temperature))
   (setq _transmit_voltage (float __transmit_voltage))
   (setq _transmit_current (float __transmit_current))
   (setq _transmit_impedance (float __transmit_impedance))
   (setq _velocity __velocity)
   (setq _velocity_error (float __velocity_error))
   (setq _hr_velocity __hr_velocity)
   (setq _hr_velocity_error (float __hr_velocity_error))
   (setq _is_hr_velocity_valid __is_hr_velocity_valid)
   (setq _beam_ranges __beam_ranges)
   (setq _beam_correlations __beam_correlations)
   (setq _beam_amplitudes __beam_amplitudes)
   (setq _beam_percent_goods __beam_percent_goods)
   (setq _beam_time_to_bottoms __beam_time_to_bottoms)
   (setq _beam_standard_deviations __beam_standard_deviations)
   (setq _beam_time_of_validities __beam_time_of_validities)
   (setq _beam_rssis __beam_rssis)
   (setq _slant_range (float __slant_range))
   (setq _axis_delta_range (float __axis_delta_range))
   (setq _vertical_range (float __vertical_range))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:ensemble_number
   (&optional __ensemble_number)
   (if __ensemble_number (setq _ensemble_number __ensemble_number)) _ensemble_number)
  (:test_status
   (&rest __test_status)
   (if (keywordp (car __test_status))
       (send* _test_status __test_status)
     (progn
       (if __test_status (setq _test_status (car __test_status)))
       _test_status)))
  (:health_status
   (&rest __health_status)
   (if (keywordp (car __health_status))
       (send* _health_status __health_status)
     (progn
       (if __health_status (setq _health_status (car __health_status)))
       _health_status)))
  (:shallow_mode
   (&optional (__shallow_mode :null))
   (if (not (eq __shallow_mode :null)) (setq _shallow_mode __shallow_mode)) _shallow_mode)
  (:depth
   (&optional __depth)
   (if __depth (setq _depth __depth)) _depth)
  (:pressure
   (&optional __pressure)
   (if __pressure (setq _pressure __pressure)) _pressure)
  (:pressure_variance
   (&optional __pressure_variance)
   (if __pressure_variance (setq _pressure_variance __pressure_variance)) _pressure_variance)
  (:heading
   (&optional __heading)
   (if __heading (setq _heading __heading)) _heading)
  (:pitch
   (&optional __pitch)
   (if __pitch (setq _pitch __pitch)) _pitch)
  (:roll
   (&optional __roll)
   (if __roll (setq _roll __roll)) _roll)
  (:speed_of_sound
   (&optional __speed_of_sound)
   (if __speed_of_sound (setq _speed_of_sound __speed_of_sound)) _speed_of_sound)
  (:salinity
   (&optional __salinity)
   (if __salinity (setq _salinity __salinity)) _salinity)
  (:temperature
   (&optional __temperature)
   (if __temperature (setq _temperature __temperature)) _temperature)
  (:transmit_voltage
   (&optional __transmit_voltage)
   (if __transmit_voltage (setq _transmit_voltage __transmit_voltage)) _transmit_voltage)
  (:transmit_current
   (&optional __transmit_current)
   (if __transmit_current (setq _transmit_current __transmit_current)) _transmit_current)
  (:transmit_impedance
   (&optional __transmit_impedance)
   (if __transmit_impedance (setq _transmit_impedance __transmit_impedance)) _transmit_impedance)
  (:velocity
   (&rest __velocity)
   (if (keywordp (car __velocity))
       (send* _velocity __velocity)
     (progn
       (if __velocity (setq _velocity (car __velocity)))
       _velocity)))
  (:velocity_error
   (&optional __velocity_error)
   (if __velocity_error (setq _velocity_error __velocity_error)) _velocity_error)
  (:hr_velocity
   (&rest __hr_velocity)
   (if (keywordp (car __hr_velocity))
       (send* _hr_velocity __hr_velocity)
     (progn
       (if __hr_velocity (setq _hr_velocity (car __hr_velocity)))
       _hr_velocity)))
  (:hr_velocity_error
   (&optional __hr_velocity_error)
   (if __hr_velocity_error (setq _hr_velocity_error __hr_velocity_error)) _hr_velocity_error)
  (:is_hr_velocity_valid
   (&optional (__is_hr_velocity_valid :null))
   (if (not (eq __is_hr_velocity_valid :null)) (setq _is_hr_velocity_valid __is_hr_velocity_valid)) _is_hr_velocity_valid)
  (:beam_ranges
   (&optional __beam_ranges)
   (if __beam_ranges (setq _beam_ranges __beam_ranges)) _beam_ranges)
  (:beam_correlations
   (&optional __beam_correlations)
   (if __beam_correlations (setq _beam_correlations __beam_correlations)) _beam_correlations)
  (:beam_amplitudes
   (&optional __beam_amplitudes)
   (if __beam_amplitudes (setq _beam_amplitudes __beam_amplitudes)) _beam_amplitudes)
  (:beam_percent_goods
   (&optional __beam_percent_goods)
   (if __beam_percent_goods (setq _beam_percent_goods __beam_percent_goods)) _beam_percent_goods)
  (:beam_time_to_bottoms
   (&optional __beam_time_to_bottoms)
   (if __beam_time_to_bottoms (setq _beam_time_to_bottoms __beam_time_to_bottoms)) _beam_time_to_bottoms)
  (:beam_standard_deviations
   (&optional __beam_standard_deviations)
   (if __beam_standard_deviations (setq _beam_standard_deviations __beam_standard_deviations)) _beam_standard_deviations)
  (:beam_time_of_validities
   (&optional __beam_time_of_validities)
   (if __beam_time_of_validities (setq _beam_time_of_validities __beam_time_of_validities)) _beam_time_of_validities)
  (:beam_rssis
   (&optional __beam_rssis)
   (if __beam_rssis (setq _beam_rssis __beam_rssis)) _beam_rssis)
  (:slant_range
   (&optional __slant_range)
   (if __slant_range (setq _slant_range __slant_range)) _slant_range)
  (:axis_delta_range
   (&optional __axis_delta_range)
   (if __axis_delta_range (setq _axis_delta_range __axis_delta_range)) _axis_delta_range)
  (:vertical_range
   (&optional __vertical_range)
   (if __vertical_range (setq _vertical_range __vertical_range)) _vertical_range)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; int64 _ensemble_number
    8
    ;; std_msgs/String _test_status
    (send _test_status :serialization-length)
    ;; std_msgs/String _health_status
    (send _health_status :serialization-length)
    ;; bool _shallow_mode
    1
    ;; float64 _depth
    8
    ;; float64 _pressure
    8
    ;; float64 _pressure_variance
    8
    ;; float64 _heading
    8
    ;; float64 _pitch
    8
    ;; float64 _roll
    8
    ;; float64 _speed_of_sound
    8
    ;; float64 _salinity
    8
    ;; float64 _temperature
    8
    ;; float64 _transmit_voltage
    8
    ;; float64 _transmit_current
    8
    ;; float64 _transmit_impedance
    8
    ;; geometry_msgs/Vector3 _velocity
    (send _velocity :serialization-length)
    ;; float64 _velocity_error
    8
    ;; geometry_msgs/Vector3 _hr_velocity
    (send _hr_velocity :serialization-length)
    ;; float64 _hr_velocity_error
    8
    ;; bool _is_hr_velocity_valid
    1
    ;; float64[4] _beam_ranges
    (* 8    4)
    ;; float64[4] _beam_correlations
    (* 8    4)
    ;; float64[4] _beam_amplitudes
    (* 8    4)
    ;; float64[4] _beam_percent_goods
    (* 8    4)
    ;; float64[4] _beam_time_to_bottoms
    (* 8    4)
    ;; float64[4] _beam_standard_deviations
    (* 8    4)
    ;; float64[4] _beam_time_of_validities
    (* 8    4)
    ;; float64[4] _beam_rssis
    (* 8    4)
    ;; float64 _slant_range
    8
    ;; float64 _axis_delta_range
    8
    ;; float64 _vertical_range
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; int64 _ensemble_number
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke _ensemble_number (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class _ensemble_number) (= (length (_ensemble_number . bv)) 2)) ;; bignum
              (write-long (ash (elt (_ensemble_number . bv) 0) 0) s)
              (write-long (ash (elt (_ensemble_number . bv) 1) -1) s))
             ((and (class _ensemble_number) (= (length (_ensemble_number . bv)) 1)) ;; big1
              (write-long (elt (_ensemble_number . bv) 0) s)
              (write-long (if (>= _ensemble_number 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long _ensemble_number s)(write-long (if (>= _ensemble_number 0) 0 #xffffffff) s)))
     ;; std_msgs/String _test_status
       (send _test_status :serialize s)
     ;; std_msgs/String _health_status
       (send _health_status :serialize s)
     ;; bool _shallow_mode
       (if _shallow_mode (write-byte -1 s) (write-byte 0 s))
     ;; float64 _depth
       (sys::poke _depth (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _pressure
       (sys::poke _pressure (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _pressure_variance
       (sys::poke _pressure_variance (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _heading
       (sys::poke _heading (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _pitch
       (sys::poke _pitch (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _roll
       (sys::poke _roll (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _speed_of_sound
       (sys::poke _speed_of_sound (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _salinity
       (sys::poke _salinity (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _temperature
       (sys::poke _temperature (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _transmit_voltage
       (sys::poke _transmit_voltage (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _transmit_current
       (sys::poke _transmit_current (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _transmit_impedance
       (sys::poke _transmit_impedance (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; geometry_msgs/Vector3 _velocity
       (send _velocity :serialize s)
     ;; float64 _velocity_error
       (sys::poke _velocity_error (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; geometry_msgs/Vector3 _hr_velocity
       (send _hr_velocity :serialize s)
     ;; float64 _hr_velocity_error
       (sys::poke _hr_velocity_error (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; bool _is_hr_velocity_valid
       (if _is_hr_velocity_valid (write-byte -1 s) (write-byte 0 s))
     ;; float64[4] _beam_ranges
     (dotimes (i 4)
       (sys::poke (elt _beam_ranges i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[4] _beam_correlations
     (dotimes (i 4)
       (sys::poke (elt _beam_correlations i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[4] _beam_amplitudes
     (dotimes (i 4)
       (sys::poke (elt _beam_amplitudes i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[4] _beam_percent_goods
     (dotimes (i 4)
       (sys::poke (elt _beam_percent_goods i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[4] _beam_time_to_bottoms
     (dotimes (i 4)
       (sys::poke (elt _beam_time_to_bottoms i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[4] _beam_standard_deviations
     (dotimes (i 4)
       (sys::poke (elt _beam_standard_deviations i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[4] _beam_time_of_validities
     (dotimes (i 4)
       (sys::poke (elt _beam_time_of_validities i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[4] _beam_rssis
     (dotimes (i 4)
       (sys::poke (elt _beam_rssis i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64 _slant_range
       (sys::poke _slant_range (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _axis_delta_range
       (sys::poke _axis_delta_range (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _vertical_range
       (sys::poke _vertical_range (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; int64 _ensemble_number
#+(or :alpha :irix6 :x86_64)
      (setf _ensemble_number (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf _ensemble_number (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
                  (b1 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4))))
              (cond ((= b1 -1) b0)
                     ((and (= b1  0)
                           (<= lisp::most-negative-fixnum b0 lisp::most-positive-fixnum))
                      b0)
                    ((= b1  0) (make-instance bignum :size 1 :bv (integer-vector b0)))
                    (t (make-instance bignum :size 2 :bv (integer-vector b0 (ash b1 1)))))))
   ;; std_msgs/String _test_status
     (send _test_status :deserialize buf ptr-) (incf ptr- (send _test_status :serialization-length))
   ;; std_msgs/String _health_status
     (send _health_status :deserialize buf ptr-) (incf ptr- (send _health_status :serialization-length))
   ;; bool _shallow_mode
     (setq _shallow_mode (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float64 _depth
     (setq _depth (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _pressure
     (setq _pressure (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _pressure_variance
     (setq _pressure_variance (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _heading
     (setq _heading (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _pitch
     (setq _pitch (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _roll
     (setq _roll (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _speed_of_sound
     (setq _speed_of_sound (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _salinity
     (setq _salinity (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _temperature
     (setq _temperature (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _transmit_voltage
     (setq _transmit_voltage (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _transmit_current
     (setq _transmit_current (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _transmit_impedance
     (setq _transmit_impedance (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; geometry_msgs/Vector3 _velocity
     (send _velocity :deserialize buf ptr-) (incf ptr- (send _velocity :serialization-length))
   ;; float64 _velocity_error
     (setq _velocity_error (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; geometry_msgs/Vector3 _hr_velocity
     (send _hr_velocity :deserialize buf ptr-) (incf ptr- (send _hr_velocity :serialization-length))
   ;; float64 _hr_velocity_error
     (setq _hr_velocity_error (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; bool _is_hr_velocity_valid
     (setq _is_hr_velocity_valid (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; float64[4] _beam_ranges
   (dotimes (i (length _beam_ranges))
     (setf (elt _beam_ranges i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[4] _beam_correlations
   (dotimes (i (length _beam_correlations))
     (setf (elt _beam_correlations i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[4] _beam_amplitudes
   (dotimes (i (length _beam_amplitudes))
     (setf (elt _beam_amplitudes i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[4] _beam_percent_goods
   (dotimes (i (length _beam_percent_goods))
     (setf (elt _beam_percent_goods i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[4] _beam_time_to_bottoms
   (dotimes (i (length _beam_time_to_bottoms))
     (setf (elt _beam_time_to_bottoms i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[4] _beam_standard_deviations
   (dotimes (i (length _beam_standard_deviations))
     (setf (elt _beam_standard_deviations i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[4] _beam_time_of_validities
   (dotimes (i (length _beam_time_of_validities))
     (setf (elt _beam_time_of_validities i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[4] _beam_rssis
   (dotimes (i (length _beam_rssis))
     (setf (elt _beam_rssis i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64 _slant_range
     (setq _slant_range (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _axis_delta_range
     (setq _axis_delta_range (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _vertical_range
     (setq _vertical_range (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(setf (get tauv_msgs::TeledyneDvlData :md5sum-) "65bd41b11739b2abbd5b158cfef08147")
(setf (get tauv_msgs::TeledyneDvlData :datatype-) "tauv_msgs/TeledyneDvlData")
(setf (get tauv_msgs::TeledyneDvlData :definition-)
      "Header header

int64 ensemble_number

std_msgs/String test_status
std_msgs/String health_status

bool shallow_mode
float64 depth
float64 pressure
float64 pressure_variance

float64 heading
float64 pitch
float64 roll
float64 speed_of_sound
float64 salinity
float64 temperature

float64 transmit_voltage
float64 transmit_current
float64 transmit_impedance

geometry_msgs/Vector3 velocity
float64 velocity_error

geometry_msgs/Vector3 hr_velocity
float64 hr_velocity_error
bool is_hr_velocity_valid

float64[4] beam_ranges
float64[4] beam_correlations
float64[4] beam_amplitudes
float64[4] beam_percent_goods
float64[4] beam_time_to_bottoms
float64[4] beam_standard_deviations
float64[4] beam_time_of_validities
float64[4] beam_rssis

float64 slant_range
float64 axis_delta_range
float64 vertical_range


================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: std_msgs/String
string data

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
float64 z
")



(provide :tauv_msgs/TeledyneDvlData "65bd41b11739b2abbd5b158cfef08147")


