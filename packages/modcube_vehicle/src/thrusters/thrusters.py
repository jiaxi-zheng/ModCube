import rospy
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from math import floor

from .maestro import Maestro
from modcube_util.types import tl
from modcube_msgs.msg import Battery as BatteryMsg, Servos as ServosMsg
from modcube_msgs.msg import Thrust
from std_msgs.msg import Float64, Int16
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from geometry_msgs.msg import Wrench, WrenchStamped
from std_msgs.msg import Bool

from modcube_alarms import Alarm, AlarmClient

class Thrusters:
    def __init__(self):
        self._ac: AlarmClient = AlarmClient()

        self._dt: float = 0.02
        self._timeout: float = 1.0

        self._load_config()

        self._is_armed: bool = True
        self._arm_service: rospy.Service = rospy.Service('vehicle/thrusters/arm', SetBool, self._handle_arm)

        self._target_thrust_subs: [rospy.Subscriber] = []
        self._target_pwm_pubs: [rospy.Publisher] = []
        self.pwm_msg = Thrust()

        self._target_pwm_pub: rospy.Publisher = rospy.Publisher('vehicle/target_pwm', Thrust, queue_size=10)

        # Initialize _target_thrusts with zeros, matching the number of thruster channels
        self._target_thrusts = [0.0] * len(self._thruster_channels)

        for thruster_id in self._thruster_ids:
            target_pwm_pub = rospy.Publisher(
                f'vehicle/thrusters/{thruster_id}/target_pwm',
                Float64,
                queue_size=10
            )
            self._target_pwm_pubs.append(target_pwm_pub)

        for thruster_id in range(len(self._thruster_channels)):
            target_thrust_sub = rospy.Subscriber(
                f'vehicle/thrusters/{thruster_id}/target_thrust',
                Float64,
                self._handle_target_thrust,
                callback_args=thruster_id
            )
            self._target_thrust_subs.append(target_thrust_sub)

        self._target_position_subs = []
        self._target_positions = [0.0] * len(self._servo_channels)
        for servo_id in range(len(self._servo_channels)):
            target_position_sub = rospy.Subscriber(
                f'vehicle/servos/{servo_id}/target_position',
                Float64,
                self._handle_target_position,
                callback_args=servo_id
            )
            self._target_position_subs.append(target_position_sub)

        self._battery_sub: rospy.Subscriber = rospy.Subscriber('vehicle/battery', BatteryMsg, self._handle_battery)
        self._active_pub: rospy.Publisher = rospy.Publisher('vehicle/thrusters/active', Bool, queue_size=10)

        self._battery_voltage: float = self._default_battery_voltage
        self._thrust_update_time: rospy.Time = rospy.Time.now()

        # Initialize PWM storage for thrusters
        self._thruster_pwm = [0] * len(self._thruster_channels)

    # Rest of the methods remain unchanged...

    def start(self):
        rospy.loginfo('start')
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        self._ac.set(Alarm.SUB_DISARMED, value=not self._is_armed)
        self._active_pub.publish(self._is_armed)

        if (rospy.Time.now() - self._thrust_update_time).to_sec() > self._timeout \
                or not self._is_armed:
            self._target_thrusts = [0] * len(self._thruster_channels)
            self._thrust_update_time = rospy.Time.now()

        for (thruster, thrust) in enumerate(self._target_thrusts):
            self._set_thrust(thruster, thrust)

        for (servo, position) in enumerate(self._target_positions):
            self._set_position(servo, position)

        self._ac.clear(Alarm.THRUSTERS_NOT_INITIALIZED)

    def _handle_arm(self, req: SetBoolRequest):
        rospy.loginfo('armed' if req.data else 'disarmed')
        self._is_armed = req.data
        return SetBoolResponse(True, '')

    def _handle_battery(self, msg: BatteryMsg):
        self._battery_voltage = msg.voltage

    def _handle_target_thrust(self, msg: Float64, thruster_id: int):
        if self._is_armed:
            self._target_thrusts[thruster_id] = msg.data
            self._thrust_update_time = rospy.Time.now()

    def _handle_target_position(self, msg: Float64, servo_id: int):
        if self._is_armed:
            self._target_positions[servo_id] = msg.data

    def _set_thrust(self, thruster: int, thrust: float):
        pwm = self._get_pwm_speed(thruster, thrust)
        # pwm_s = int((pwm-1500)/4) + 125
        self.pwm_msg.thruster_pwm[thruster] = pwm
        # pwm_msgs = Float64()
        # pwm_msgs.data = pwm
        # self._target_pwm_pubs[thruster].publish(pwm_msgs)
        self._target_pwm_pub.publish(self.pwm_msg)

    def _set_position(self, servo: int, position: float):
        if position > 0:
            pwm_speed = self._servo_zero_pwms[servo] + position * (self._servo_max_pwms[servo] - self._servo_zero_pwms[servo])
        else:
            pwm_speed = self._servo_zero_pwms[servo] - position * (self._servo_min_pwms[servo] - self._servo_zero_pwms[servo])
        # self._maestro.setTarget(int(pwm_speed * 4), self._servo_channels[servo])

    def _get_pwm_speed(self, thruster: int, thrust: float) -> int:
        pwm_speed = 1500

        thrust = thrust * self._thrust_inversions[thruster]

        if thrust < 0 and -self._negative_max_thrust < thrust < -self._negative_min_thrust:
            thrust_curve = Polynomial(
                (self._negative_thrust_coefficients[0]
                    + self._negative_thrust_coefficients[1] * self._battery_voltage
                    + self._negative_thrust_coefficients[2] * self._battery_voltage ** 2
                    - thrust,
                 self._negative_thrust_coefficients[3],
                 self._negative_thrust_coefficients[4]),
            )

            target_pwm_speed = floor(thrust_curve.roots()[0])

            if self._minimum_pwm_speed < target_pwm_speed < self._maximum_pwm_speed:
                pwm_speed = target_pwm_speed

        elif thrust > 0 and self._positive_min_thrust < thrust < self._positive_max_thrust:
            thrust_curve = Polynomial(
                (self._positive_thrust_coefficients[0]
                    + self._positive_thrust_coefficients[1] * self._battery_voltage
                    + self._positive_thrust_coefficients[2] * self._battery_voltage ** 2
                    - thrust,
                 self._positive_thrust_coefficients[3],
                 self._positive_thrust_coefficients[4]),
            )

            target_pwm_speed = floor(thrust_curve.roots()[1])

            if self._minimum_pwm_speed < target_pwm_speed < self._maximum_pwm_speed:
                pwm_speed = target_pwm_speed

        return pwm_speed

    def _load_config(self):
        # self._maestro_port: str = rospy.get_param('~maestro_port')
        self._can_port: str = rospy.get_param('~can_port')
        self._thruster_channels: [int] = rospy.get_param('~thruster_channels')
        self._servo_channels: [int] = rospy.get_param('~servo_channels')
        self._servo_min_pwms: [int] = rospy.get_param('~servo_min_pwms')
        self._servo_max_pwms: [int] = rospy.get_param('~servo_max_pwms')
        self._servo_zero_pwms: [int] = rospy.get_param('~servo_zero_pwms')
        self._default_battery_voltage: float = rospy.get_param('~default_battery_voltage')
        self._minimum_pwm_speed: float = rospy.get_param('~minimum_pwm_speed')
        self._maximum_pwm_speed: float = rospy.get_param('~maximum_pwm_speed')
        self._negative_min_thrust: float = rospy.get_param('~negative_min_thrust')
        self._negative_max_thrust: float = rospy.get_param('~negative_max_thrust')
        self._positive_min_thrust: float = rospy.get_param('~positive_min_thrust')
        self._positive_max_thrust: float = rospy.get_param('~positive_max_thrust')
        self._positive_thrust_coefficients: np.array = np.array(rospy.get_param('~positive_thrust_coefficients'))
        self._negative_thrust_coefficients: np.array = np.array(rospy.get_param('~negative_thrust_coefficients'))
        self._thrust_inversions: [float] = rospy.get_param('~thrust_inversions')
        self._thruster_ids = np.array(rospy.get_param('~thruster_ids'))

def clamp(x, x_min, x_max):
    return min(max(x, x_min), x_max)

def main():
    rospy.init_node('thrusters')
    t = Thrusters()
    t.start()