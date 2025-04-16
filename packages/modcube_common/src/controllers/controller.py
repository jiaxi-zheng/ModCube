import rospy
import numpy as np
from typing import Optional

from dynamics.dynamics import Dynamics
from geometry_msgs.msg import WrenchStamped, Vector3
from modcube_msgs.msg import ControllerCommand, NavigationState, ControllerDebug
from modcube_msgs.srv import TuneDynamics, TuneDynamicsRequest, TuneDynamicsResponse, TuneController, TuneControllerRequest, TuneControllerResponse
from modcube_util.types import tl, tm
from modcube_util.pid import PID, pi_clip
from modcube_util.transforms import euler_velocity_to_axis_velocity

from modcube_alarms import Alarm, AlarmClient

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class Controller:

    def __init__(self):
        self._ac = AlarmClient()

        self._load_config()

        self._dt: float = 1.0 / self._frequency
        self._navigation_state: Optional[NavigationState] = None
        self._controller_command: Optional[ControllerCommand] = None
        self._dyn: Dynamics = Dynamics(
            m=self._m,
            v=self._v,
            rho=self._rho,
            r_G=self._r_G,
            r_B=self._r_B,
            I=self._I,
            D=self._D,
            D2=self._D2,
            Ma=self._Ma,
        )

        self.pid_roll = PIDController(kp=0.5, ki=0, kd=0)
        self.pid_pitch = PIDController(kp=0.5, ki=0, kd=0)
        self.pid_yaw = PIDController(kp=0.5, ki=0, kd=0)

        self.desired_orientation = [0, 0, 0]

        self._build_pids()

        self._navigation_state_sub: rospy.Subscriber = rospy.Subscriber('gnc/navigation_state', NavigationState, self._handle_navigation_state)
        self._controller_command_sub: rospy.Subscriber = rospy.Subscriber('gnc/controller_command', ControllerCommand, self._handle_controller_command)
        self._controller_debug_pub = rospy.Publisher('gnc/controller_debug', ControllerDebug, queue_size=10)
        self._wrench_pub: rospy.Publisher = rospy.Publisher('gnc/target_wrench', WrenchStamped, queue_size=10)
        self._tune_dynamics_srv: rospy.Service = rospy.Service('gnc/tune_dynamics', TuneDynamics, self._handle_tune_dynamics)
        # self._tune_controller_srv: rospy.Service = rospy.Service('gnc/tune_controller', TuneController, self._handle_tune_controller)


    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        if self._navigation_state is None or self._controller_command is None:
            return

        state = self._navigation_state
        cmd = self._controller_command

        position = tl(state.position)
        orientation = tl(state.orientation)
        euler_velocity = tl(state.euler_velocity)
        linear_velocity = tl(state.linear_velocity)

        eta = np.concatenate((
            position,
            orientation
        ))

        axis_velocity = euler_velocity_to_axis_velocity(orientation, euler_velocity)
        v = np.concatenate((
            linear_velocity,
            axis_velocity
        ))

        vd = np.array([
            cmd.a_x,
            cmd.a_y,
            cmd.a_z,
            cmd.a_roll, 
            cmd.a_pitch, 
            cmd.a_yaw
        ])

        current_time = rospy.Time.now().to_sec()
        if not hasattr(self, '_last_time'):
            self._last_time = current_time

        dt = current_time - self._last_time
        self._last_time = current_time

        roll_error = self.desired_orientation[0] - orientation[0]
        pitch_error = self.desired_orientation[1] - orientation[1]
        yaw_error = self.desired_orientation[2] - orientation[2]

        # PID outputs
        additional_pid_loop_roll = self.pid_roll.compute(roll_error, dt)
        additional_pid_loop_pitch = self.pid_pitch.compute(pitch_error, dt)
        additional_pid_loop_yaw = self.pid_yaw.compute(yaw_error, dt)


        tau = self._dyn.compute_tau(eta, v, vd)

        tau = np.sign(tau) * np.minimum(np.abs(tau), self._max_wrench)

        wrench: WrenchStamped = WrenchStamped()
        wrench.header.frame_id = f'{self._tf_namespace}/vehicle'
        wrench.header.stamp = rospy.Time.now()
        wrench.wrench.force = Vector3(tau[0], tau[1], tau[2])
        # wrench.wrench.torque = Vector3(tau[3], tau[4], tau[5])
        wrench.wrench.torque = Vector3(additional_pid_loop_roll, additional_pid_loop_pitch, additional_pid_loop_yaw)
    
        self._wrench_pub.publish(wrench)

        # controller_debug: ControllerDebug = ControllerDebug()
        # controller_debug.z.tuning = self._pids[0].get_tuning()
        # controller_debug.z.value = position[2]
        # controller_debug.z.error = z_error
        # controller_debug.z.setpoint = cmd.setpoint_z
        # controller_debug.z.proportional = self._pids[0]._proportional
        # controller_debug.z.integral = self._pids[0]._integral
        # controller_debug.z.derivative = self._pids[0]._derivative
        # controller_debug.z.effort = z_effort
        # controller_debug.roll.tuning = self._pids[1].get_tuning()
        # controller_debug.roll.value = orientation[0]
        # controller_debug.roll.error = roll_error
        # controller_debug.roll.setpoint = cmd.setpoint_roll
        # controller_debug.roll.proportional = self._pids[1]._proportional
        # controller_debug.roll.integral = self._pids[1]._integral
        # controller_debug.roll.derivative = self._pids[1]._derivative
        # controller_debug.roll.effort = roll_effort
        # controller_debug.pitch.tuning = self._pids[2].get_tuning()
        # controller_debug.pitch.value = orientation[1]
        # controller_debug.pitch.setpoint = cmd.setpoint_pitch
        # controller_debug.pitch.error = pitch_error
        # controller_debug.pitch.proportional = self._pids[2]._proportional
        # controller_debug.pitch.integral = self._pids[2]._integral
        # controller_debug.pitch.derivative = self._pids[2]._derivative
        # controller_debug.pitch.effort = pitch_effort
        # self._controller_debug_pub.publish(controller_debug)

        self._ac.clear(Alarm.CONTROLLER_NOT_INITIALIZED)

    def _handle_navigation_state(self, msg: NavigationState):
        self._navigation_state = msg

    def _handle_controller_command(self, msg: ControllerCommand):
        self._controller_command = msg

    def _handle_tune_dynamics(self, req: TuneDynamicsRequest) -> TuneDynamicsResponse:
        if req.tuning.update_mass:
            self._m = req.tuning.mass

        if req.tuning.update_volume:
            self._v = req.tuning.volume

        if req.tuning.update_water_density:
            self._rho = req.tuning.water_density

        if req.tuning.update_center_of_gravity:
            self._r_G = req.tuning.center_of_gravity

        if req.tuning.update_center_of_buoyancy:
            self._r_B = req.tuning.center_of_buoyancy

        if req.tuning.update_moments:
            self._I = req.tuning.moments

        if req.tuning.update_linear_damping:
            self._D = req.tuning.linear_damping

        if req.tuning.update_quadratic_damping:
            self._D2 = req.tuning.quadratic_damping

        if req.tuning.update_added_mass:
            self._Ma = req.tuning.added_mass

        self._dyn: Dynamics = Dynamics(
            m = self._m,
            v = self._v,
            rho = self._rho,
            r_G = self._r_G,
            r_B = self._r_B,
            I = self._I,
            D = self._D,
            D2 = self._D2,
            Ma = self._Ma,
        )
        return TuneDynamicsResponse(True)

    def _build_pids(self):
        pids = []

        # z, roll, pitch
        for i in range(6):
            pid = PID(
                Kp=self._kp[i],
                Ki=self._ki[i],
                Kd=self._kd[i],
                error_map=pi_clip,
                output_limits=self._limits[i],
                proportional_on_measurement=False,
                sample_time=self._dt,
                d_alpha=self._dt / self._tau[i] if self._tau[i] > 0 else 1
            )
            pids.append(pid)

        self._pids = pids

    def _load_config(self):
        self._tf_namespace = rospy.get_param('tf_namespace')
        self._frequency = rospy.get_param('~frequency')
        self._kp = np.array(rospy.get_param('~kp'))
        self._ki = np.array(rospy.get_param('~ki'))
        self._kd = np.array(rospy.get_param('~kd'))
        self._tau = np.array(rospy.get_param('~tau'))
        self._limits = np.array(rospy.get_param('~limits'))

        self._max_wrench = np.array(rospy.get_param('~max_wrench'))

        self._m = rospy.get_param('~dynamics/mass')
        self._v = rospy.get_param('~dynamics/volume')
        self._rho = rospy.get_param('~dynamics/water_density')
        self._r_G = np.array(rospy.get_param('~dynamics/center_of_gravity'))
        self._r_B = np.array(rospy.get_param('~dynamics/center_of_buoyancy'))
        self._I = np.array(rospy.get_param('~dynamics/moments'))
        self._D = np.array(rospy.get_param('~dynamics/linear_damping'))
        self._D2 = np.array(rospy.get_param('~dynamics/quadratic_damping'))
        self._Ma = np.array(rospy.get_param('~dynamics/added_mass'))



def main():
    rospy.init_node('controller')
    c = Controller()
    c.start()
