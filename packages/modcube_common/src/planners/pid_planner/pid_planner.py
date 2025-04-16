#!/usr/bin/env python3

import rospy
import numpy as np
from typing import Optional

from modcube_alarms import AlarmClient
from modcube_msgs.msg import NavigationState, ControllerCommand, PIDPlannerDebug
from modcube_msgs.srv import GetTrajectory, GetTrajectoryRequest, TunePIDPlanner, TunePIDPlannerRequest, TunePIDPlannerResponse
from modcube_util.transforms import build_pose, build_pose_with_quat, build_twist, twist_body_to_world, quat_to_rpy
from modcube_util.types import tl
from modcube_util.pid import PID, pi_clip
from geometry_msgs.msg import PoseStamped, Pose, Twist
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation

class PIDPlanner:

    def __init__(self):
        self._use_roll_pitch = True

        self._ac = AlarmClient()
        self._load_config()

        self._get_traj_service: rospy.ServiceProxy = rospy.ServiceProxy('gnc/get_trajectory', GetTrajectory)
        self._navigation_state_sub: rospy.Subscriber = rospy.Subscriber('gnc/navigation_state', NavigationState, self._handle_navigation_state)
        self._controller_command_pub: rospy.Publisher = rospy.Publisher('gnc/controller_command', ControllerCommand, queue_size=10)
        self._pid_planner_debug_pub: rospy.Publisher = rospy.Publisher('gnc/pid_planner_debug', PIDPlannerDebug, queue_size=10)

        # Add a publisher for the path visualization
        self._path_pub = rospy.Publisher('trajectory_path', Path, queue_size=10)

        # Keep a buffer of the last N target poses for visualization
        self._path_length = 200
        self._target_poses_buffer = []

        self._tune_pid_planner_service: rospy.Service = rospy.Service('gnc/tune_pid_planner', TunePIDPlanner, self._handle_tune_pid_planner)

        self._dt: float = 1.0 / self._frequency
        self._navigation_state: Optional[NavigationState] = None

        self._build_pids()

        self._body_position_effort = np.zeros(3)
        self._world_angular_effort = np.zeros(3)

        self._y_reached_limit = False
        self._z_reached_limit = False
        
    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        if self._navigation_state is None:
            return

        target_pose, target_world_twist = self._get_target()
        if target_pose is None or target_world_twist is None:
            return
        target_position = tl(target_pose.position)
        target_orientation = quat_to_rpy(target_pose.orientation)

        target_world_velocity = tl(target_world_twist.linear)
        target_world_angular_velocity = tl(target_world_twist.angular)

        current_position = tl(self._navigation_state.position)
        current_orientation = tl(self._navigation_state.orientation)

        pose = build_pose(tl(self._navigation_state.position), tl(self._navigation_state.orientation))
        body_twist = build_twist(tl(self._navigation_state.linear_velocity), tl(self._navigation_state.euler_velocity))
        world_twist = twist_body_to_world(pose, body_twist)

        position_acceleration = target_world_velocity - tl(world_twist.linear)
        angular_acceleration = target_world_angular_velocity - tl(world_twist.angular)

        position_error = current_position - target_position
        orientation_error = current_orientation - target_orientation

        world_position_effort = np.zeros(3)
        for i in range(3):
            world_position_effort[i] = self._pids[i](position_error[i]) + position_acceleration[i]

        world_angular_effort = np.zeros(3)
        for i in range(3):
            world_angular_effort[i] = self._pids[i+3](orientation_error[i])

        R = Rotation.from_euler('ZYX', np.flip(current_orientation)).inv()

        body_position_effort = R.apply(world_position_effort)

        # Low-pass filter
        # body_position_effort = self._tau[0:3] * self._body_position_effort + (1 - self._tau[0:3]) * body_position_effort
        # world_angular_effort = self._tau[3:6] * self._world_angular_effort + (1 - self._tau[3:6]) * world_angular_effort

        self._body_position_effort = body_position_effort
        self._world_angular_effort = world_angular_effort

        controller_command = ControllerCommand()
        controller_command.a_x = body_position_effort[0]
        controller_command.a_y = body_position_effort[1]
        controller_command.a_z = body_position_effort[2]
        controller_command.a_yaw = world_angular_effort[2]
        controller_command.a_roll = world_angular_effort[0]
        controller_command.a_pitch = world_angular_effort[1]
        self._controller_command_pub.publish(controller_command)


        pid_planner_debug = PIDPlannerDebug()

        # Debug info for x-axis
        pid_planner_debug.x.tuning = self._pids[0].get_tuning()
        pid_planner_debug.x.value = current_position[0]
        pid_planner_debug.x.setpoint = target_position[0]
        pid_planner_debug.x.error = position_error[0]
        pid_planner_debug.x.proportional = self._pids[0]._proportional
        pid_planner_debug.x.integral = self._pids[0]._integral
        pid_planner_debug.x.derivative = self._pids[0]._derivative
        pid_planner_debug.x.effort = world_position_effort[0]

        # Debug info for y-axis
        pid_planner_debug.y.tuning = self._pids[1].get_tuning()
        pid_planner_debug.y.value = current_position[1]
        pid_planner_debug.y.setpoint = target_position[1]
        pid_planner_debug.y.error = position_error[1]
        pid_planner_debug.y.proportional = self._pids[1]._proportional
        pid_planner_debug.y.integral = self._pids[1]._integral
        pid_planner_debug.y.derivative = self._pids[1]._derivative
        pid_planner_debug.y.effort = world_position_effort[1]

        # Debug info for z-axis
        pid_planner_debug.z.tuning = self._pids[2].get_tuning()
        pid_planner_debug.z.value = current_position[2]
        pid_planner_debug.z.setpoint = target_position[2]
        pid_planner_debug.z.error = position_error[2]
        pid_planner_debug.z.proportional = self._pids[2]._proportional
        pid_planner_debug.z.integral = self._pids[2]._integral
        pid_planner_debug.z.derivative = self._pids[2]._derivative
        pid_planner_debug.z.effort = world_position_effort[2]

        # Debug info roll
        pid_planner_debug.roll.value = current_orientation[0]
        pid_planner_debug.roll.setpoint = target_orientation[0]
        pid_planner_debug.roll.error = orientation_error[0]
        pid_planner_debug.roll.tuning = self._pids[3].get_tuning()
        pid_planner_debug.roll.proportional = self._pids[3]._proportional
        pid_planner_debug.roll.integral = self._pids[3]._integral
        pid_planner_debug.roll.derivative = self._pids[3]._derivative
        pid_planner_debug.roll.effort = world_angular_effort[0]

        # Debug info pitch
        pid_planner_debug.pitch.value = current_orientation[1]
        pid_planner_debug.pitch.setpoint = target_orientation[1]
        pid_planner_debug.pitch.error = orientation_error[1]
        pid_planner_debug.pitch.tuning = self._pids[4].get_tuning()
        pid_planner_debug.pitch.proportional = self._pids[4]._proportional
        pid_planner_debug.pitch.integral = self._pids[4]._integral
        pid_planner_debug.pitch.derivative = self._pids[4]._derivative
        pid_planner_debug.pitch.effort = world_angular_effort[1]

        # Debug info yaw
        pid_planner_debug.yaw.value = current_orientation[2]
        pid_planner_debug.yaw.setpoint = target_orientation[2]
        pid_planner_debug.yaw.error = orientation_error[2]
        pid_planner_debug.yaw.tuning = self._pids[5].get_tuning()
        pid_planner_debug.yaw.proportional = self._pids[5]._proportional
        pid_planner_debug.yaw.integral = self._pids[5]._integral
        pid_planner_debug.yaw.derivative = self._pids[5]._derivative
        pid_planner_debug.yaw.effort = world_angular_effort[2]

        self._pid_planner_debug_pub.publish(pid_planner_debug)

        # Add the current target pose to the buffer for visualization
        self._add_pose_to_buffer(target_pose)
        self._publish_path()

    def _add_pose_to_buffer(self, pose: Pose):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = self._tf_namespace  # Frame for visualization
        pose_stamped.pose = pose
        self._target_poses_buffer.append(pose_stamped)

    def _publish_path(self):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "kf/odom"  # Frame for visualization
        path_msg.poses = self._target_poses_buffer
        self._path_pub.publish(path_msg)

    def _get_target(self) -> (Optional[Pose], Optional[Twist]):
        pose = build_pose(tl(self._navigation_state.position), tl(self._navigation_state.orientation))
        body_twist = build_twist(tl(self._navigation_state.linear_velocity), tl(self._navigation_state.euler_velocity))

        req = GetTrajectoryRequest()
        req.curr_pose = pose
        req.curr_twist = twist_body_to_world(pose, body_twist)
        req.len = 2
        req.dt = 0
        req.header.stamp = rospy.Time.now()
        req.header.frame_id = f'{self._tf_namespace}/vehicle'
        req.curr_time = rospy.Time.now()

        try:
           res = self._get_traj_service(req)
        except rospy.ServiceException:
            return None, None

        if not res.success:
            return None, None

        target_pose = res.poses[0]
        target_world_twist = res.twists[0]
        return target_pose, target_world_twist

    # def _get_target(self) -> (Optional[Pose], Optional[Twist]):
    #     t = rospy.Time.now().to_sec()
        
    #     # Zig-zag parameters
    #     v_x = 0.2
    #     A = 1.0
    #     omega = 0.5
    #     z_depth = 1.0

    #     # Start position at (5, 5, -2)
    #     start_x = 5.0
    #     start_y = -5.0
    #     start_z = 2.0

    #     x = start_x + v_x * t
    #     y = start_y + A * np.sin(omega * t)
    #     z = start_z + z_depth

    #     # Heading calculation with auto-winding
    #     yaw = 0  # Replace with your dynamic calculation if needed
    #     pitch = 0
    #     roll = 0

    #     # if not hasattr(self, '_last_yaw'):
    #     #     self._last_yaw = yaw  # Initialize last_yaw for the first call

    #     # yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    #     # self._last_yaw = (self._last_yaw + np.pi) % (2 * np.pi) - np.pi
    #     # if yaw - self._last_yaw < -np.pi:
    #     #     yaw += 2 * np.pi
    #     # if yaw - self._last_yaw > np.pi:
    #     #     yaw -= 2 * np.pi
    #     # self._last_yaw = yaw

    #     # Convert Euler angles to quaternion
    #     target_rotation = Rotation.from_euler('ZYX', [yaw, pitch, roll])
    #     q = target_rotation.as_quat()

    #     # Compute velocities
    #     vx = v_x
    #     vy = A * omega * np.cos(omega * t)
    #     vz = 0.0
    #     wx, wy, wz = 0.0, 0.0, 0.0

    #     # Apply limits and constraints to y, z
    #     if not self._y_reached_limit:
    #         if y > 10.0:
    #             y = 3.0
    #             vy = 0.0
    #             self._y_reached_limit = True
    #             rospy.loginfo("Y-axis reached upper limit.")
    #         elif y < -10.0:
    #             y = -3.0
    #             vy = 0.0
    #             self._y_reached_limit = True
    #             rospy.loginfo("Y-axis reached lower limit.")
    #     else:
    #         y = np.clip(y, -3.0, 3.0)
    #         vy = 0.0  # Hold velocity at zero

    #     if not self._z_reached_limit:
    #         if z > 2.0:
    #             z = 2.0
    #             vz = 0.0
    #             self._z_reached_limit = True
    #             rospy.loginfo("Z-axis reached upper limit.")
    #         elif z < -3.0:
    #             z = -3.0
    #             vz = 0.0
    #             self._z_reached_limit = True
    #             rospy.loginfo("Z-axis reached lower limit.")
    #     else:
    #         z = np.clip(z, -3.0, 2.0)
    #         vz = 0.0  # Hold velocity at zero

    #     if x > 15.0:
    #         x = 15.0
    #         vx = 0.0
    #         rospy.loginfo("X-axis reached maximum limit.")

    #     # Create target pose
    #     target_pose = Pose()
    #     target_pose.position.x = x
    #     target_pose.position.y = y
    #     target_pose.position.z = z
    #     target_pose.orientation.x = q[0]
    #     target_pose.orientation.y = q[1]
    #     target_pose.orientation.z = q[2]
    #     target_pose.orientation.w = q[3]

    #     # Create target twist
    #     target_twist = Twist()
    #     target_twist.linear.x = vx
    #     target_twist.linear.y = vy
    #     target_twist.linear.z = vz
    #     target_twist.angular.x = wx
    #     target_twist.angular.y = wy
    #     target_twist.angular.z = wz

    #     return target_pose, target_twist



    def _build_pids(self):
        pids = []
        # x, y, z, roll, pitch, yaw
        for i in range(6):
            pid = PID(
                Kp=self._kp[i],
                Ki=self._ki[i],
                Kd=self._kd[i],
                output_limits=(self._limits[i, 0], self._limits[i, 1]),
                error_map=pi_clip if i == 5 else lambda x: x,
                # error_map=pi_clip if 3 <= i <= 5 else lambda x: x,
                proportional_on_measurement=False,
                sample_time=self._dt,
                d_alpha=self._dt / self._tau[i] if self._tau[i] > 0 else 1
            )
            pids.append(pid)

        self._pids = pids

    def _load_config(self):
        self._frequency = rospy.get_param('~frequency')
        self._tf_namespace = rospy.get_param('tf_namespace')
        self._kp = np.array(rospy.get_param('~kp'), dtype=np.float64)
        self._ki = np.array(rospy.get_param('~ki'), dtype=np.float64)
        self._kd = np.array(rospy.get_param('~kd'), dtype=np.float64)
        self._tau = np.array(rospy.get_param('~tau'), dtype=np.float64)
        self._limits = np.array(rospy.get_param('~limits'), dtype=np.float64)

    def _handle_navigation_state(self, msg: NavigationState):
        self._navigation_state = msg

    def _handle_tune_pid_planner(self, req: TunePIDPlannerRequest) -> TunePIDPlannerResponse:
        fields = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}

        for tuning in req.tunings:
            field = fields.get(tuning.axis)
            if field is None:
                return TunePIDPlannerResponse(False)

            self._kp[field] = tuning.kp
            self._ki[field] = tuning.ki
            self._kd[field] = tuning.kd
            self._tau[field] = tuning.tau
            self._limits[field, 0] = tuning.limits[0]
            self._limits[field, 1] = tuning.limits[1]

        self._build_pids()
        return TunePIDPlannerResponse(True)


def main():
    rospy.init_node('pid_planner')
    p = PIDPlanner()
    p.start()

if __name__ == '__main__':
    main()
