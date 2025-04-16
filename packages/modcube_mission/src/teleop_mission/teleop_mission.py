import rospy
import argparse
from typing import Optional
import numpy as np
from math import atan2, cos, sin, e, pi
from modcube_msgs.msg import PIDTuning, DynamicsTuning, DynamicsParameterConfigUpdate
from modcube_msgs.srv import \
    TuneController, TuneControllerRequest,\
    TunePIDPlanner, TunePIDPlannerRequest,\
    TuneDynamics, TuneDynamicsRequest,\
    UpdateDynamicsParameterConfigs, UpdateDynamicsParameterConfigsRequest, UpdateDynamicsParameterConfigsResponse
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
from std_srvs.srv import SetBool, Trigger
from std_msgs.msg import Float64
from modcube_msgs.srv import MapFind, MapFindRequest, MapFindClosest, MapFindClosestRequest
from motion.motion_utils import MotionUtils
from motion.trajectories import TrajectoryStatus
from motion_client import MotionClient
from spatialmath import SE3, SO3, SE2, SO2
from nav_msgs.msg import Path, Odometry as OdometryMsg


class ArgumentParserError(Exception): pass


class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgumentParserError(message)

class TeleopMission:

    def __init__(self):

        self._parser = self._build_parser()

        self._motion = MotionUtils()
        # self._motion = MotionClient()

        self._pose: Optional[Pose] = None
        self._twist: Optional[Twist] = None

        self._tune_controller_srv: rospy.ServiceProxy = rospy.ServiceProxy('gnc/tune_controller', TuneController)
        self._tune_pid_planner_srv: rospy.ServiceProxy = rospy.ServiceProxy('gnc/tune_pid_planner', TunePIDPlanner)
        self._tune_dynamics_srv: rospy.ServiceProxy = rospy.ServiceProxy('gnc/tune_dynamics', TuneDynamics)
        self._update_dynamics_parameter_configs_srv: rospy.ServiceProxy = rospy.ServiceProxy(
            'gnc/update_dynamics_parameter_configs', UpdateDynamicsParameterConfigs
        )

        self._find_srv = rospy.ServiceProxy("global_map/find", MapFind)
        self._find_closest_srv = rospy.ServiceProxy("global_map/find_closest", MapFindClosest)
        self._map_reset_srv = rospy.ServiceProxy("global_map/reset", Trigger)
        self._map_reset_srv = rospy.ServiceProxy("global_map/reset", Trigger)
        self._odom_sub = rospy.Subscriber('/kf/gnc/odom', OdometryMsg, self._handle_odom)


        self._prequal_timer = None

    def start(self):
        while not rospy.is_shutdown():
            cmd = input('>>> ')

            try:
                args = self._parser.parse_args(cmd.split())
                args.func(args)
            except ArgumentParserError as e:
                print('error:', e)
                continue

    def _handle_tune_controller(self, args):
        print('tune_controller', args.z, args.roll, args.pitch)
        pid_tunings = []

        if args.z is not None:
            p = PIDTuning(
                axis="z",
                kp=args.z[0],
                ki=args.z[1],
                kd=args.z[2],
                tau=args.z[3],
                limits=args.z[4:6]
            )
            pid_tunings.append(p)

        if args.roll is not None:
            p = PIDTuning(
                axis="roll",
                kp=args.roll[0],
                ki=args.roll[1],
                kd=args.roll[2],
                tau=args.roll[3],
                limits=args.roll[4:6]
            )
            pid_tunings.append(p)

        if args.pitch is not None:
            p = PIDTuning(
                axis="pitch",
                kp=args.pitch[0],
                ki=args.pitch[1],
                kd=args.pitch[2],
                tau=args.pitch[3],
                limits=args.pitch[4:6]
            )
            pid_tunings.append(p)

        req = TuneControllerRequest()
        req.tunings = pid_tunings
        try:
            self._tune_controller_srv.call(req)
        except Exception as e:
            print(e)

    def _handle_tune_pid_planner(self, args):
        print('tune_pid_planner', args.x, args.y, args.z, args.roll, args.pitch, args.yaw)

        pid_tunings = []

        if args.x is not None:
            p = PIDTuning(
                axis="x",
                kp=args.x[0],
                ki=args.x[1],
                kd=args.x[2],
                tau=args.x[3],
                limits=args.x[4:6]
            )
            pid_tunings.append(p)

        if args.y is not None:
            p = PIDTuning(
                axis="y",
                kp=args.y[0],
                ki=args.y[1],
                kd=args.y[2],
                tau=args.y[3],
                limits=args.y[4:6]
            )
            pid_tunings.append(p)

        if args.z is not None:
            p = PIDTuning(
                axis="z",
                kp=args.z[0],
                ki=args.z[1],
                kd=args.z[2],
                tau=args.z[3],
                limits=args.z[4:6]
            )
            pid_tunings.append(p)

        if args.roll is not None:
            p = PIDTuning(
                axis="roll",
                kp=args.roll[0],
                ki=args.roll[1],
                kd=args.roll[2],
                tau=args.roll[3],
                limits=args.roll[4:6]
            )
            pid_tunings.append(p)

        if args.pitch is not None:
            p = PIDTuning(
                axis="pitch",
                kp=args.pitch[0],
                ki=args.pitch[1],
                kd=args.pitch[2],
                tau=args.pitch[3],
                limits=args.pitch[4:6]
            )
            pid_tunings.append(p)

        if args.yaw is not None:
            p = PIDTuning(
                axis="yaw",
                kp=args.yaw[0],
                ki=args.yaw[1],
                kd=args.yaw[2],
                tau=args.yaw[3],
                limits=[args.yaw[4], args.yaw[5]]
            )
            pid_tunings.append(p)

        req = TunePIDPlannerRequest()
        req.tunings = pid_tunings
        try:
            self._tune_pid_planner_srv.call(req)
        except Exception as e:
            print(e)

    def _handle_tune_dynamics(self, args):
        print('tune_dynamics', args.mass, args.volume, args.water_density, args.center_of_gravity,
              args.center_of_buoyancy, args.moments, args.linear_damping, args.quadratic_damping, args.added_mass)

        t = DynamicsTuning()
        if args.mass is not None:
            t.update_mass = True
            t.mass = args.mass

        if args.volume is not None:
            t.update_volume = True
            t.volume = args.volume

        if args.water_density is not None:
            t.update_water_density = True
            t.water_density = args.water_density

        if args.center_of_gravity is not None:
            t.update_center_of_gravity = True
            t.center_of_gravity = args.center_of_gravity

        if args.center_of_buoyancy is not None:
            t.update_center_of_buoyancy = True
            t.center_of_buoyancy = args.center_of_buoyancy

        if args.moments is not None:
            t.update_moments = True
            t.moments = args.moments

        if args.linear_damping is not None:
            t.update_linear_damping = True
            t.linear_damping = args.linear_damping

        if args.quadratic_damping is not None:
            t.update_quadratic_damping = True
            t.quadratic_damping = args.quadratic_damping

        if args.added_mass is not None:
            t.update_added_mass = True
            t.added_mass = args.added_mass

        req: TuneDynamicsRequest = TuneDynamicsRequest()
        req.tuning = t
        self._tune_dynamics_srv.call(req)


    def _handle_odom(self, msg: OdometryMsg):
        self.pose = msg.pose.pose
        
###################constant accelaration#####################
    # def _handle_goto(self, args):
    #     print('goto')
    #     pose = SE3.Rt(SO3.Rx(np.deg2rad(args.yaw)), np.array([args.x, args.y, args.z]))
    #     self._motion.goto(pose)
        
###################minimum snap#####################
    def _handle_goto(self, args):
        v = args.v if args.v is not None else .1
        a = args.a if args.a is not None else .1
        j = args.j if args.j is not None else .4

        x1 = self.pose.position.x
        y1 = self.pose.position.y
        z1 = self.pose.position.z

        x2 = x1 + 1
        y2 = y1 + 1
        z2 = z1 - 0.3

        x3 = x1 + 2
        y3 = y1 
        z3 = z1 - 0.6
        
        x4 = x1 + 1 
        y4 = y1 - 1
        z4 = z1 - 0.9

        x5 = x1  
        y5 = y1 - 2
        z5 = z1 - 0.6

        x6 = x1 + 1 
        y6 = y1 - 3
        z6 = z1 - 0.3
        
        x7 = x1 + 2
        y7 = y1 - 2
        z7 = z1 

        x8 = x1 + 1 
        y8 = y1 - 1
        z8 = z1 
        
        x9 = x1
        y9 = y1
        z9 = z1

        x10 = x1 + 1
        y10 = y1 + 1
        z10 = z1 - 0.3

        q11 = 1
        q12 = 0
        q13 = 0
        q14 = 0

        pose1 = Pose()
        pose2 = Pose()
        pose2 = Pose()
        pose3 = Pose()
        pose4 = Pose()
        pose5 = Pose()
        pose6 = Pose()
        pose7 = Pose()
        pose8 = Pose()
        pose9 = Pose()
        pose10 = Pose()
        pose1.position = Point(x1, y1, z1)
        pose2.position = Point(x2, y2, z2)
        pose3.position = Point(x3, y3, z3)
        pose4.position = Point(x4, y4, z4)
        pose5.position = Point(x5, y5, z5)
        pose6.position = Point(x6, y6, z6)
        pose7.position = Point(x7, y7, z7)
        pose8.position = Point(x8, y8, z8)
        pose9.position = Point(x9, y9, z9)
        pose10.position = Point(x10, y10, z10)

        pose1.orientation = Quaternion(q11, q12, q13, q14)
        pose2.orientation = Quaternion(q11, q12, q13, q14)    
        pose3.orientation = Quaternion(q11, q12, q13, q14)    
        pose4.orientation = Quaternion(q11, q12, q13, q14)    
        pose5.orientation = Quaternion(q11, q12, q13, q14)    
        pose6.orientation = Quaternion(q11, q12, q13, q14)    
        pose7.orientation = Quaternion(q11, q12, q13, q14)    
        pose8.orientation = Quaternion(q11, q12, q13, q14)    
        pose9.orientation = Quaternion(q11, q12, q13, q14)    
        pose10.orientation = Quaternion(q11, q12, q13, q14)    

        poses_list = [pose1,pose2,pose3,pose4,pose5,pose6,pose7,pose8,pose9,pose10]

        try:
            self._motion.goto(
                poses_list,
                # args.yaw,
                v=v,
                a=a,
                j=j,
                block=TrajectoryStatus.EXECUTING
            )
        except Exception as e:
            print("Exception from teleop_mission! (Gleb)")
            print(e)

    def _handle_goto_relative(self, args):
        v = args.v if args.v is not None else .1
        a = args.a if args.a is not None else .1
        j = args.j if args.j is not None else .4

        try:
            self._motion.goto_relative(
                (args.x, args.y, args.z),
                args.yaw,
                v=v,
                a=a,
                j=j
            )
        except Exception as e:
            print(e)

    def _handle_config_param_est(self, args):
        req = UpdateDynamicsParameterConfigsRequest()
        update = DynamicsParameterConfigUpdate()
        update.name = args.name

        if args.initial_value is not None:
            update.update_initial_value = True
            update.initial_value = args.initial_value

        if args.fixed is not None:
            update.update_fixed = True
            update.fixed = args.fixed == "true"

        if args.initial_covariance is not None:
            update.update_initial_covariance = True
            update.initial_covariance = args.initial_covariance

        if args.process_covariance is not None:
            update.update_process_covariance = True
            update.process_covariance = args.process_covariance

        if args.limits is not None:
            update.update_limits = True
            update.limits = args.limits

        update.reset = args.reset

        req.updates = [update]

        self._update_dynamics_parameter_configs_srv.call(req)

    def _handle_prequal(self, args):
        if self._prequal_timer is not None:
            self._prequal_timer.shutdown()

        self._prequal_timer = rospy.Timer(rospy.Duration(30), self._handle_update_prequal, oneshot=True)

    def _handle_update_prequal(self, timer_event):
        self._motion.reset()

        print('running!')

        depth = 1.5

        start_position = self._motion.get_position()
        start_yaw = self._motion.get_orientation()[2]

        waypoints = np.array([
            [0, 0, depth, 0],
            [3, 0, depth, 0],
            [12, 2, depth, 0],
            [12, -2, depth, 0],
            [3, 0, depth, 0],
            [0, 0, depth, 0],
        ])
        n_waypoints = waypoints.shape[0]

        for i in range(n_waypoints):
            position = waypoints[i, 0:3]
            yaw = waypoints[i, 3]

            transformed_position = start_position + np.array([
                position[0] * cos(start_yaw) + position[1] * -sin(start_yaw),
                position[0] * sin(start_yaw) + position[1] * cos(start_yaw),
                position[2]
            ])

            transformed_yaw = start_yaw + yaw

            self._motion.goto(transformed_position, transformed_yaw, v=0.3, a=0.05, j=0.04)


    def _build_parser(self) -> argparse.ArgumentParser:
        parser = ThrowingArgumentParser(prog="teleop_mission")
        subparsers = parser.add_subparsers()

        tune_controller = subparsers.add_parser('tune_controller')
        tune_controller.add_argument('--z', type=float, nargs=6)
        tune_controller.add_argument('--roll', type=float, nargs=6)
        tune_controller.add_argument('--pitch', type=float, nargs=6)
        tune_controller.set_defaults(func=self._handle_tune_controller)

        tune_pid_planner = subparsers.add_parser('tune_pid_planner')
        tune_pid_planner.add_argument('--x', type=float, nargs=6)
        tune_pid_planner.add_argument('--y', type=float, nargs=6)
        tune_pid_planner.add_argument('--z', type=float, nargs=6)
        tune_pid_planner.add_argument('--roll', type=float, nargs=6)
        tune_pid_planner.add_argument('--pitch', type=float, nargs=6)
        tune_pid_planner.add_argument('--yaw', type=float, nargs=6)
        tune_pid_planner.set_defaults(func=self._handle_tune_pid_planner)

        tune_dynamics = subparsers.add_parser('tune_dynamics')
        tune_dynamics.add_argument('--mass', type=float)
        tune_dynamics.add_argument('--volume', type=float)
        tune_dynamics.add_argument('--water_density', type=float)
        tune_dynamics.add_argument('--center_of_gravity', type=float, nargs=3)
        tune_dynamics.add_argument('--center_of_buoyancy', type=float, nargs=3)
        tune_dynamics.add_argument('--moments', type=float, nargs=6)
        tune_dynamics.add_argument('--linear_damping', type=float, nargs=6)
        tune_dynamics.add_argument('--quadratic_damping', type=float, nargs=6)
        tune_dynamics.add_argument('--added_mass', type=float, nargs=6)
        tune_dynamics.set_defaults(func=self._handle_tune_dynamics)

        goto = subparsers.add_parser('goto')
        goto.add_argument('x', type=float)
        goto.add_argument('y', type=float)
        goto.add_argument('z', type=float)
        goto.add_argument('q1', type=float)
        goto.add_argument('q2', type=float)
        goto.add_argument('q3', type=float)
        goto.add_argument('q4', type=float)
        # goto.add_argument('roll', type=float)
        # goto.add_argument('pitch', type=float)
        # goto.add_argument('yaw', type=float)
        goto.add_argument('--v', type=float)
        goto.add_argument('--a', type=float)
        goto.add_argument('--j', type=float)
        goto.set_defaults(func=self._handle_goto)

        goto_relative = subparsers.add_parser('goto_relative')
        goto_relative.add_argument('x', type=float)
        goto_relative.add_argument('y', type=float)
        goto_relative.add_argument('z', type=float)
        goto_relative.add_argument('yaw', type=float)
        goto_relative.add_argument('--v', type=float)
        goto_relative.add_argument('--a', type=float)
        goto_relative.add_argument('--j', type=float)
        goto_relative.set_defaults(func=self._handle_goto_relative)

        config_param_est = subparsers.add_parser('config_param_est')
        config_param_est.add_argument('name', type=str)
        config_param_est.add_argument('--initial_value', type=float)
        config_param_est.add_argument('--fixed', type=str, choices=('true', 'false'))
        config_param_est.add_argument('--initial_covariance', type=float)
        config_param_est.add_argument('--process_covariance', type=float)
        config_param_est.add_argument('--limits', type=float, nargs=2)
        config_param_est.add_argument('--reset', default=False, action='store_true')
        config_param_est.set_defaults(func=self._handle_config_param_est)

        prequal = subparsers.add_parser('prequal')
        prequal.set_defaults(func=self._handle_prequal)

        return parser


def main():
    rospy.init_node('teleop_mission')
    m = TeleopMission()
    m.start()