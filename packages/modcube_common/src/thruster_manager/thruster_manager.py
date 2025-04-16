import rospy
import numpy as np
from typing import Optional
import tf2_ros as tf2
from modcube_util.transforms import tf2_transform_to_translation, tf2_transform_to_quat, quat_to_rotm
from modcube_util.types import tl
from visualization_msgs.msg import Marker, MarkerArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64

class ThrusterManager:

    def __init__(self):
        self._load_config()

        self._dt: float = 1.0 / self._frequency
        self._num_thrusters: int = len(self._thruster_ids)
        self._wrench: Optional[WrenchStamped] = None

        self._tf_buffer: tf2.Buffer = tf2.Buffer()
        self._tf_listener: tf2.TransformListener = tf2.TransformListener(self._tf_buffer)

        self.transs = np.zeros((self._num_thrusters, 3))
        self.quats = np.zeros((self._num_thrusters, 4))

        self._wrench_sub: rospy.Subscriber = rospy.Subscriber('gnc/target_wrench', WrenchStamped, self._handle_wrench)
        self._target_thrust_pubs: [rospy.Publisher] = []
        self._marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)

        for thruster_id in self._thruster_ids:
            target_thrust_pub = rospy.Publisher(
                f'vehicle/thrusters/{thruster_id}/target_thrust',
                Float64,
                queue_size=10
            )
            self._target_thrust_pubs.append(target_thrust_pub)

        self._build_tam()

    def _publish_thrust_markers(self, transs, quats, thrusts):
        marker_array = MarkerArray()
        # for i, (trans, quat, thrust) in enumerate(zip(transs, quats, thrusts)):
        for i in range(self._num_thrusters):
            # Create an arrow marker
            marker = Marker()
            # marker.header.frame_id = f"{self._tf_namespace}/thruster_{self._thruster_ids[i]}"
            marker.header.frame_id = "kf/vehicle"

            marker.header.stamp = rospy.Time.now()
            marker.ns = "thrusters"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            # Set the pose of the marker to the thruster's frame
            marker.pose.position.x = transs[i,0]
            marker.pose.position.y = transs[i,1]
            marker.pose.position.z = transs[i,2]
            marker.pose.orientation.x = quats[i,0]
            marker.pose.orientation.y = quats[i,1]
            marker.pose.orientation.z = quats[i,2]
            marker.pose.orientation.w = quats[i,3]

            # Set the scale of the arrow (length is proportional to thrust)
            marker.scale.x = thrusts[i]*0.01  # arrow length
            marker.scale.y = 0.005  # arrow width
            marker.scale.z = 0.005  # arrow height

            # Set the color of the arrow (e.g., blue)
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker.lifetime = rospy.Duration(0.2)  # How long the arrow will be displayed

            marker_array.markers.append(marker)

        # Publish the MarkerArray
        self._marker_pub.publish(marker_array)

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _command_thrusts(self, thrusts: [float]):
        for (i, thrust) in enumerate(thrusts):
            thrust_msg = Float64()
            thrust_msg.data = thrust
            self._target_thrust_pubs[i].publish(thrust_msg)

    def _update(self, timer_event):
        if self._wrench is None:
            return

        tau = np.hstack((
            tl(self._wrench.wrench.force),
            tl(self._wrench.wrench.torque)
        ))

        thrusts = self._inv_tam @ tau

        self._publish_thrust_markers(self.transs, self.quats, thrusts)

        self._command_thrusts(thrusts)

    def _handle_wrench(self, wrench: WrenchStamped):
        self._wrench = wrench

    def _build_tam(self):
        tam: np.array = np.zeros((6, self._num_thrusters))
        for (i, thruster_id) in enumerate(self._thruster_ids):
            base_frame = f'{self._tf_namespace}/vehicle'
            thruster_frame = f'{self._tf_namespace}/thruster_{thruster_id}'
            current_time = rospy.Time.now()
            try:
                transform = self._tf_buffer.lookup_transform(
                    base_frame,
                    thruster_frame,
                    current_time,
                    rospy.Duration(30.0)
                )

                trans = tf2_transform_to_translation(transform)
                quat = tf2_transform_to_quat(transform)

                self.transs[i] = trans
                self.quats[i] = quat

                rotm = quat_to_rotm(quat)

                force = rotm @ np.array([1, 0, 0])
                torque = np.cross(trans, force)

                tau = np.hstack((force, torque)).transpose()

                tam[:, i] = tau

            except (tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException) as e:
                rospy.logerr(f'Could not get transform from {base_frame} to {thruster_frame}: {e}')

        self._inv_tam = np.linalg.pinv(tam)

    def _load_config(self):
        self._tf_namespace = rospy.get_param('tf_namespace')
        self._frequency = rospy.get_param('~frequency')
        self._thruster_ids = np.array(rospy.get_param('~thruster_ids'))

def main():
    rospy.init_node('thruster_manager')
    t = ThrusterManager()
    t.start()