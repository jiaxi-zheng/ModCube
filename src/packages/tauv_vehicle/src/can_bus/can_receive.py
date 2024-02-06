import can
import rospy
from can_msgs.msg import Frame
from tauv_msgs.msg import Can
import struct
import time

bus = can.interface.Bus(channel='can1', bustype='socketcan')
fram_num = 0

rospy.init_node('can_receiver')  

Can_data = Can()

Can_pub = rospy.Publisher('can_frames', Can, queue_size=10)

while not rospy.is_shutdown():
    msg = bus.recv()
    if msg is not None:
        if msg.arbitration_id == 0x301:  

            use_data_buf = msg.data

            if len(msg.data) == 8:  
                Can_data.FB_auv_pit[0] = struct.unpack('>h', use_data_buf[0:2])[0]
                Can_data.FB_auv_rol[0] = struct.unpack('>h', use_data_buf[2:4])[0]
                Can_data.FB_auv_yaw[0] = struct.unpack('>h', use_data_buf[4:6])[0]
                Can_data.FB_auv_deep[0] = struct.unpack('>h', use_data_buf[6:8])[0]

                # print("###########")

                Can_pub.publish(Can_data)

        elif msg.arbitration_id == 0x302:

            use_data_buf = msg.data

            if len(msg.data) == 8:  
                Can_data.FB_auv_deep_vel[0] = struct.unpack('>h', use_data_buf[0:2])[0]
                Can_data.FB_auv_ang_vel_pit[0] = struct.unpack('>h', use_data_buf[2:4])[0]
                Can_data.FB_auv_ang_vel_rol[0] = struct.unpack('>h', use_data_buf[4:6])[0]
                Can_data.FB_auv_ang_vel_yaw[0] = struct.unpack('>h', use_data_buf[6:8])[0]

                Can_pub.publish(Can_data)
