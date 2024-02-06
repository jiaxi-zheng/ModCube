import can
import time
import can
import rospy
from can_msgs.msg import Frame
from tauv_msgs.msg import Can, Ctrl_cmd
import struct
import time

bus = can.interface.Bus(channel='can0', bustype='socketcan')

Ctrl_cmd_sub = rospy.Subscriber('/Ctrl_cmd', Ctrl_cmd, Cmd_callback)
Ctrl_cmd_sub = rospy.Subscriber('/Ctrl_cmd', Ctrl_cmd, Cmd_callback)

def Cmd_callback(data):
    msgs = [can.Message(arbitration_id=0x330, data=[data.Ctrl_vel_X[0],data.Ctrl_vel_Y[0],data.Ctrl_vel_Z[0],data.Ctrl_fixed_Z[0],
                                                    data.Ctrl_vel_Rol[0],data.Ctrl_vel_Pit[0],data.Ctrl_vel_Yaw[0],data.Ctrl_fixed_Yaw[0]]),
            can.Message(arbitration_id=0x331, data=[data.Ctrl_vel_X[1],data.Ctrl_vel_Y[1],data.Ctrl_vel_Z[1],data.Ctrl_fixed_Z[1],
                                                    data.Ctrl_vel_Rol[1],data.Ctrl_vel_Pit[1],data.Ctrl_vel_Yaw[1],data.Ctrl_fixed_Yaw[1]]),
            can.Message(arbitration_id=0x332, data=[data.Ctrl_vel_X[2],data.Ctrl_vel_Y[2],data.Ctrl_vel_Z[2],data.Ctrl_fixed_Z[2],
                                                    data.Ctrl_vel_Rol[2],data.Ctrl_vel_Pit[2],data.Ctrl_vel_Yaw[2],data.Ctrl_fixed_Yaw[2]]),
            can.Message(arbitration_id=0x333, data=[data.Ctrl_vel_X[3],data.Ctrl_vel_Y[3],data.Ctrl_vel_Z[3],data.Ctrl_fixed_Z[3],
                                                    data.Ctrl_vel_Rol[3],data.Ctrl_vel_Pit[3],data.Ctrl_vel_Yaw[3],data.Ctrl_fixed_Yaw[3]]),
            can.Message(arbitration_id=0x334, data=[data.Ctrl_pivot_1[0],data.Ctrl_pivot_2[0],data.Ctrl_pivot_3[0],data.Ctrl_pivot_4[0],
                                                    data.Ctrl_emagnet_1[0],data.Ctrl_emagnet_2[0],data.Ctrl_emagnet_3[0],data.Ctrl_emagnet_4[0]]),
            can.Message(arbitration_id=0x334, data=[data.Ctrl_pivot_1[1],data.Ctrl_pivot_2[1],data.Ctrl_pivot_3[1],data.Ctrl_pivot_4[1],
                                                    data.Ctrl_emagnet_1[1],data.Ctrl_emagnet_2[1],data.Ctrl_emagnet_3[1],data.Ctrl_emagnet_4[1]]),
            can.Message(arbitration_id=0x334, data=[data.Ctrl_pivot_1[2],data.Ctrl_pivot_2[2],data.Ctrl_pivot_3[2],data.Ctrl_pivot_4[2],
                                                    data.Ctrl_emagnet_1[2],data.Ctrl_emagnet_2[2],data.Ctrl_emagnet_3[2],data.Ctrl_emagnet_4[2]]),
            can.Message(arbitration_id=0x334, data=[data.Ctrl_pivot_1[3],data.Ctrl_pivot_2[3],data.Ctrl_pivot_3[3],data.Ctrl_pivot_4[3],
                                                    data.Ctrl_emagnet_1[3],data.Ctrl_emagnet_2[3],data.Ctrl_emagnet_3[3],data.Ctrl_emagnet_4[3]])]

    for msg in msgs:
        bus.send(msg)
        print("Sent CAN message: ID=0x{:X}, data={}".format(msg.arbitration_id, bytes(msg.data)))

if __name__ == '__main__':
    import sys 
    print(sys.version) 
    rospy.init_node('can_transmit', anonymous=True)
    # gpr = GPR(optimize=False)
    rospy.spin()

#     xx, yy, zz = 0, 0, 0

#     while True:

#         xx += 1
#         yy += 1
#         zz += 1
        
#         xx = range_protect(xx,255,0)
#         yy = range_protect(yy,255,0)
#         zz = range_protect(zz,255,0)
        
#         mmsg(xx, yy, zz)

#         time.sleep(0.1)

# bus.shutdown()