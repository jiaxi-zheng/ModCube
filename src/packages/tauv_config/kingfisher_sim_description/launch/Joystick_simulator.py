#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Joy

def publish_joy():
    # Initialize the ROS node
    rospy.init_node('virtual_joystick')

    # Create a publisher object
    joy_pub = rospy.Publisher('/joy', Joy, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    # Create a Joy message
    joy_msg = Joy()
    joy_msg.buttons = [0]*12  # Initialize all buttons to 0
    joy_msg.axes = [0.0]*8    # Initialize all axes to 0.0

    # Set the Y button (typically button 3, but depends on the joystick mapping)
    joy_msg.buttons[3] = 1  # Change the index if the Y button is mapped differently

    # Publish the message in a loop
    while not rospy.is_shutdown():
        joy_msg.header.stamp = rospy.Time.now()
        joy_pub.publish(joy_msg)
        rospy.loginfo("Published Y button press")
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_joy()
    except rospy.ROSInterruptException:
        pass
