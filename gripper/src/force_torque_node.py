#!/usr/bin/env python3

import rospy
import time
import signal
# from std_msgs.msg import
from sensor_msgs.msg import JointState
from RobotiqHand import RobotiqHand
import matplotlib.pyplot as plt

HOST = "192.168.56.2"
PORT = 54321

cont = True


def handler(signal, frame):
    global cont
    cont = False


def joints_callback(data):
    # rospy.loginfo(data.header.stamp.secs)
    # rospy.loginfo(data.header.stamp.nsecs)
    rospy.loginfo(data.effort)


if __name__ == '__main__':

    rospy.init_node('force_torque_node', anonymous=True)

    print("Node initiated")

    # rate = rospy.Rate(10.0)  # 10hz
    rate = rospy.Rate(5000.0)  # 10hz

    signal.signal(signal.SIGINT, handler)

    print('test_force start')
    hand = RobotiqHand()
    hand.connect(HOST, PORT)

    print('activate: start')
    hand.reset()
    hand.activate()
    result = hand.wait_activate_complete()
    print('activate: result = 0x{:02x}'.format(result))
    if result != 0x31:
        hand.disconnect()
    print('adjust: start')
    hand.adjust()
    print('adjust: finish')

    rospy.Subscriber("joint_states", JointState, joints_callback)
    # time.sleep(1)

    Year = [1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
    Unemployment_Rate = [9.8, 12, 8, 7.2, 6.9, 7, 6.5, 6.2, 5.5, 6.3]

    plt.plot(Year, Unemployment_Rate)
    plt.title('Unemployment Rate Vs Year')
    plt.xlabel('Year')
    plt.ylabel('Unemployment Rate')
    plt.show()

    while not rospy.is_shutdown():
        status = hand.get_instant_gripper_status()
        print(status.actual_force_motor_current)
        now = rospy.get_rostime()
        rospy.loginfo("Current time %i %i", now.secs, now.nsecs)
        rate.sleep()
    rospy.spin()

    # while cont:
    #     time.sleep(0.05)
    #     status = hand.get_instant_gripper_status()

    hand.disconnect()