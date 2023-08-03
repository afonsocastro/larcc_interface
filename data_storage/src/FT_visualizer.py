#!/usr/bin/env python3
from larcc_classes.data_storage.DataForLearning import DataForLearning
import rospy
import time
import numpy as np
import os
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Wrench, Pose, WrenchStamped
from tf2_msgs.msg import TFMessage
from colorama import Fore


def wrench_callback(data):
    global wrench_force_torque
    wrench_force_torque = data.wrench


if __name__ == '__main__':
    wrench_force_torque = Wrench()
    rospy.init_node("test", anonymous=True)

    rospy.Subscriber("/wrench", WrenchStamped, wrench_callback)

    while not rospy.is_shutdown():
        print("\n")
        print(Fore.LIGHTBLUE_EX + 'wrench_force_torque: \n' + Fore.YELLOW + str(wrench_force_torque) + '\n')
        time.sleep(1)