#!/usr/bin/env python3
from larcc_classes.data_storage.DataForLearning import DataForLearning
import rospy
import time
import numpy as np
import os
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import numpy as np
import matplotlib.pyplot as plt


from geometry_msgs.msg import Wrench, Pose, WrenchStamped
from tf2_msgs.msg import TFMessage
from colorama import Fore


def wrench_callback(data):
    global wrench_force_torque
    wrench_force_torque = data.wrench


if __name__ == '__main__':
    wrench_force_torque = Wrench()
    rospy.init_node("FT_visualizer", anonymous=True)

    rospy.Subscriber("/wrench", WrenchStamped, wrench_callback)

    while not rospy.is_shutdown():
        print("\n")
        print(Fore.LIGHTBLUE_EX + 'wrench_force_torque: \n' + Fore.YELLOW + str(wrench_force_torque) + '\n')

        plt.axis([0, 10, 0, 1])

        for i in range(10):
            y = np.random.random()
            plt.scatter(i, y)
            plt.pause(0.05)

        plt.show()
        # time.sleep(1)