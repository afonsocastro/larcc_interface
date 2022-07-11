#!/usr/bin/env python3
import json

import rospy
import time
import numpy as np
import os
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Wrench, Pose, WrenchStamped
from tf2_msgs.msg import TFMessage
from colorama import Fore
import argparse
from gripper.src.RobotiqHand import RobotiqHand
from lib.src import ArmGripperComm as ag


cont = True

array = np.empty((1, 28), dtype=float)

first_read_exist = False
first_timestamp = (0, 0)


class DataForLearning:

    def __init__(self):

        self.joints_time = (0, 0)
        self.wrench_time = (0, 0)
        self.tf_time = (0, 0)

        self.joints_position = []
        self.joints_effort = []

        self.wrench_pose = Pose()

        self.wrench_force_torque = Wrench()

        self.gripper_current = 0

        self.object_detection = 0

        rospy.Subscriber("joint_states", JointState, self.joint_states_callback)
        rospy.Subscriber("tf", TFMessage, self.tf_callback)
        rospy.Subscriber("wrench", WrenchStamped, self.wrench_callback)

    def joint_states_callback(self, data):

        self.joints_effort = data.effort
        self.joints_position = data.position
        self.joints_time = str(rospy.Time.now())

    def tf_callback(self, data):

        self.wrench_pose.position.x = data.transforms[0].transform.translation.x
        self.wrench_pose.position.y = data.transforms[0].transform.translation.y
        self.wrench_pose.position.z = data.transforms[0].transform.translation.z
        self.wrench_pose.orientation = data.transforms[0].transform.rotation
        self.tf_time = str(rospy.Time.now())

    def wrench_callback(self, data):
        self.wrench_force_torque = data.wrench
        self.wrench_time = str(rospy.Time.now())

    def timestamp(self):

        j_times = float(str(self.joints_time[:10]) + "." + str(self.joints_time[10:]))
        t_times = float(str(self.tf_time[:10]) + "." + str(self.tf_time[10:]))
        w_times = float(str(self.wrench_time[:10]) + "." + str(self.wrench_time[10:]))
        #
        # print(j_times)
        # print(t_times)
        # print(w_times)
        # print("---------------------------------------------------")

        return max([t_times, j_times, w_times])

    def __repr__(self):
        rep = '-------------------------------------\n' + \
              'Data For Learning\n' + \
              '-------------------------------------\n' + \
              '\n' + \
              Fore.LIGHTBLUE_EX + 'timestamp: ' + Fore.YELLOW + str(self.timestamp()) + '\n' + Fore.RESET + \
              '\n' + \
              Fore.LIGHTBLUE_EX + 'joints_position: ' + Fore.YELLOW + str(self.joints_position) + '\n' + \
              '\n' + \
              Fore.LIGHTBLUE_EX + 'joints_effort: ' + Fore.YELLOW + str(self.joints_effort) + '\n' + \
              '\n' + \
              Fore.LIGHTBLUE_EX + 'wrench_pose: \n' + Fore.YELLOW + str(self.wrench_pose) + '\n' + \
              '\n' + \
              Fore.LIGHTBLUE_EX + 'wrench_force_torque: \n' + Fore.YELLOW + str(self.wrench_force_torque) + '\n' + \
              '\n' + \
              Fore.LIGHTBLUE_EX + 'gripper_current: ' + Fore.YELLOW + str(self.gripper_current) + '\n' + \
              '\n' + \
              Fore.LIGHTBLUE_EX + 'object_detection: ' + Fore.YELLOW + str(self.object_detection) + '\n' + \
              '\n' + Fore.RESET
        return rep



