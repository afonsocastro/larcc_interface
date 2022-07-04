#!/usr/bin/env python3
import datetime

import rospy
import time
import numpy as np
from datetime import date
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Wrench, Pose, WrenchStamped
from tf2_msgs.msg import TFMessage
from colorama import Fore
# from gripper.src.RobotiqHand import RobotiqHand


HOST = "192.168.56.2"
PORT = 54321

cont = True

joint_states_time = (0,0)
tf_time = (0,0)
wrench_time = (0,0)

array = np.empty((1, 28), dtype=float)

first_read_exist = False
first_timestamp = (0,0)

# TEST CONFIGURATION
test_time = 5 # segundos
collection_rate = 1 # Hertz


class DataForLearning:

    def __init__(self):
        self.timestamp = 0

        self.joints_position = []
        self.joints_effort = []

        self.wrench_pose = Pose()

        self.wrench_force_torque = Wrench()

        self.gripper_current = 0

        self.object_detection = 0

    def joint_states_callback(self, data):
        global joint_states_time

        self.joints_effort = data.effort
        self.joints_position = data.position

        joint_states_time = (data.header.stamp.secs, data.header.stamp.nsecs)

    def tf_callback(self, data):
        global tf_time

        self.wrench_pose.position.x = data.transforms[0].transform.translation.x
        self.wrench_pose.position.y = data.transforms[0].transform.translation.y
        self.wrench_pose.position.z = data.transforms[0].transform.translation.z
        self.wrench_pose.orientation = data.transforms[0].transform.rotation

        tf_time = (data.transforms[0].header.stamp.secs, data.transforms[0].header.stamp.nsecs)

    def wrench_callback(self, data):
        global wrench_time

        self.wrench_force_torque = data.wrench

        wrench_time = (data.header.stamp.secs, data.header.stamp.nsecs)

    def __repr__(self):
        rep = '-------------------------------------\n' + \
              'Data For Learning\n' + \
              '-------------------------------------\n' + \
              '\n' + \
              Fore.LIGHTBLUE_EX + 'timestamp: ' + Fore.YELLOW + str(self.timestamp) + '\n' + Fore.RESET + \
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


def time_stamps_comparison(joint_states_t, tf_t, wrench_t):
    global first_read_exist
    global first_timestamp

    # print("joint_states_time")
    # print(joint_states_t)
    # print("tf_time")
    # print(tf_t)
    # print("wrench_time")
    # print(wrench_t)
    # print('\n')

    # Choose the nsecs calulation

    nsecs = int((joint_states_t[1] + tf_t[1] + wrench_t[1]) / 3) # mean

    # nsecs = max(joint_states_t[1], tf_t[1], wrench_t[1])

    secs = max(joint_states_t[0], tf_t[0], wrench_t[0])

    if first_read_exist:
        secs = secs - first_timestamp[0]

    elif not first_read_exist and secs > 0:
        first_timestamp = (secs, nsecs)

        nsecs = 0
        secs = 0
        first_read_exist = True

    return float(str(secs) + "." + str(nsecs))


def add_to_array(data):
    global array

    row = np.array([data.timestamp, data.joints_position[0], data.joints_position[1], data.joints_position[2],
                    data.joints_position[3], data.joints_position[4], data.joints_position[5], data.joints_effort[0],
                    data.joints_effort[1], data.joints_effort[2], data.joints_effort[3], data.joints_effort[4],
                    data.joints_effort[5], data.wrench_pose.position.x, data.wrench_pose.position.y,
                    data.wrench_pose.position.z, data.wrench_pose.orientation.x, data.wrench_pose.orientation.y,
                    data.wrench_pose.orientation.z, data.wrench_pose.orientation.w, data.wrench_force_torque.force.x,
                    data.wrench_force_torque.force.y, data.wrench_force_torque.force.z, data.wrench_force_torque.torque.x,
                    data.wrench_force_torque.torque.y, data.wrench_force_torque.torque.z, data.gripper_current,
                    data.object_detection])

    array = np.append(array, [row], axis=0)


if __name__ == '__main__':

    rospy.init_node('data_aquisition_node', anonymous=True)

    # rate = rospy.Rate(0.55844257)  # 10hz
    rate = rospy.Rate(collection_rate)  # 10hz

    data_for_learning = DataForLearning()

    # pub_force_trorque = rospy.Publisher('force_torque_values', ForceTorque, queue_size=1000)

    rospy.Subscriber("joint_states", JointState, data_for_learning.joint_states_callback)
    rospy.Subscriber("tf", TFMessage, data_for_learning.tf_callback)
    rospy.Subscriber("wrench", WrenchStamped, data_for_learning.wrench_callback)

    # hand = RobotiqHand()
    # hand.connect(HOST, PORT)

    # TODO JOEL should initialize gripper communication and extract gripper information
    # TODO JOEL gripper information = "actual_force_motor_current" + "object_detected" (SEE gripper/src/force_torque_node.py)

    i = 0

    limit = test_time * collection_rate

    while (not rospy.is_shutdown()) and i < limit:
        # status = hand.get_instant_gripper_status()
        # data_for_learning.gripper_current = status.actual_force_motor_current
        # print(status.object_detected)

        data_for_learning.timestamp = time_stamps_comparison(joint_states_time, tf_time, wrench_time)

        st = time.time()

        try:
            if data_for_learning.timestamp > 0:
                add_to_array(data_for_learning)
                i += 1
                print(data_for_learning)
        except:
            pass

        et = time.time()
        print(et - st)
        rate.sleep()

    print("Adquiring data")
    np.save(f"../data/test{datetime.datetime.now()}.npy", array)
    print("Data aquired")

    rospy.spin()
