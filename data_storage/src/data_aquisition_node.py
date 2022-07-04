#!/usr/bin/env python3

import rospy
import time
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Wrench, Pose, Transform, WrenchStamped
from tf2_msgs.msg import TFMessage
from colorama import Fore

joint_states_time = (0,0)
tf_time = (0,0)
wrench_time = (0,0)


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

        #TODO JOEL should create a proper time variable (seconds.nseconds), and assign the median of those 3 to self.timestamp
        self.timestamp = data.header.stamp.secs

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
              Fore.LIGHTBLUE_EX + 'wrench_pose: ' + Fore.YELLOW + str(self.wrench_pose) + '\n' + \
              '\n' + \
              Fore.LIGHTBLUE_EX + 'wrench_force_torque: ' + Fore.YELLOW + str(self.wrench_force_torque) + '\n' + \
              '\n' + \
              Fore.LIGHTBLUE_EX + 'gripper_current: ' + Fore.YELLOW + str(self.gripper_current) + '\n' + \
              '\n' + \
              Fore.LIGHTBLUE_EX + 'object_detection: ' + Fore.YELLOW + str(self.object_detection) + '\n' + \
              '\n' + Fore.RESET
        return rep


def time_stamps_comparison(joint_states_t, tf_t, wrench_t):
    print("joint_states_time")
    print(joint_states_t)
    print("tf_time")
    print(tf_t)
    print("wrench_time")
    print(wrench_t)
    print ('\n')


if __name__ == '__main__':

    rospy.init_node('data_aquisition_node', anonymous=True)

    # rate = rospy.Rate(0.55844257)  # 10hz
    rate = rospy.Rate(5000.0)  # 10hz

    data_for_learning = DataForLearning()

    # pub_force_trorque = rospy.Publisher('force_torque_values', ForceTorque, queue_size=1000)

    rospy.Subscriber("joint_states", JointState, data_for_learning.joint_states_callback)
    rospy.Subscriber("tf", TFMessage, data_for_learning.tf_callback)
    rospy.Subscriber("wrench", WrenchStamped, data_for_learning.wrench_callback)

    # TODO JOEL should initialize gripper communication and extract gripper information
    # TODO JOEL gripper information = "actual_force_motor_current" + "object_detected" (SEE gripper/src/force_torque_node.py)

    while not rospy.is_shutdown():
        time_stamps_comparison(joint_states_time, tf_time, wrench_time)
        print(data_for_learning)
        rate.sleep()


    # TODO JOEL should store every Data For Learning in a proper file (SQLite, JSON, txt, etc)
    rospy.spin()
