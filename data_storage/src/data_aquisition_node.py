#!/usr/bin/env python3

import rospy
import time
import numpy as np
import os
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Wrench, Pose, WrenchStamped
from tf2_msgs.msg import TFMessage
from colorama import Fore
import argparse
from gripper.src.RobotiqHand import RobotiqHand

cont = True

joint_states_time = (0, 0)
tf_time = (0, 0)
wrench_time = (0, 0)

array = np.empty((1, 28), dtype=float)

first_read_exist = False
first_timestamp = (0, 0)


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


# Chooses the timestamp using the three different timestamps obtained
def time_stamps_comparison(joint_states_t, tf_t, wrench_t):
    global first_read_exist
    global first_timestamp

    nsecs = int((joint_states_t[1] + tf_t[1] + wrench_t[1]) / 3) # mean

    secs = max(joint_states_t[0], tf_t[0], wrench_t[0])

    if first_read_exist:

        secs = secs - first_timestamp[0]

    elif not first_read_exist and secs > 0:

        first_timestamp = (secs, nsecs)

        nsecs = 0
        secs = 0

    else:
        secs = -1

    return float(str(secs) + "." + str(nsecs))


# Adds a new line to the Nx28 array. This new line corresponds to a new variable state in a fixed time
def add_to_array(data):
    global array
    global first_read_exist
    first_read_exist = True

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


def gripper_init(hand_robot):
    print('activate: start')
    hand_robot.reset()
    hand_robot.activate()
    result = hand_robot.wait_activate_complete()
    print('activate: result = 0x{:02x}'.format(result))
    if result != 0x31:
        hand_robot.disconnect()
    print('adjust: start')
    hand_robot.adjust()
    print('adjust: finish')

    return hand_robot


def calc_data_mean(data):
    values = np.array([data.joints_effort[0], data.joints_effort[1], data.joints_effort[2], data.joints_effort[3],
                       data.joints_effort[4], data.joints_effort[5], data.wrench_force_torque.force.x,
                       data.wrench_force_torque.force.y, data.wrench_force_torque.force.z, data.wrench_force_torque.torque.x,
                       data.wrench_force_torque.torque.y, data.wrench_force_torque.torque.z])

    return np.mean(values)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Just an example")
    parser.add_argument("-t", "--time", type=float, default=5, help="Desired duration of data collection in seconds.")
    parser.add_argument("-r", "--rate", type=float, default=10, help="How many collections per second will be performed.")
    parser.add_argument("-a", "--action", type=str, default="push", help="Classify the action performed in the "
                                                                         "experiment (ex: push, pull, twist).")
    parser.add_argument("-p", "--position", type=int, default=1, help="Index that identifies the experiments' positon.")
    parser.add_argument("-n", "--number_reps", type=int, default=1, help="Number of repetitions in the same execution")

    args = vars(parser.parse_args())

    rospy.init_node('data_aquisition_node', anonymous=True)

    rate = rospy.Rate(args["rate"])  # 10hz

    data_for_learning = DataForLearning()

    rospy.Subscriber("joint_states", JointState, data_for_learning.joint_states_callback)
    rospy.Subscriber("tf", TFMessage, data_for_learning.tf_callback)
    rospy.Subscriber("wrench", WrenchStamped, data_for_learning.wrench_callback)

    HOST = "192.168.56.2"
    PORT = 54321

    hand = RobotiqHand()

    hand.connect(HOST, PORT)

    status = hand.get_instant_gripper_status()

    print("Checking if there is object...")
    time.sleep(2)

    if not status.object_detected:
        print("No object found")
        time.sleep(0.5)

        print("Initiate the gripper")
        hand = gripper_init(hand)
        time.sleep(2)

        hand.move(255, 0, 1)
        print("Closing gripper")
        time.sleep(5)
    else:
        print("Object detected")
        time.sleep(2)

    list_calibration = []

    print("Calculating rest state variables...")

    for i in range(0, 99):
        list_calibration.append(calc_data_mean(data_for_learning))
        time.sleep(0.01)

    rest_state_mean = np.mean(np.array(list_calibration))

    limit = float(args["time"]) * float(args["rate"])

    for j in range(0, args["number_reps"]):

        print(f"Waiting for action to initiate experiment {j + 1}...")

        while not rospy.is_shutdown():
            data_mean = calc_data_mean(data_for_learning)
            variance = data_mean - rest_state_mean

            if abs(variance) > 0.3:
                break

            time.sleep(0.1)

        i = 0

        while (not rospy.is_shutdown()) and i < limit:

            # The aquisition of the gripper status is done before checking the timestamp so the difference between the
            # timestamp and the actual time of measurement be residual
            # With this script order, there is a measurement error of about 10^-7 seconds

            st = time.time()

            status = hand.get_instant_gripper_status()
            data_for_learning.gripper_current = status.actual_force_motor_current

            if status.object_detected:
                data_for_learning.object_detection = 1
            else:
                data_for_learning.object_detection = 0

            data_for_learning.timestamp = time_stamps_comparison(joint_states_time, tf_time, wrench_time)

            if data_for_learning.timestamp >= 0:

                try:
                    add_to_array(data_for_learning)
                    i += 1
                    print(data_for_learning)
                except:
                    pass

                data_mean = calc_data_mean(data_for_learning)
                variance = data_mean - rest_state_mean

                print(variance)
                if abs(variance) < 0.1:
                    break

                et = time.time()
                # print(et - st)

            rate.sleep()

        first_read_exist = False
        action = args["action"]
        position = str(args["position"])

        path = f"./../data/{action}"

        files = os.listdir(path)

        count = 0

        for file in files:
            if file.find(f"pos{position}") != -1:
                count += 1

        np.save(f"../data/{action}/pos{position}_sample{count + 1}.npy", array)
        print(f"Experiment {j + 1} data saved in " + Fore.GREEN + f"pos{position}_sample{count + 1}.npy " + Fore.RESET + "file")

    hand.disconnect()

    print(Fore.GREEN + "-----------------------------------------------------")
    print("             EXPERIMENT FINALIZED                              ")
    print("-----------------------------------------------------" + Fore.RESET)

    rospy.spin()
