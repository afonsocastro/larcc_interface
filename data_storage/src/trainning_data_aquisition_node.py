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

joint_states_time = (0, 0)
tf_time = (0, 0)
wrench_time = (0, 0)


trainning_array = None
test_array = None

vector = None

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

    # nsecs = int((joint_states_t[1] + tf_t[1] + wrench_t[1]) / 3) # mean

    nsecs = max(joint_states_t[1], tf_t[1], wrench_t[1])
    secs = min(joint_states_t[0], tf_t[0], wrench_t[0])

    if first_read_exist:

        secs = secs - first_timestamp[0]

    elif not first_read_exist and secs > 0:

        first_timestamp = (secs, nsecs)

        nsecs = 0
        secs = 0

    else:
        secs = -1

    return float(str(secs) + "." + str(nsecs))


def add_to_test_array(test, category):
    row = [0, 0, 0]

    if category.find("pull") != -1:
        row = [1, 0, 0]
    elif category.find("push") != -1:
        row = [0, 1, 0]
    elif category.find("twist") != -1:
        row = [0, 0, 1]

    test = np.append(test, [row], axis=0)

    return test


# Adds a new line to the Nx28 array. This new line corresponds to a new variable state in a fixed time
def add_to_array(data):
    global array
    global first_read_exist
    global vector
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

    vector = np.append(vector, [row])


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


def save_trainnning_data(data, result, is_trainning):

    path = "./../data/trainning/"

    files = os.listdir(path)

    if is_trainning == 1:
        for file in files:
            if file.find("trainning_data.") != -1:
                prev_data_array = np.load(f"../data/trainning/trainning_data.npy")
                data = np.append(prev_data_array, data, axis=0)

            if file.find("trainning_data_results.") != -1:
                prev_result_array = np.load(f"../data/trainning/trainning_data_results.npy")
                result = np.append(prev_result_array, result, axis=0)

        np.save(f"../data/trainning/trainning_data.npy", data.astype('float32'))
        np.save(f"../data/trainning/trainning_data_results.npy", result.astype('float32'))

    else:
        for file in files:
            if file.find("test_data.") != -1:
                prev_data_array = np.load(f"../data/trainning/test_data.npy")
                data = np.append(prev_data_array, data, axis=0)

            if file.find("test_data_results.") != -1:
                prev_result_array = np.load(f"../data/trainning/test_data_results.npy")
                result = np.append(prev_result_array, result, axis=0)

        rng_state = np.random.get_state()
        np.random.shuffle(data)
        np.random.set_state(rng_state)
        np.random.shuffle(result)

        np.save(f"../data/trainning/test_data.npy", data.astype('float32'))
        np.save(f"../data/trainning/test_data_results.npy", result.astype('float32'))

    print("Trainning: " + str(data.shape))

    print("Test: " + str(result.shape))


if __name__ == '__main__':

    # ---------------------------------------------------------------------------------------------
    # -------------------------------------GET USER INPUTS-----------------------------------------
    # ---------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description="Just an example")
    parser.add_argument("-m", "--measurements", type=int, default=30,
                        help="Corresponds to the number of rows in the sample array.")
    parser.add_argument("-r", "--rate", type=float, default=10, help="How many collections per second will be performed.")
    parser.add_argument("-a", "--action", type=str, default="push", help="Classify the action performed in the "
                                                                         "experiment (ex: push, pull, twist).")
    parser.add_argument("-p", "--position", type=int, default=1, help="Index that identifies the experiments' positon.")
    parser.add_argument("-n", "--number_reps", type=int, default=1, help="Number of repetitions in the same execution")
    parser.add_argument("-it", "--is_trainning", type=int, default=1, help="1 - Perform trainning; 0 - Perform test")

    args = vars(parser.parse_args())

    vector = np.empty((0, args["measurements"] * 28), dtype=float)
    trainning_array = vector
    test_array = np.empty((0, 3), dtype=float)

    # ---------------------------------------------------------------------------------------------
    # ---------------------------------INITIATE COMMUNICATIONS-------------------------------------
    # ---------------------------------------------------------------------------------------------

    rospy.init_node('data_aquisition_node', anonymous=True)

    rate = rospy.Rate(args["rate"])  # 10hz

    data_for_learning = DataForLearning()

    rospy.Subscriber("joint_states", JointState, data_for_learning.joint_states_callback)
    rospy.Subscriber("tf", TFMessage, data_for_learning.tf_callback)
    rospy.Subscriber("wrench", WrenchStamped, data_for_learning.wrench_callback)

    pub_arm = rospy.Publisher('arm_request', String, queue_size=10)

    time.sleep(0.2)

    # ---------------------------------------------------------------------------------------------
    # ------------------------------PREPARE ROBOT FOR EXPERIMENT-----------------------------------
    # ---------------------------------------------------------------------------------------------

    ag.move_arm_to_initial_pose(pub_arm)
    print("Moving to testing position")

    time.sleep(3)

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

    limit = args["measurements"]

    # ---------------------------------------------------------------------------------------------
    # -------------------------------------GET DATA------------------------------------------------
    # ---------------------------------------------------------------------------------------------

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

                et = time.time()
                # print(et - st)

            rate.sleep()

        first_read_exist = False

        trainning_array = np.append(trainning_array, [vector], axis=0)
        test_array = add_to_test_array(test_array, args["action"])
        vector = np.empty((0, 0), dtype=float)

        print(f"Experiment {j + 1} data saved")

    hand.disconnect()


    #---------------------------------------------------------------------------------------------
    #-------------------------------------STORING DATA--------------------------------------------
    #---------------------------------------------------------------------------------------------

    action = args["action"]
    position = str(args["position"])

    out = input("Save experiment? (s/n)")

    if out == "s":
        print("Trainning saved!")
        save_trainnning_data(trainning_array, test_array, args["is_trainning"])
    else:
        print("Trainning not saved!")

    print(Fore.GREEN + "-----------------------------------------------------")
    print("             EXPERIMENT FINALIZED                              ")
    print("-----------------------------------------------------" + Fore.RESET)

