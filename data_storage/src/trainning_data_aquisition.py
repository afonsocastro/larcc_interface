#!/usr/bin/env python3
import argparse
import os
import time
from data_storage.src.data_aquisition_node import DataForLearning
import numpy as np
import rospy
from sklearn.preprocessing import normalize
from lib.src.ArmGripperComm import ArmGripperComm


def normalize_data(vector, measurements):

    data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
    experiment_array_norm = normalize(data_array, axis=0, norm='max')

    vector_data_norm = np.reshape(experiment_array_norm, (1, vector.shape[0]))
    return vector_data_norm


def add_to_vector(data, vector, first_timestamp):

    if first_time_stamp is None:
        first_timestamp = data.timestamp()
        timestamp = 0.0
    else:
        timestamp = data.timestamp() - first_timestamp

    new_data = np.array([timestamp, data.joints_effort[0], data.joints_effort[1], data.joints_effort[2],
                         data.joints_effort[3], data.joints_effort[4], data.joints_effort[5],
                         data.wrench_force_torque.force.x, data.wrench_force_torque.force.y,
                         data.wrench_force_torque.force.z, data.wrench_force_torque.torque.x,
                         data.wrench_force_torque.torque.y, data.wrench_force_torque.torque.z])

    return np.append(vector, new_data), first_timestamp


def calc_data_mean(data):
    values = np.array([data.joints_effort[0], data.joints_effort[1], data.joints_effort[2], data.joints_effort[3],
                       data.joints_effort[4], data.joints_effort[5], data.wrench_force_torque.force.x,
                       data.wrench_force_torque.force.y, data.wrench_force_torque.force.z, data.wrench_force_torque.torque.x,
                       data.wrench_force_torque.torque.y, data.wrench_force_torque.torque.z])

    return np.mean(values)


def save_trainnning_data(data, categ):

    path = "./../data/trainning2"

    files = os.listdir(path)

    vector_categ = np.ones((data.shape[0], 1)) * float(categ)

    data = np.append(data, vector_categ, axis=1)

    for file in files:
        if file.find("learning_data.") != -1:
            prev_data_array = np.load(path + "/learning_data.npy")
            data = np.append(prev_data_array, data, axis=0)

    np.save(path + "/learning_data.npy", data.astype('float32'))

    idx_dic = {}
    idx_list = []
    for line in data:
        key = str(int(line[-1]))
        if not str(key) in idx_dic:
            idx_dic[key] = 1
            idx_list.append(key)
        else:
            idx_dic[key] += 1

    idx_list.sort()

    print(f"There are now {data.shape[0]} experiments in total")

    for idx in idx_list:
        print(f"category {idx}: {idx_dic[idx]} experiments")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Arguments for trainning script")
    parser.add_argument("-ag", "--activate_gripper", type=int, default=0,
                        help="1 - activates gripper; 0 - doesn't activate gripper (default)")
    parser.add_argument("-r", "--rate", type=float, default=100,
                        help="Defines how many measurements are performed per second. (default 100 Hz)")
    parser.add_argument("-t", "--time", type=int, default=0.5,
                        help="Corresponds to the time in seconds of each experiment. (default 0.5 seconds)")
    parser.add_argument("-n", "--number_reps", type=int, default=1,
                        help="Corresponds to the number of times the experiment will be performed (default 1)")
    parser.add_argument("-c", "--category", type=int, default=-1,
                        help="Classifies the category (default no classification)")
    parser.add_argument("-ft", "--force_treshold", type=float, default=0.1,
                        help="Defines the force treshold to stop the experiment (default 0.1). The threshold to start"
                             "is 0.2")

    args = vars(parser.parse_args())

    # ---------------------------------------------------------------------------------------------
    # -------------------------------INITIATE COMMUNICATION------------------------------------------------
    # ---------------------------------------------------------------------------------------------

    rospy.init_node("tranning_data_aquisition", anonymous=True)

    data_for_learning = DataForLearning()
    arm_gripper_comm = ArmGripperComm()

    rate = rospy.Rate(args["rate"])

    time.sleep(0.2) # Waiting time to ros nodes properly initiate

    # ---------------------------------------------------------------------------------------------
    # -------------------------------INITIATE ROBOT------------------------------------------------
    # ---------------------------------------------------------------------------------------------

    arm_gripper_comm.move_arm_to_initial_pose()

    if args["activate_gripper"] == 1:
        input("Press ENTER to activate gripper in 3 secs")
        time.sleep(3)
        arm_gripper_comm.gripper_init()
        time.sleep(1.5)

    arm_gripper_comm.gripper_close_fast()

    list_calibration = []

    print("Calculating rest state variables...")

    for i in range(0, 99):
        list_calibration.append(calc_data_mean(data_for_learning))
        time.sleep(0.01)

    rest_state_mean = np.mean(np.array(list_calibration))

    limit = int(args["time"] * args["rate"])

    trainning_data_array = np.empty((0, limit * 13)) # 13 is the number of variables being stored

    for j in range(0, args["number_reps"]):

        print(f"Waiting for action to initiate experiment {j + 1}...")

        while not rospy.is_shutdown():
            data_mean = calc_data_mean(data_for_learning)
            variance = data_mean - rest_state_mean
            print(variance)
            if abs(variance) > 0.15:
                break

            time.sleep(0.1)

        time.sleep(0.2) # time waiting to initiate the experiment

        # ---------------------------------------------------------------------------------------------
        # -------------------------------------GET DATA------------------------------------------------
        # ---------------------------------------------------------------------------------------------

        end_experiment = False
        first_time_stamp = None
        vector_data = np.empty((0, 0))

        i = 0
        treshold_counter = 0

        rate.sleep()  # The first time rate sleep was used it was giving problems (would not wait the right amout of time)
        while not rospy.is_shutdown() and i < limit:

            i += 1
            # print(data_for_learning)
            vector_data, first_time_stamp = add_to_vector(data_for_learning, vector_data, first_time_stamp)

            data_mean = calc_data_mean(data_for_learning)
            variance = data_mean - rest_state_mean

            print(variance)
            if abs(variance) < args["force_treshold"]:
                treshold_counter += 1
                if treshold_counter >= 3:
                    end_experiment = True
                    break
            else:
                treshold_counter = 0

            rate.sleep()

        if end_experiment:
            print(f"Data collection interrupted in experiment {j + 1}")
            trainning_data_array = trainning_data_array[:-1, :]
            break

        vector_norm = normalize_data(vector_data, limit)
        trainning_data_array = np.append(trainning_data_array, vector_norm, axis=0)

    del data_for_learning, arm_gripper_comm

    # ---------------------------------------------------------------------------------------------
    # -------------------------------SAVE TRAINNING DATA-------------------------------------------
    # ---------------------------------------------------------------------------------------------

    if args["category"] == -1:
        category = input("Input the number that corresponds to the category trainned: ")
    else:
        category = args["category"]

    out = input(f"You wish to save the {trainning_data_array.shape[0]} experiment? (s/n)\n")

    if out == "s":
        print(f"Trainning saved!\nIt was saved {trainning_data_array.shape[0]} experiments")
        save_trainnning_data(trainning_data_array, category)
    else:
        print("Trainning not saved!")

