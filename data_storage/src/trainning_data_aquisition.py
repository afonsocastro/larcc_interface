#!/usr/bin/env python3
import argparse
from datetime import datetime
import json
import os
import time
from larcc_classes.data_storage.DataForLearning import DataForLearning
import numpy as np
import rospy
# from larcc_classes.src.ArmGripperComm import ArmGripperComm


def add_to_vector(data, vector, first_timestamp, dic_offset):

    if first_time_stamp is None:
        first_timestamp = data.timestamp()
        timestamp = 0.0
    else:
        timestamp = data.timestamp() - first_timestamp

    new_data = np.array([timestamp, data.joints_effort[0] - dic_offset["j0"],
                         data.joints_effort[1] - dic_offset["j1"],
                         data.joints_effort[2] - dic_offset["j2"],
                         data.joints_effort[3] - dic_offset["j3"],
                         data.joints_effort[4] - dic_offset["j4"],
                         data.joints_effort[5] - dic_offset["j5"],
                         data.wrench_force_torque.force.x - dic_offset["fx"],
                         data.wrench_force_torque.force.y - dic_offset["fy"],
                         data.wrench_force_torque.force.z - dic_offset["fz"],
                         data.wrench_force_torque.torque.x - dic_offset["mx"],
                         data.wrench_force_torque.torque.y - dic_offset["my"],
                         data.wrench_force_torque.torque.z - dic_offset["mz"]])

    return np.append(vector, new_data), first_timestamp


def calc_data_mean(data):
    values = np.array([data.wrench_force_torque.force.z/10, data.wrench_force_torque.torque.x,
                       data.wrench_force_torque.torque.y, data.wrench_force_torque.torque.z])

    return np.mean(abs(values))


def offset_calculation(dic):

    dic_offset_mean = {}

    for key in dic:

        dic_offset_mean[key] = np.mean(dic[key])

    return dic_offset_mean


def save_trainnning_data(data, categ, action_list):

    path = "../data/new_acquisition"

    files = os.listdir(path)

    vector_categ = np.ones((data.shape[0], 1)) * float(categ)

    data = np.append(data, vector_categ, axis=1)

    for file in files:
        if file.find("raw_learning_data.") != -1:
            prev_data_array = np.load(path + "/raw_learning_data.npy", allow_pickle=False)
            data = np.append(prev_data_array, data, axis=0)

    np.save(path + "/raw_learning_data.npy", data, allow_pickle=False)

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
        print(f"category {action_list[int(idx)]}: {idx_dic[idx]} experiments")


if __name__ == '__main__':

    # ---------------------------------------------------------------------------------------------
    # --------------------------------------INPUT VARIABLES----------------------------------------
    # ---------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description="Arguments for trainning script")
    parser.add_argument("-ag", "--activate_gripper", type=int, default=0,
                        help="1 - activates gripper; 0 - doesn't activate gripper (default)")
    parser.add_argument("-n", "--number_reps", type=int, default=1,
                        help="Corresponds to the number of times the experiment will be performed (default 1)")
    parser.add_argument("-c", "--category", type=int, default=-1,
                        help="Classifies the category (default no classification)")

    args = vars(parser.parse_args())

    f = open('../../../larcc_interface/data_storage/config/data_storage_config.json')

    config = json.load(f)

    f.close()

    if args["category"] >= 0:
        action = config["action_classes"][args["category"]]
        print(f"The action chosen was {action}")

    # ---------------------------------------------------------------------------------------------
    # -------------------------------INITIATE COMMUNICATION----------------------------------------
    # ---------------------------------------------------------------------------------------------

    rospy.init_node("training_data_aquisition", anonymous=True)

    data_for_learning = DataForLearning()
    # arm_gripper_comm = ArmGripperComm()

    rate = rospy.Rate(config["rate"])

    time.sleep(0.2) # Waiting time to ros nodes properly initiate

    # ---------------------------------------------------------------------------------------------
    # -------------------------------INITIATE ROBOT------------------------------------------------
    # ---------------------------------------------------------------------------------------------

    # try:
    #     joints = config["initial_pose"]
    #     arm_gripper_comm.move_arm_to_joints_state(joints[0], joints[1], joints[2], joints[3], joints[4], joints[5])
    #     if args["activate_gripper"] == 1:
    #         input("Press ENTER to activate gripper in 3 secs")
    #         for i in range(0, 3):
    #             print(i + 1)
    #             time.sleep(1)
    #
    #         arm_gripper_comm.gripper_init()
    #         time.sleep(1.5)
    #
    #         arm_gripper_comm.gripper_close_fast()
    #         time.sleep(0.5)
    #         arm_gripper_comm.gripper_disconnect()
    # except:
    #     print("ctrl+C pressed")

    list_gripper_calibration = []
    dic_offset_calibration = {"fx": [],
                              "fy": [],
                              "fz": [],
                              "mx": [],
                              "my": [],
                              "mz": [],
                              "j0": [],
                              "j1": [],
                              "j2": [],
                              "j3": [],
                              "j4": [],
                              "j5": []}
    print("CALIBRATING...")

    for i in range(0, 50):
        list_gripper_calibration.append(calc_data_mean(data_for_learning))
        dic_offset_calibration["fx"].append(data_for_learning.wrench_force_torque.force.x)
        dic_offset_calibration["fy"].append(data_for_learning.wrench_force_torque.force.y)
        dic_offset_calibration["fz"].append(data_for_learning.wrench_force_torque.force.z)
        dic_offset_calibration["mx"].append(data_for_learning.wrench_force_torque.torque.x)
        dic_offset_calibration["my"].append(data_for_learning.wrench_force_torque.torque.y)
        dic_offset_calibration["mz"].append(data_for_learning.wrench_force_torque.torque.z)

        dic_offset_calibration["j0"].append(data_for_learning.joints_effort[0])
        dic_offset_calibration["j1"].append(data_for_learning.joints_effort[1])
        dic_offset_calibration["j2"].append(data_for_learning.joints_effort[2])
        dic_offset_calibration["j3"].append(data_for_learning.joints_effort[3])
        dic_offset_calibration["j4"].append(data_for_learning.joints_effort[4])
        dic_offset_calibration["j5"].append(data_for_learning.joints_effort[5])
        rate.sleep()

    rest_state_mean = np.mean(np.array(list_gripper_calibration))
    dic_variable_offset = offset_calculation(dic_offset_calibration)

    limit = int(config["time"] * config["rate"])

    trainning_data_array = np.empty((0, limit * len(config["data"])))

    for j in range(0, args["number_reps"]):

        print(f"Waiting for action to initiate experiment {j + 1}...")

        while not rospy.is_shutdown():
            data_mean = calc_data_mean(data_for_learning)
            variance = data_mean - rest_state_mean
            # print(variance)
            if abs(variance) > config["force_threshold_start"]:
                break

            time.sleep(0.1)

        time.sleep(config["waiting_offset"]) # time waiting to initiate the experiment

        # ---------------------------------------------------------------------------------------------
        # -------------------------------------GET DATA------------------------------------------------
        # ---------------------------------------------------------------------------------------------

        end_experiment = False
        first_time_stamp = None
        vector_data = np.empty((0, 0))

        i = 0
        treshold_counter = 0

        rate.sleep()  # The first time rate sleep was used it was giving problems (would not wait the right amout of time)

        try:
            while not rospy.is_shutdown() and i < limit:

                i += 1
                print(data_for_learning)
                vector_data, first_time_stamp = add_to_vector(data_for_learning,
                                                              vector_data, first_time_stamp, dic_variable_offset)

                data_mean = calc_data_mean(data_for_learning)
                variance = data_mean - rest_state_mean

                # print(variance)
                if abs(variance) < config["force_threshold_end"]:
                    treshold_counter += 1
                    if treshold_counter >= config["threshold_counter_limit"]:
                        end_experiment = True
                        break
                else:
                    treshold_counter = 0

                rate.sleep()
        except:
            print("ctrl+C pressed")

        if end_experiment:
            print(f"Data collection interrupted in experiment {j + 1}")
            trainning_data_array = trainning_data_array[:-1, :]
            break

        print(vector_data.shape)
        # vector_norm = normalize_data(vector_data, limit)
        trainning_data_array = np.append(trainning_data_array, [vector_data], axis=0)


    if args["category"] == -1:
        category = input("Input the number that corresponds to the category trainned: ")
    else:
        category = args["category"]

    pos = f'time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}, action: {category}, ' \
          f'number_experiments: {trainning_data_array.shape[0]}, ' \
          f'joints_pos: {data_for_learning.joints_position}, gripper_pos: ' \
          f'({data_for_learning.wrench_pose.position.x}, ' \
          f'{data_for_learning.wrench_pose.position.y}, {data_for_learning.wrench_pose.position.z}, ' \
          f'{data_for_learning.wrench_pose.orientation.x}, {data_for_learning.wrench_pose.orientation.y}, ' \
          f'{data_for_learning.wrench_pose.orientation.z}, {data_for_learning.wrench_pose.orientation.w})'

    del data_for_learning
    # del data_for_learning, arm_gripper_comm

    # ---------------------------------------------------------------------------------------------
    # -------------------------------SAVE TRAINNING DATA-------------------------------------------
    # ---------------------------------------------------------------------------------------------

    classification = config["action_classes"][int(category)]
    out = input(f"You wish to save the {trainning_data_array.shape[0]} {classification} experiment? (s/n)\n")

    if out == "s":
        with open('../data/position_historic.txt', 'a') as f:
            f.write(pos + "\n")

        print(f"Trainning saved!\nIt was saved {trainning_data_array.shape[0]} experiments")
        save_trainnning_data(trainning_data_array, category, config["action_classes"])
    else:
        print("Trainning not saved!")

