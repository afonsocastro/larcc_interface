#!/usr/bin/env python3
import argparse
import json
import os
import time

from colorama import Fore
from tensorflow import keras

from data_storage.src.data_aquisition_node import DataForLearning
import numpy as np
import rospy
from sklearn.preprocessing import normalize
from lib.src.ArmGripperComm import ArmGripperComm
from tabulate import tabulate
import pyfiglet


def print_tabulate(real_time_predictions, config):

    labels = config["action_classes"]
    max_idx = np.argmax(real_time_predictions)

    result = pyfiglet.figlet_format(labels[max_idx], font="space_op", width=500)

    print(Fore.LIGHTBLUE_EX + result + Fore.RESET)

    for pred in list(real_time_predictions):
        data = [['Output', pred[0], pred[1], pred[2], pred[3]]]
        print(tabulate(list(data), headers=[" ", "PULL", "PUSH", "SHAKE", "TWIST"], tablefmt="fancy_grid"))
        print("\n")


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
    # values = np.array([data.wrench_force_torque.force.x,
    #                    data.wrench_force_torque.force.y, data.wrench_force_torque.force.z])
    # values = np.array([data.wrench_force_torque.force.x,
    #                    data.wrench_force_torque.force.y, data.wrench_force_torque.force.z, data.wrench_force_torque.torque.x,
    #                    data.wrench_force_torque.torque.y, data.wrench_force_torque.torque.z])
    values = np.array([data.wrench_force_torque.force.z/10, data.wrench_force_torque.torque.x,
                       data.wrench_force_torque.torque.y, data.wrench_force_torque.torque.z])
    # # values = np.array([data.joints_effort[0], data.joints_effort[1], data.joints_effort[2], data.joints_effort[3],
    #                    data.joints_effort[4], data.joints_effort[5], data.wrench_force_torque.force.x,
    #                    data.wrench_force_torque.force.y, data.wrench_force_torque.force.z, data.wrench_force_torque.torque.x,
    #                    data.wrench_force_torque.torque.y, data.wrench_force_torque.torque.z])

    return np.mean(values)


if __name__ == '__main__':

    # ---------------------------------------------------------------------------------------------
    # --------------------------------------INPUT VARIABLES----------------------------------------
    # ---------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description="Arguments for trainning script")
    parser.add_argument("-ag", "--activate_gripper", type=int, default=0,
                        help="1 - activates gripper; 0 - doesn't activate gripper (default)")

    args = vars(parser.parse_args())

    f = open('../config/config.json')

    config = json.load(f)

    f.close()

    model_path = "../../neural_networks/keras"

    model = keras.models.load_model(model_path + "/myModel")

    # ---------------------------------------------------------------------------------------------
    # -------------------------------INITIATE COMMUNICATION----------------------------------------
    # ---------------------------------------------------------------------------------------------

    rospy.init_node("action_intreperter", anonymous=True)

    data_for_learning = DataForLearning()
    arm_gripper_comm = ArmGripperComm()

    rate = rospy.Rate(config["rate"])

    time.sleep(0.2) # Waiting time to ros nodes properly initiate

    # ---------------------------------------------------------------------------------------------
    # -------------------------------INITIATE ROBOT------------------------------------------------
    # ---------------------------------------------------------------------------------------------

    try:
        # arm_gripper_comm.move_arm_to_initial_pose()

        if args["activate_gripper"] == 1:
            input("Press ENTER to activate gripper in 3 secs")
            for i in range(0, 3):
                print(i + 1)
                time.sleep(1)

            arm_gripper_comm.gripper_init()
            time.sleep(1.5)

            arm_gripper_comm.gripper_close_fast()
            time.sleep(0.5)

            arm_gripper_comm.gripper_disconnect()
    except:
        print("ctrl+C pressed")

    list_calibration = []

    print("Calculating rest state variables...")

    for i in range(0, 99):
        list_calibration.append(calc_data_mean(data_for_learning))
        time.sleep(0.01)

    rest_state_mean = np.mean(np.array(list_calibration))

    limit = int(config["time"] * config["rate"])

    trainning_data_array = np.empty((0, limit * config["n_variables"]))

    sequential_actions = False

    while not rospy.is_shutdown():

        if not sequential_actions:
            print(f"Waiting for action to initiate prediction ...")

        while not rospy.is_shutdown():
            data_mean = calc_data_mean(data_for_learning)
            variance = data_mean - rest_state_mean

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
                # print(data_for_learning)
                vector_data, first_time_stamp = add_to_vector(data_for_learning, vector_data, first_time_stamp)

                data_mean = calc_data_mean(data_for_learning)
                variance = data_mean - rest_state_mean

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
            sequential_actions = False
            print("\nNot enough for prediction\n")
        else:
            sequential_actions = True
            vector_norm = normalize_data(vector_data, limit)
            print("-----------------------------------------------------------")
            predictions = model.predict(x=vector_norm, verbose=2)
            print_tabulate(list(predictions), config)
            print("-----------------------------------------------------------")

    del data_for_learning, arm_gripper_comm

