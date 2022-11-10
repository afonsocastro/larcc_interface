#!/usr/bin/env python3
import argparse
import json
import math
import time

from colorama import Fore
from matplotlib import pyplot as plt
from tensorflow import keras
from std_msgs.msg import String, Float64MultiArray, Float64, Bool
from larcc_classes.data_storage.DataForLearning import DataForLearning
import numpy as np
import rospy

from tabulate import tabulate
import pyfiglet

from config.definitions import ROOT_DIR
from config.definitions import NN_DIR


def print_tabulate(label, real_time_predictions):

    result = pyfiglet.figlet_format(label, font="space_op", width=500)

    print(Fore.LIGHTBLUE_EX + result + Fore.RESET)

    for pred in list(real_time_predictions):
        data = [['Output', pred[0], pred[1], pred[2], pred[3]]]
        print(tabulate(list(data), headers=[" ", "PULL", "PUSH", "SHAKE", "TWIST"], tablefmt="fancy_grid"))
        print("\n")


def normalize_data(vector, measurements, train_config):

    data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
    data_array_norm = np.empty((data_array.shape[0], 0))

    idx = 0
    for n in train_config["normalization_clusters"]:
        data_sub_array = data_array[:, idx:idx + n]
        idx += n

        data_max = abs(max(data_sub_array.min(), data_sub_array.max(), key=abs))

        data_sub_array_norm = data_sub_array / data_max
        data_array_norm = np.hstack((data_array_norm, data_sub_array_norm))

    vector_data_norm = np.reshape(data_array_norm, (1, vector.shape[0]))

    # data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
    # experiment_array_norm = normalize(data_array, axis=0, norm='max')
    #
    # vector_data_norm = np.reshape(experiment_array_norm, (1, vector.shape[0]))

    return vector_data_norm


# def add_to_vector(data, vector, first_timestamp, list_idx):
def add_to_vector(data, vector, func_first_timestamp, dic_offset,pub_data):

    msg = Float64MultiArray()

    if func_first_timestamp is None:
        func_first_timestamp = data.timestamp()
        timestamp = 0.0
    else:
        timestamp = data.timestamp() - func_first_timestamp
    #
    # new_data = [timestamp, data.joints_effort[0], data.joints_effort[1], data.joints_effort[2],
    #             data.joints_effort[3], data.joints_effort[4], data.joints_effort[5],
    #             data.wrench_force_torque.force.x, data.wrench_force_torque.force.y,
    #             data.wrench_force_torque.force.z, data.wrench_force_torque.torque.x,
    #             data.wrench_force_torque.torque.y, data.wrench_force_torque.torque.z]

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
                         data.wrench_force_torque.torque.z - - dic_offset["mz"]])

    # dim = []
    # msg.layout.data_offset = 0
    # dim.append(MultiArrayDimension("line", 1, 13))
    # msg.layout.dim = dim
    msg.data = new_data
    pub_data.publish(msg)

    return np.append(vector, new_data), func_first_timestamp


def calc_data_mean(data, pub):
    values = np.array([data.wrench_force_torque.force.z/10, data.wrench_force_torque.torque.x,
                       data.wrench_force_torque.torque.y, data.wrench_force_torque.torque.z])

    mean_value = np.mean(values)

    pub.publish(mean_value)

    return mean_value


def get_statistics(data_list):
    data_list_mean = np.mean(np.array(data_list))

    summ = 0
    for x in data_list:
        summ += (x-data_list_mean)**2

    data_list_var = math.sqrt(summ/len(data_list))
    return data_list_mean, data_list_var


def offset_calculation(dic):

    dic_offset_mean = {}

    for key in dic:

        dic_offset_mean[key] = np.mean(dic[key])

    return dic_offset_mean


if __name__ == '__main__':

    # ---------------------------------------------------------------------------------------------
    # --------------------------------------INPUT VARIABLES----------------------------------------
    # ---------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description="Arguments for trainning script")
    parser.add_argument("-c", "--config_file", type=str, default="data_storage_config",
                        help="If argmument is present, activates gripper")

    args = vars(parser.parse_args())

    f = open(ROOT_DIR + '/data_storage/config/' + args["config_file"] + '.json')

    storage_config = json.load(f)

    f.close()

    f = open(ROOT_DIR + '/data_storage/config/training_config.json')

    trainning_config = json.load(f)

    f.close()

    model = keras.models.load_model(NN_DIR + "/feedforward/myModel")

    # ---------------------------------------------------------------------------------------------
    # -------------------------------INITIATE COMMUNICATION----------------------------------------
    # ---------------------------------------------------------------------------------------------

    rospy.init_node("action_intreperter", anonymous=True)

    # For force/torque GUI
    pub_vector = rospy.Publisher("learning_data", Float64MultiArray, queue_size=10)
    pub_class = rospy.Publisher("classification", String, queue_size=10)

    # For trigger GUI
    pub_trigger = rospy.Publisher("trigger_data", Float64, queue_size=10)
    pub_calibration = rospy.Publisher("calibration", Float64, queue_size=10)
    pub_force_detection = rospy.Publisher("force_detection", Bool, queue_size=10)

    data_for_learning = DataForLearning()
    # arm_gripper_comm = ArmGripperComm()

    rate = rospy.Rate(storage_config["rate"])

    time.sleep(0.2) # Waiting time to ros nodes properly initiate

    # ---------------------------------------------------------------------------------------------
    # -------------------------------INITIATE ROBOT------------------------------------------------
    # ---------------------------------------------------------------------------------------------
    #
    # try:
    #     if args["move_arm_to_inicial_position"]:
    #         arm_gripper_comm.move_arm_to_initial_pose()
    #
    #     if args["activate_gripper"]:
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
    #
    #         arm_gripper_comm.gripper_disconnect()
    # except:
    #     print("ctrl+C pressed")

    list_calibration = []
    dic_offset_calibration = {"fx": [], "fy": [], "fz": [], "mx": [],
                              "my": [], "mz": [], "j0": [], "j1": [],
                              "j2": [], "j3": [], "j4": [], "j5": []}
    dic_variable_offset = None

    # print("Calculating rest state variables...")
    #
    # for i in range(0, 99):
    #     list_calibration.append(calc_data_mean(data_for_learning))
    #     time.sleep(0.005)
    #
    # rest_state_mean = np.mean(np.array(list_calibration))

    limit = int(storage_config["time"] * storage_config["rate"])

    trainning_data_array = np.empty((0, limit * len(storage_config["data"])))

    sequential_actions = False
    first_time_stamp_show = None
    vector_data_show = np.empty((0, 0))
    rest_state_mean = 0
    pub_force_detection.publish(False)
    predicted_data_saved = np.empty((0, 651))
    predictions_saved = np.empty((0, 4))

    while not rospy.is_shutdown(): # This is the data acquisition cycle

        if not sequential_actions:
            st = time.time()
            while not rospy.is_shutdown(): # This is the calibration cycle
                print("Calculating rest state variables...")
                list_calibration = []
                dic_offset_calibration = {"fx": [], "fy": [], "fz": [], "mx": [],
                                          "my": [], "mz": [], "j0": [], "j1": [],
                                          "j2": [], "j3": [], "j4": [], "j5": []}

                pub_class.publish("Calibrating")

                for i in range(0, 49):
                    list_calibration.append(calc_data_mean(data_for_learning, pub_trigger))
                    if dic_variable_offset is not None:
                        add_to_vector(data_for_learning, vector_data_show, None, dic_variable_offset, pub_vector)

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

                    time.sleep(0.005)

                pub_class.publish("None")
                rest_state_mean, rest_state_var = get_statistics(list_calibration)
                dic_variable_offset = offset_calculation(dic_offset_calibration)
                pub_calibration.publish(rest_state_mean)
                print(rest_state_mean)
                print(rest_state_var)

                if rest_state_var < 0.03:
                    break
            print("Calibration time: " + str(time.time() - st))
            print(f"Waiting for action to initiate prediction ...")

            while not rospy.is_shutdown(): # This cycle waits for the external force to start storing data
                data_mean = calc_data_mean(data_for_learning, pub_trigger)
                variance = data_mean - rest_state_mean

                add_to_vector(data_for_learning, vector_data_show, None, dic_variable_offset, pub_vector)

                pub_calibration.publish(rest_state_mean)

                if abs(variance) > trainning_config["force_threshold_start"]:
                    pub_force_detection.publish(True)
                    break

                time.sleep(0.1)

            time.sleep(storage_config["waiting_offset"]) # time waiting to initiate the experiment

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
            while not rospy.is_shutdown() and i < limit: # This cycle stores data for a fixed amount of time
                pub_calibration.publish(rest_state_mean)
                i += 1
                # print(data_for_learning)
                vector_data, first_time_stamp = add_to_vector(data_for_learning,
                                                              vector_data, first_time_stamp, dic_variable_offset, pub_vector)
                # vector_data, first_time_stamp = add_to_vector(data_for_learning, vector_data, first_time_stamp,
                #                                               list_filter_idx)

                data_mean = calc_data_mean(data_for_learning, pub_trigger)
                variance = data_mean - rest_state_mean

                if abs(variance) < trainning_config["force_threshold_end"]:
                    treshold_counter += 1
                    if treshold_counter >= trainning_config["threshold_counter_limit"]:
                        end_experiment = True
                        pub_force_detection.publish(False)
                        break
                else:
                    treshold_counter = 0

                rate.sleep()
        except:
            print("ctrl+C pressed")

        try:
            if end_experiment:
                sequential_actions = False
                print("\nNot enough for prediction\n")
                pub_class.publish("None")
            else:
                sequential_actions = True
                # print(vector_data)
                vector_norm = normalize_data(vector_data, limit, trainning_config)

                predictions = model.predict(x=vector_norm, verbose=2)

                labels = storage_config["action_classes"]
                max_idx = np.argmax(list(predictions))
                print(max_idx)
                print(predictions[0][int(max_idx)])
                print(predictions)
                predicted_label = labels[int(max_idx)]

                vector_data = np.append(vector_data, max_idx)
                predicted_data_saved = np.append(predicted_data_saved, [vector_data], axis=0)
                predictions_saved = np.append(predictions_saved, predictions, axis=0)
                # print(predicted_data_saved.shape)
                # print(predictions_saved.shape)

                pub_class.publish(predicted_label + " " + str(round(float(predictions[0][int(max_idx)] * 100), 2)) + "%")

                print("-----------------------------------------------------------")
                print_tabulate(predicted_label, predictions)
                print("-----------------------------------------------------------")
        except:
            print("ctrl+C pressed")

    data_save_dic = {"data_predicted": predicted_data_saved.tolist(),
                     "predictions_confidence": predictions_saved.tolist()}
    json_object = json.dumps(data_save_dic)
    with open(ROOT_DIR + "/data_storage/data/predicted_learning_data/multi_class_sample.json", "w") as outfile:
        outfile.write(json_object)

    del data_for_learning
