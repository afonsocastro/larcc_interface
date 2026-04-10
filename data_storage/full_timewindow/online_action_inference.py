#!/usr/bin/env python3
import argparse
import json
import math
import time
from colorama import Fore
from matplotlib import pyplot as plt
from tensorflow import keras
from keras_nlp.layers import SinePositionEncoding, TransformerEncoder
from std_msgs.msg import String, Float64MultiArray, Float64, Bool
from larcc_classes.data_storage.DataForLearning import DataForLearning
from larcc_classes.arm.UR10eArm import UR10eArm
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

def normalize_data(vector, measurements, train_config, clusters_max_min):

    data_max_timestamp = abs(max(clusters_max_min["timestamp"]["max"], clusters_max_min["timestamp"]["min"], key=abs))
    data_max_joints = abs(max(clusters_max_min["joints"]["max"], clusters_max_min["joints"]["min"], key=abs))
    data_max_gripper_F = abs(max(clusters_max_min["gripper_F"]["max"], clusters_max_min["gripper_F"]["min"], key=abs))
    data_max_gripper_M = abs(max(clusters_max_min["gripper_M"]["max"], clusters_max_min["gripper_M"]["min"], key=abs))

    data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
    data_array_norm = np.empty((data_array.shape[0], 0))

    idx = 0
    for n in train_config["normalization_clusters"]:
        data_sub_array = data_array[:, idx:idx + n]

        if idx == 0:
            data_sub_array_norm = data_sub_array / data_max_timestamp
        elif idx == 1:
            data_sub_array_norm = data_sub_array / data_max_joints
        elif idx == 7:
            data_sub_array_norm = data_sub_array / data_max_gripper_F
        elif idx == 10:
            data_sub_array_norm = data_sub_array / data_max_gripper_M

        idx += n
        data_array_norm = np.hstack((data_array_norm, data_sub_array_norm))

    vector_data_norm = np.reshape(data_array_norm, (1, vector.shape[0]))

    return vector_data_norm

def add_to_vector(data, vector, func_first_timestamp, dic_offset,pub_data):
    msg = Float64MultiArray()
    if func_first_timestamp is None:
        func_first_timestamp = data.timestamp()
        timestamp = 0.0
    else:
        timestamp = data.timestamp() - func_first_timestamp

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

    f = open(ROOT_DIR + '/data_storage/config/data_storage_config.json')
    storage_config = json.load(f)
    f.close()

    f = open(ROOT_DIR + '/data_storage/config/training_config.json')
    trainning_config = json.load(f)
    f.close()

    f = open(ROOT_DIR + '/data_storage/src/clusters_max_min.json')
    clusters_max_min = json.load(f)
    f.close()

    model_cnn = keras.models.load_model(ROOT_DIR + "/data_storage/models/cnn_v1_1.keras")
    model_transformer = keras.models.load_model(ROOT_DIR + "/data_storage/models/transformer_v1_1.keras",
        custom_objects={"SinePositionEncoding": SinePositionEncoding, "TransformerEncoder": TransformerEncoder},compile=False)

    # ---------------------------------------------------------------------------------------------
    # -------------------------------INITIATE COMMUNICATION----------------------------------------
    # ---------------------------------------------------------------------------------------------

    rospy.init_node("online_action_inference", anonymous=True)

    # For force/torque GUI
    pub_vector = rospy.Publisher("learning_data", Float64MultiArray, queue_size=10)
    pub_class_cnn = rospy.Publisher("cnn_classification", String, queue_size=10)
    pub_class_transformer = rospy.Publisher("transformer_classification", String, queue_size=10)

    # For trigger GUI
    pub_trigger = rospy.Publisher("trigger_data", Float64, queue_size=10)
    pub_calibration = rospy.Publisher("calibration", Float64, queue_size=10)
    pub_force_detection = rospy.Publisher("force_detection", Bool, queue_size=10)

    data_for_learning = DataForLearning()
    rate = rospy.Rate(storage_config["rate"])

    time.sleep(0.2) # Waiting time to ros nodes properly initiate

    # ---------------------------------------------------------------------------------------------
    # -------------------------------INITIATE ROBOT------------------------------------------------
    # ---------------------------------------------------------------------------------------------

    arm = UR10eArm()
    state = False
    while not state:
        state = arm.go_to_joint_state(storage_config["initial_pose"][0], storage_config["initial_pose"][1],
                                      storage_config["initial_pose"][2], storage_config["initial_pose"][3],
                                      storage_config["initial_pose"][4], storage_config["initial_pose"][5], 1, 1)
        time.sleep(0.1)

    list_calibration = []
    dic_offset_calibration = {"fx": [], "fy": [], "fz": [], "mx": [],
                              "my": [], "mz": [], "j0": [], "j1": [],
                              "j2": [], "j3": [], "j4": [], "j5": []}
    dic_variable_offset = None

    limit = int(storage_config["time"] * storage_config["rate"])

    trainning_data_array = np.empty((0, limit * len(storage_config["data"])))

    sequential_actions = False
    first_time_stamp_show = None
    vector_data_show = np.empty((0, 0))
    rest_state_mean = 0
    pub_force_detection.publish(False)

    while not rospy.is_shutdown(): # This is the data acquisition cycle

        if not sequential_actions:
            st = time.time()
            while not rospy.is_shutdown(): # This is the calibration cycle
                print("Calculating rest state variables...")
                list_calibration = []
                dic_offset_calibration = {"fx": [], "fy": [], "fz": [], "mx": [],
                                          "my": [], "mz": [], "j0": [], "j1": [],
                                          "j2": [], "j3": [], "j4": [], "j5": []}

                pub_class_cnn.publish("Calibrating")

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

                pub_class_cnn.publish("None")
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
            print("Aqui?")

        # try:
        if end_experiment:
            sequential_actions = False
            print("\nNot enough for prediction\n")
            pub_class_cnn.publish("None")
        else:
            sequential_actions = True
            vector_norm = normalize_data(vector_data, limit, trainning_config, clusters_max_min)

            x_sample = np.reshape(vector_norm, (1, limit, 13))
            x_sample = x_sample[:, :, 1:]

            predictions_cnn = model_cnn.predict(x=x_sample, verbose=2)
            predictions_transformer = model_transformer.predict(x=x_sample, verbose=2)

            labels = storage_config["action_classes"]
            max_idx_cnn = np.argmax(list(predictions_cnn))
            max_idx_transformer = np.argmax(list(predictions_transformer))
            predicted_label_cnn = labels[int(max_idx_cnn)]
            predicted_label_transformer = labels[int(max_idx_transformer)]

            # vector_data = np.append(vector_data, max_idx)
            pub_class_cnn.publish(predicted_label_cnn + " " + str(round(float(predictions_cnn[0][int(max_idx_cnn)] * 100), 1)) + "%")
            pub_class_transformer.publish(predicted_label_transformer + " " + str(round(float(predictions_transformer[0][int(max_idx_transformer)] * 100), 1)) + "%")
            print("-----------------------------------------------------------")
            print_tabulate(predicted_label_cnn, predictions_cnn)
            print("-----------------------------------------------------------")

            print("\n-----------------------------------------------------------")
            print_tabulate(predicted_label_transformer, predictions_transformer)
            print("-----------------------------------------------------------")
        # except:
        #     print("ctrl+C pressed")

    del data_for_learning
