#!/usr/bin/env python3
import argparse
import json
import math
import os
import time

from colorama import Fore
from matplotlib import pyplot as plt
from tensorflow import keras

from data_storage.src.data_aquisition_node import DataForLearning
import numpy as np
import rospy
from sklearn.preprocessing import normalize
from lib.src.ArmGripperComm import ArmGripperComm
from tabulate import tabulate
import pyfiglet
from config.definitions import ROOT_DIR


from config.definitions import ROOT_DIR
from config.definitions import NN_DIR


class ActionIntreperter:
    def __init__(self):

        self.data_for_learning = DataForLearning()
        while True:
            print(self.data_for_learning)
            time.sleep(0.5)

# def update_graph(data_vector):
#     for ax in ax:
#         ax.cla()
#         ax.grid()
#
#     # data_vector = data[idx, :]
#
#     data_array = np.reshape(data_vector, (measurements, int(len(data_vector) / measurements)))
#     graph_color = ["-r", "-g", "-b", "-y", "-k", "-m", "-r", "-g", "-b", "-r", "-g", "-b"]
#
#     joints_max = []
#     forces_max = []
#     torques_max = []
#
#     for i in range(0, 6):
#         joints_max.append(max(abs(data_array[:, i + 1])))
#         line = ax[0].plot(data_array[:, 0], data_array[:, i + 1], graph_color[i])
#         lines.append(line[0])
#
#     joints_y_lim = math.ceil(1.1 * max(joints_max))
#
#     ax[0].set_title("Joints efforts")
#     ax[0].legend(lines[0:6], ["J0", "J1", "J2", "J3", "J4", "J5"])
#     ax[0].set_ylim((-joints_y_lim, joints_y_lim))
#
#     for i in range(6, 9):
#         forces_max.append(max(abs(data_array[:, i + 1])))
#         line = ax[1].plot(data_array[:, 0], data_array[:, i + 1], graph_color[i])
#         lines.append(line[0])
#
#     forces_y_lim = math.ceil(1.1 * max(forces_max))
#
#     ax[1].set_title("Gripper Forces")
#     ax[1].legend(lines[6:9], ["Fx", "Fy", "Fz"])
#     ax[1].set_ylim((-forces_y_lim, forces_y_lim))
#
#     for i in range(9, 12):
#         torques_max.append(max(abs(data_array[:, i + 1])))
#         line = ax[2].plot(data_array[:, 0], data_array[:, i + 1], graph_color[i])
#         lines.append(line[0])
#
#     torques_y_lim = math.ceil(1.1 * max(torques_max))
#
#     ax[2].set_title("Gripper Moments")
#     ax[2].legend(lines[9:12], ["Mx", "My", "Mz"])
#     ax[2].set_ylim((-torques_y_lim, torques_y_lim))
#
#     true_classification = str(config["action_classes"][int(classifications[idx])])
#
#     if true_classification == prediction:
#         outcome = "Accurate"
#     else:
#         outcome = "Fail"
#
#     fig.suptitle("Predicted=" + prediction + ", True=" +
#                       true_classification + ", " + outcome +
#                       ", " + str(idx) + " => [" + str(round(outputs[idx][0], 3)) + ", " +
#                       str(round(outputs[idx][1], 3)) + ", " + str(round(outputs[idx][2], 3))
#                       + ", " + str(round(outputs[idx][3], 3)) + "]", fontsize=20)
#
#     is_graph_outdated = False
#
#
# def print_tabulate(real_time_predictions, config):
#
#     labels = config["action_classes"]
#     max_idx = np.argmax(real_time_predictions)
#
#     result = pyfiglet.figlet_format(labels[max_idx], font="space_op", width=500)
#
#     print(Fore.LIGHTBLUE_EX + result + Fore.RESET)
#
#     for pred in list(real_time_predictions):
#         data = [['Output', pred[0], pred[1], pred[2], pred[3]]]
#         print(tabulate(list(data), headers=[" ", "PULL", "PUSH", "SHAKE", "TWIST"], tablefmt="fancy_grid"))
#         print("\n")
#
#
# def normalize_data(vector, measurements, train_config):
#
#     data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
#     data_array_norm = np.empty((data_array.shape[0], 0))
#
#     idx = 0
#     for n in train_config["normalization_clusters"]:
#         data_sub_array = data_array[:, idx:idx + n]
#         idx += n
#
#         data_max = abs(max(data_sub_array.min(), data_sub_array.max(), key=abs))
#
#         data_sub_array_norm = data_sub_array / data_max
#         data_array_norm = np.hstack((data_array_norm, data_sub_array_norm))
#
#     vector_data_norm = np.reshape(data_array_norm, (1, vector.shape[0]))
#
#     # data_array = np.reshape(vector, (measurements, int(len(vector) / measurements)))
#     # experiment_array_norm = normalize(data_array, axis=0, norm='max')
#     #
#     # vector_data_norm = np.reshape(experiment_array_norm, (1, vector.shape[0]))
#
#     return vector_data_norm
#
#
# # def add_to_vector(data, vector, first_timestamp, list_idx):
# def add_to_vector(data, vector, first_timestamp):
#
#     if first_time_stamp is None:
#         first_timestamp = data.timestamp()
#         timestamp = 0.0
#     else:
#         timestamp = data.timestamp() - first_timestamp
#
#     new_data = [timestamp, data.joints_effort[0], data.joints_effort[1], data.joints_effort[2],
#                 data.joints_effort[3], data.joints_effort[4], data.joints_effort[5],
#                 data.wrench_force_torque.force.x, data.wrench_force_torque.force.y,
#                 data.wrench_force_torque.force.z, data.wrench_force_torque.torque.x,
#                 data.wrench_force_torque.torque.y, data.wrench_force_torque.torque.z]
#
#     return np.append(vector, new_data), first_timestamp
#
#
# def calc_data_mean(data):
#     values = np.array([data.wrench_force_torque.force.z/10, data.wrench_force_torque.torque.x,
#                        data.wrench_force_torque.torque.y, data.wrench_force_torque.torque.z])
#
#     return np.mean(values)
#
#
if __name__ == '__main__':

    # ---------------------------------------------------------------------------------------------
    # --------------------------------------INPUT VARIABLES----------------------------------------
    # ---------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description="Arguments for trainning script")
    parser.add_argument("-ag", "--activate_gripper", action="store_true",
                        help="If argmument is present, activates gripper")
    parser.add_argument("-maip", "--move_arm_to_inicial_position", action="store_true",
                        help="If argmument is present, activates gripper")
    parser.add_argument("-c", "--config_file", type=str, default="data_storage_config",
                        help="If argmument is present, activates gripper")
    parser.add_argument("-gui", "--gui_active", action="store_true", default=False,
                        help="If argmument is present, activates gripper")

    args = vars(parser.parse_args())

    f = open(ROOT_DIR + '/data_storage/config/' + args["config_file"] + '.json')

    storage_config = json.load(f)

    f.close()

    f = open(ROOT_DIR + '/data_storage/config/training_config.json')

    trainning_config = json.load(f)

    f.close()

    model = keras.models.load_model(NN_DIR + "/keras/myModel")

    # ---------------------------------------------------------------------------------------------
    # -------------------------------INITIATE COMMUNICATION----------------------------------------
    # ---------------------------------------------------------------------------------------------

    rospy.init_node("action_intreperter", anonymous=True)
#
#     data_for_learning = DataForLearning()
#     arm_gripper_comm = ArmGripperComm()
#
#     rate = rospy.Rate(storage_config["rate"])
#
#     time.sleep(0.2) # Waiting time to ros nodes properly initiate
#
#     # ---------------------------------------------------------------------------------------------
#     # -------------------------------INITIATE ROBOT------------------------------------------------
#     # ---------------------------------------------------------------------------------------------
#
#     try:
#         if args["move_arm_to_inicial_position"]:
#             arm_gripper_comm.move_arm_to_initial_pose()
#
#         if args["activate_gripper"]:
#             input("Press ENTER to activate gripper in 3 secs")
#             for i in range(0, 3):
#                 print(i + 1)
#                 time.sleep(1)
#
#             arm_gripper_comm.gripper_init()
#             time.sleep(1.5)
#
#             arm_gripper_comm.gripper_close_fast()
#             time.sleep(0.5)
#
#             arm_gripper_comm.gripper_disconnect()
#     except:
#         print("ctrl+C pressed")
#
#     list_calibration = []
#
#     print("Calculating rest state variables...")
#
#     for i in range(0, 99):
#         list_calibration.append(calc_data_mean(data_for_learning))
#         time.sleep(0.01)
#
#     rest_state_mean = np.mean(np.array(list_calibration))
#
#     limit = int(storage_config["time"] * storage_config["rate"])
#
#     trainning_data_array = np.empty((0, limit * len(storage_config["data"])))
#
#     sequential_actions = False
#
#     if args["gui_active"]:
#         plt.ion()
#         fig, axs = plt.subplots(3, 1)
#         lines = []
#         plt.show()
#
#     while not rospy.is_shutdown():
#
#         if not sequential_actions:
#             print(f"Waiting for action to initiate prediction ...")
#
#         while not rospy.is_shutdown():
#             data_mean = calc_data_mean(data_for_learning)
#             variance = data_mean - rest_state_mean
#
#             if abs(variance) > storage_config["force_threshold_start"]:
#                 break
#
#             time.sleep(0.1)
#
#         time.sleep(storage_config["waiting_offset"]) # time waiting to initiate the experiment
#
#         # ---------------------------------------------------------------------------------------------
#         # -------------------------------------GET DATA------------------------------------------------
#         # ---------------------------------------------------------------------------------------------
#
#         end_experiment = False
#         first_time_stamp = None
#         vector_data = np.empty((0, 0))
#
#         i = 0
#         treshold_counter = 0
#
#         rate.sleep()  # The first time rate sleep was used it was giving problems (would not wait the right amout of time)
#
#         try:
#             while not rospy.is_shutdown() and i < limit:
#
#                 i += 1
#                 # print(data_for_learning)
#                 vector_data, first_time_stamp = add_to_vector(data_for_learning, vector_data, first_time_stamp)
#                 # vector_data, first_time_stamp = add_to_vector(data_for_learning, vector_data, first_time_stamp,
#                 #                                               list_filter_idx)
#
#                 data_mean = calc_data_mean(data_for_learning)
#                 variance = data_mean - rest_state_mean
#
#                 if abs(variance) < storage_config["force_threshold_end"]:
#                     treshold_counter += 1
#                     if treshold_counter >= storage_config["threshold_counter_limit"]:
#                         end_experiment = True
#                         break
#                 else:
#                     treshold_counter = 0
#
#                 rate.sleep()
#         except:
#             print("ctrl+C pressed")
#
#         if end_experiment:
#             sequential_actions = False
#             print("\nNot enough for prediction\n")
#         else:
#             sequential_actions = True
#             vector_norm = normalize_data(vector_data, limit, trainning_config)
#             print("-----------------------------------------------------------")
#             predictions = model.predict(x=vector_norm, verbose=2)
#             print_tabulate(list(predictions), storage_config)
#             print("-----------------------------------------------------------")
#
#     del data_for_learning, arm_gripper_comm
